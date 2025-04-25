import csv
import warnings
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from torchvision.datasets import ImageFolder
from timm.models import load_checkpoint, resume_checkpoint
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import random
import traceback
from tqdm import tqdm
import timm

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom ImageFolder to skip missing files
class RobustImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.samples = [(path, cls) for path, cls in self.samples if os.path.isfile(path)]
        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found in {root}")

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving models ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

# Validate dataset and list files
def validate_dataset(train_dir, val_dir, class_names):
    print("Validating dataset...")
    total_images = 0
    for split_dir in [train_dir, val_dir]:
        split_name = "train" if split_dir == train_dir else "val"
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist")
        for class_name in class_names:
            class_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} does not exist")
                continue
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"{split_name}/{class_name}: {len(files)} images found")
            total_images += len(files)
            for f in files:
                file_path = os.path.join(class_path, f)
                if not os.path.isfile(file_path):
                    print(f"Error: File {file_path} is inaccessible")
                try:
                    Image.open(file_path).verify()
                except Exception as e:
                    print(f"Error: Corrupted image {file_path}: {e}")
    if total_images == 0:
        raise ValueError("No valid images found in dataset")
    print(f"Dataset validation complete: {total_images} images found.")

# Select a random test image from val folder
def select_random_test_image(val_dir):
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_paths = []
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(class_path, f))
    if not image_paths:
        print(f"Warning: No valid images found in {val_dir}")
        return None
    return random.choice(image_paths)

# Inference function
def predict_image(model, image_path, transform, device, class_names):
    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading test image {image_path}: {e}")
        return None, None
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class_idx].item()

    return class_names[pred_class_idx], pred_prob

# Setup logging to file and console
def setup_logging(variant, output_dir):
    log_file = os.path.join(output_dir, 'logs', f'training_{variant}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger

# Function to plot and save metrics
def plot_metrics(history, output_dir, metric_name, variant):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'][metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(history['val'][metric_name], label=f'Validation {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'plots', f'{metric_name}_plot_{variant}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

# Function to plot and save ROC curves
def plot_roc_curves(y_true, y_score, num_classes, class_names, output_dir, variant):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Best Model)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'plots', f'roc_curve_{variant}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return roc_auc

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, variant):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Best Model)')
    plot_path = os.path.join(output_dir, 'plots', f'confusion_matrix_{variant}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

# Compute models parameters
def log_model_stats(model, model_name, logger):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model {model_name}: {params:.2f}M parameters")

# Argument parser setup
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='MaxViT Training for Tomato Leaf Disease')

# Dataset / Model parameters
parser.add_argument('--data', default='dataset\\tomato_leaf_dataset', metavar='DIR',
                    help='path to tomato leaf dataset')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version from models/ folder')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize models from this checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full models and optimizer state')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state')
parser.add_argument('--num-classes', type=int, default=11, metavar='N',
                    help='number of label classes')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                    help='epochs to cooldown LR')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use Native Torch AMP')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='outputs_maxvit', type=str, metavar='PATH',
                    help='path to output folder')
parser.add_argument('--eval-metric', default='accuracy', type=str, metavar='EVAL_METRIC',
                    help='Best metric')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--test_image', type=str,
                    default="C:/Users/nisha/OneDrive/Desktop/CSWin-Transformer/dataset/tomato_leaf_dataset/val",
                    help="Path to the test image or val directory for random selection")

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, mixup_fn=None, history=None, class_names=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    f1_m = AverageMeter()
    prec_m = AverageMeter()

    model.train()

    end = time.time()
    num_updates = epoch * len(loader)
    all_preds = []
    all_targets = []

    with tqdm(total=len(loader), desc=f'Training Epoch {epoch+1}/{args.epochs}', leave=False) as pbar:
        for batch_idx, (input, target) in enumerate(loader):
            data_time_m.update(time.time() - end)
            if not args.prefetcher:
                input, target = input.cuda(), target.cuda()
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)

            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

            if not args.distributed:
                losses_m.update(loss.item(), input.size(0))
                acc1 = accuracy(output, target, topk=(1,))[0]
                acc_m.update(acc1.item(), output.size(0))
                preds = torch.argmax(output, dim=1).cpu().numpy()
                targets = target.cpu().numpy()
                f1 = f1_score(targets, preds, average='weighted', zero_division=0)
                prec = precision_score(targets, preds, average='weighted', zero_division=0)
                f1_m.update(f1, input.size(0))
                prec_m.update(prec, input.size(0))
                all_preds.extend(preds)
                all_targets.extend(targets)

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
            else:
                loss.backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            pbar.update(1)

    if args.distributed:
        losses_m = reduce_tensor(losses_m.avg, args.world_size)
        acc_m = reduce_tensor(acc_m.avg, args.world_size)
        f1_m = reduce_tensor(f1_m.avg, args.world_size)
        prec_m = reduce_tensor(prec_m.avg, args.world_size)

    if history is not None:
        history['train']['loss'].append(losses_m.avg)
        history['train']['accuracy'].append(acc_m.avg)
        history['train']['f1'].append(f1_m.avg)
        history['train']['precision'].append(prec_m.avg)

    return OrderedDict([('loss', losses_m.avg), ('accuracy', acc_m.avg), ('f1', f1_m.avg), ('precision', prec_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, output_dir='', epoch=None, history=None, class_names=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    f1_m = AverageMeter()
    prec_m = AverageMeter()

    model.eval()

    all_preds = []
    all_targets = []
    all_scores = []

    end = time.time()
    with tqdm(total=len(loader), desc='Validation', leave=False) as pbar:
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = loss_fn(output, target)
                acc1 = accuracy(output, target, topk=(1,))[0]

                preds = torch.argmax(output, dim=1).cpu().numpy()
                targets = target.cpu().numpy()
                scores = torch.softmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
                all_scores.extend(scores)

                f1 = f1_score(targets, preds, average='weighted', zero_division=0)
                prec = precision_score(targets, preds, average='weighted', zero_division=0)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                acc_m.update(acc1.item(), output.size(0))
                f1_m.update(f1, input.size(0))
                prec_m.update(prec, input.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                pbar.update(1)

    if history is not None:
        history['val']['loss'].append(losses_m.avg)
        history['val']['accuracy'].append(acc_m.avg)
        history['val']['f1'].append(f1_m.avg)
        history['val']['precision'].append(prec_m.avg)

    return (OrderedDict([('loss', losses_m.avg), ('accuracy', acc_m.avg), ('f1', f1_m.avg), ('precision', prec_m.avg)]),
            all_preds, all_targets, all_scores)

# Modified update_summary to save as DataFrame
def update_summary(epoch, train_metrics, eval_metrics, output_dir, variant, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    epoch_summaries = [rowd]
    epoch_df = pd.DataFrame(epoch_summaries)
    csv_path = os.path.join(output_dir, 'csv', f'epoch_summary_{variant}.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    mode = 'w' if write_header else 'a'
    epoch_df.to_csv(csv_path, mode=mode, index=False, header=write_header)

def main():
    setup_default_logging()
    args, args_text = _parse_args()

    # Set random seed
    set_seed(args.seed)

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning('Using more than one GPU per process in distributed mode is not allowed.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process.')
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    output_dir = args.output
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    variants = [
        ('tiny', 'maxvit_tiny_tf_224'),
        # ('small', 'maxvit_small_tf_224'),
        # ('base', 'maxvit_base_tf_224')
    ]

    global class_names
    class_names = [
        "Bacterial_spot",
        "Early_blight",
        "healthy",
        "Late_blight",
        "Leaf_Mold",
        "powdery_mildew",
        "Septoria_leaf_spot",
        "Spider_mites Two-spotted_spider_mite",
        "Target_Spot",
        "Tomato_mosaic_virus",
        "Tomato_Yellow_Leaf_Curl_Virus"
    ]

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    try:
        validate_dataset(train_dir, val_dir, class_names)
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        traceback.print_exc()
        return

    for variant, model_name in variants:
        logger = setup_logging(variant, output_dir)
        logger.info(f"Starting training for maxvit_{variant}")

        # Initialize models
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=args.num_classes,
            img_size=args.img_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )
        if args.pretrained:
            checkpoint_path = os.path.join('models', f'maxvit_{variant}_tf_224.pth')
            if not os.path.exists(checkpoint_path):
                logger.error(f"Pretrained checkpoint not found at: {checkpoint_path}")
                continue
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Successfully loaded pretrained weights from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading pretrained checkpoint {checkpoint_path}: {e}")
                continue

        log_model_stats(model, f'maxvit_{variant}', logger)

        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

        use_amp = None
        if args.amp and has_native_amp:
            use_amp = 'native'
            logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            logger.info('AMP not enabled. Training in float32.')

        if args.num_gpu > 1:
            model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

        optimizer = create_optimizer(args, model)

        amp_autocast = suppress
        loss_scaler = None
        if use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        resume_epoch = None
        if args.resume:
            resume_epoch = resume_checkpoint(
                model, args.resume,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=args.local_rank == 0)

        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        start_epoch = resume_epoch if resume_epoch is not None else 0
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)

        # Load datasets
        if not os.path.exists(train_dir):
            logger.error('Training folder does not exist at: {}'.format(train_dir))
            continue
        dataset_train = RobustImageFolder(train_dir)
        logger.info(f"Loaded {len(dataset_train)} training samples with classes: {dataset_train.classes}")

        if not os.path.exists(val_dir):
            logger.error('Validation folder does not exist at: {}'.format(val_dir))
            continue
        dataset_eval = RobustImageFolder(val_dir)
        logger.info(f"Loaded {len(dataset_eval)} validation samples with classes: {dataset_eval.classes}")

        collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                prob=1.0, switch_prob=0.5, mode='batch',
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size_multiplier * args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem
        )

        if mixup_active:
            train_loss_fn = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

        eval_metric = args.eval_metric
        best_metric = None
        best_epoch = None

        history = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'precision': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'precision': []}
        }

        early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path=os.path.join(output_dir, 'models', f'maxvit_{variant}_best.pth')
        )
        os.makedirs(os.path.dirname(early_stopping.path), exist_ok=True)

        if args.eval_checkpoint:
            load_checkpoint(model, args.eval_checkpoint)
            val_metrics, _, _, _ = validate(model, loader_eval, validate_loss_fn, args, class_names=class_names,
                                            output_dir=output_dir)
            print(f"Accuracy of the models is: {val_metrics['accuracy']:.1f}%")
            continue

        try:
            for epoch in range(start_epoch, num_epochs):
                if args.distributed:
                    loader_train.sampler.set_epoch(epoch)

                train_metrics = train_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,
                    lr_scheduler=lr_scheduler, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, mixup_fn=mixup_fn,
                    history=history, class_names=class_names)

                val_metrics, all_preds, all_targets, all_scores = validate(
                    model, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, output_dir=output_dir, epoch=epoch,
                    history=history, class_names=class_names)

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch + 1, val_metrics[eval_metric])

                early_stopping(val_metrics['loss'], model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break

                if args.local_rank == 0:
                    logger.info(
                        'Epoch {}: Train Loss: {:.4f}, Train Acc: {:.2f}%, Train F1: {:.2f}, Train Prec: {:.2f}, '
                        'Val Loss: {:.4f}, Val Acc: {:.2f}%, Val F1: {:.2f}, Val Prec: {:.2f}'.format(
                            epoch, train_metrics['loss'], train_metrics['accuracy'], train_metrics['f1'], train_metrics['precision'],
                            val_metrics['loss'], val_metrics['accuracy'], val_metrics['f1'], val_metrics['precision']))
                    update_summary(
                        epoch, train_metrics, val_metrics, output_dir, variant,
                        write_header=best_metric is None)
                    if best_metric is None or val_metrics[eval_metric] > best_metric:
                        best_metric = val_metrics[eval_metric]
                        best_epoch = epoch
                        plot_confusion_matrix(all_targets, all_preds, class_names, output_dir, variant)
                        plot_roc_curves(all_targets, np.array(all_scores), args.num_classes, class_names, output_dir, variant)

                    for metric in ['loss', 'accuracy', 'f1', 'precision']:
                        plot_metrics(history, output_dir, metric, variant)

        except KeyboardInterrupt:
            pass

        if best_metric is not None:
            logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

        # Load best models
        model.load_state_dict(torch.load(early_stopping.path))

        # Save final models
        try:
            final_model_path = os.path.join(output_dir, 'models', f'maxvit_{variant}_final.pth')
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(model.state_dict(), final_model_path)
        except Exception as e:
            logger.error(f"Error saving final models for maxvit_{variant}: {e}")
            traceback.print_exc()

        # Inference on random test image
        test_image_path = args.test_image
        if os.path.isdir(test_image_path):
            test_image_path = select_random_test_image(test_image_path)
            if test_image_path is None:
                logger.warning("No test image selected, skipping inference")
            else:
                logger.info(f"Selected test image: {test_image_path}")

        if test_image_path:
            transform = create_loader(
                dataset_eval,
                input_size=data_config['input_size'],
                batch_size=1,
                is_training=False,
                use_prefetcher=args.prefetcher,
                interpolation=data_config['interpolation'],
                mean=data_config['mean'],
                std=data_config['std'],
                num_workers=0,
                distributed=False,
                crop_pct=data_config['crop_pct'],
                pin_memory=args.pin_mem
            ).dataset.transform
            pred_class, pred_prob = predict_image(model, test_image_path, transform, torch.device(args.device), class_names)
            if pred_class is not None:
                logger.info(f"Predicted class: {pred_class}, Probability: {pred_prob:.4f}")

        logger.info(f"Completed training for maxvit_{variant}")

if __name__ == '__main__':
    main()
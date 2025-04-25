import argparse
import os
import sys
import time
import logging
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import AverageMeter, setup_default_logging, random_seed
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
import yaml
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

warnings.filterwarnings('ignore', category=FutureWarning, module='timm.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'models'))

try:
    from src.cct import cct_7_7x2_224, cct_14_7x2_224
except ImportError as e:
    raise ImportError(f"Failed to import src.cct: {e}")

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

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='PyTorch Tomato Leaf Dataset Training')
parser.add_argument('--data-dir', default='dataset/tomato_leaf_dataset', type=str)
parser.add_argument('--model', default='cct_14_7x2_224', type=str)  # Default kept for compatibility, but overridden
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--num-classes', type=int, default=11)
parser.add_argument('--img-size', type=int, default=224)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight-decay', type=float, default=0.05)
parser.add_argument('--sched', default='cosine', type=str)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup-epochs', type=int, default=5)
parser.add_argument('--min-lr', type=float, default=1e-6)
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--pin-mem', action='store_true')
parser.add_argument('--output', default='outputs_cct', type=str)
parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log-interval', type=int, default=50)
parser.add_argument('--patience', type=int, default=10)

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.invalid_files = []

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            self.invalid_files.append(path)
            dummy = torch.zeros(3, 224, 224) if self.transform is None else self.transform(Image.new('RGB', (224, 224)))
            return dummy, -1

    def __len__(self):
        return len(self.samples)

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
                tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def load_pretrained_weights(model, model_name, model_dir='models'):
    weight_path = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    _logger.info(f"Loaded pretrained weights from {weight_path}")
    return model

def validate_dataset(train_dir, val_dir, class_names):
    total_images = 0
    for split_dir in [train_dir, val_dir]:
        split_name = "train" if split_dir == train_dir else "val"
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist")
        for class_name in class_names:
            class_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_path):
                continue
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            total_images += len(files)
    if total_images == 0:
        raise ValueError("No valid images found in dataset")

def compute_metrics(outputs, targets):
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted', zero_division=0)
    acc = 100.0 * (preds == targets).mean()
    return cm, acc, precision, recall, f1

def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_precisions, train_recalls, train_f1s,
                 val_precisions, val_recalls, val_f1s, val_labels, val_preds, model_name, output_dir):
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'loss_plot_{model_name}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'accuracy_plot_{model_name}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_precisions, 'b-', label='Val Precision')
    plt.plot(epochs, val_recalls, 'r-', label='Val Recall')
    plt.plot(epochs, val_f1s, 'g-', label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'prf1_plot_{model_name}.png'))
    plt.close()

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, scaler):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc=f'Train Epoch {epoch}', total=len(loader), leave=True, dynamic_ncols=True)

    for batch_idx, (input, target) in enumerate(progress_bar):
        valid_mask = target != -1
        if not valid_mask.any():
            continue
        input, target = input[valid_mask].to(args.device), target[valid_mask].to(args.device)

        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_m.update(loss.item(), input.size(0))
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        torch.cuda.synchronize()
        batch_time_m.update(time.time() - end)

        if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f'{losses_m.avg:.4f}', 'lr': f'{lr:.3e}'}, refresh=True)
            _logger.info(f'Train: {epoch} [{batch_idx:>4d}/{last_idx}] Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f}) Time: {batch_time_m.val:.3f}s LR: {lr:.3e}')

        end = time.time()

    progress_bar.close()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean()
    return OrderedDict([('loss', losses_m.avg), ('accuracy', acc), ('precision', precision), ('recall', recall), ('f1', f1)])

def validate(model, loader, loss_fn, args, output_dir=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    all_outputs, all_targets = [], []
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc='Validate', total=len(loader), leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(progress_bar):
            valid_mask = target != -1
            if not valid_mask.any():
                continue
            input, target = input[valid_mask].to(args.device), target[valid_mask].to(args.device)
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)
            losses_m.update(loss.item(), input.size(0))
            all_outputs.append(output)
            all_targets.append(target)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
                progress_bar.set_postfix({'loss': f'{losses_m.avg:.4f}'}, refresh=True)
                _logger.info(f'Test: [{batch_idx:>4d}/{last_idx}] Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f})')

            end = time.time()

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    cm, acc, precision, recall, f1 = compute_metrics(all_outputs, all_targets)

    progress_bar.close()
    return OrderedDict([('loss', losses_m.avg), ('accuracy', acc), ('precision', precision), ('recall', recall), ('f1', f1)]), all_preds, all_labels

def train_model(model_name, args, args_text, train_dir, val_dir, train_dataset, val_dataset):
    args.model = model_name  # Update model name in args
    tqdm.write(f'\nStarting training for model: {model_name}')

    # Create model
    if model_name == 'cct_7_7x2_224':
        model = cct_7_7x2_224(pretrained=False, progress=False, img_size=args.img_size, positional_embedding='learnable', num_classes=args.num_classes)
    elif model_name == 'cct_14_7x2_224':
        model = cct_14_7x2_224(pretrained=False, progress=False, img_size=args.img_size, positional_embedding='learnable', num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = load_pretrained_weights(model, model_name, model_dir='models')
    model = model.to(args.device)
    param_count = sum([m.numel() for m in model.parameters()])
    tqdm.write(f'Model {model_name} created, param count: {param_count}')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        tqdm.write(f"Resumed from checkpoint {args.resume}")

    # Create data loaders
    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem)
    loader_eval = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_mem)

    # Setup optimizer, scheduler, and loss
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_args = argparse.Namespace(sched=args.sched, epochs=args.epochs, lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=0, cooldown_epochs=0, lr_cycle_limit=1, decay_rate=1)
    lr_scheduler, num_epochs = create_scheduler(scheduler_args, optimizer)
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(args.device)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)
    scaler = GradScaler()

    # Output directory for this model
    exp_name = args.experiment if args.experiment else f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(models_dir, f'{model_name}_best.pth'))
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    train_precisions, train_recalls, train_f1s = [], [], []
    val_precisions, val_recalls, val_f1s = [], [], []
    epoch_summaries = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args, scaler)
        eval_metrics, val_preds, val_labels = validate(model, loader_eval, validate_loss_fn, args, output_dir=output_dir)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics['accuracy'])

        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['accuracy'])
        train_precisions.append(train_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        train_f1s.append(train_metrics['f1'])
        val_losses.append(eval_metrics['loss'])
        val_accs.append(eval_metrics['accuracy'])
        val_precisions.append(eval_metrics['precision'])
        val_recalls.append(eval_metrics['recall'])
        val_f1s.append(eval_metrics['f1'])

        epoch_summary = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': eval_metrics['loss'],
            'val_accuracy': eval_metrics['accuracy'],
            'val_precision': eval_metrics['precision'],
            'val_recall': eval_metrics['recall'],
            'val_f1': eval_metrics['f1']
        }
        epoch_summaries.append(epoch_summary)
        tqdm.write(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                   f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}, Train F1: {train_metrics['f1']:.4f}, "
                   f"Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.2f}%, "
                   f"Val Precision: {eval_metrics['precision']:.4f}, Val Recall: {eval_metrics['recall']:.4f}, Val F1: {eval_metrics['f1']:.4f}")

        early_stopping(eval_metrics['loss'], model)
        if early_stopping.early_stop:
            tqdm.write("Early stopping triggered")
            break

    # Save final model and summaries
    model.load_state_dict(torch.load(os.path.join(models_dir, f'{model_name}_best.pth')))
    torch.save(model.state_dict(), os.path.join(models_dir, f'{model_name}_final.pth'))
    epoch_df = pd.DataFrame(epoch_summaries)
    csv_path = os.path.join(summaries_dir, f'epoch_summary_{model_name}.csv')
    epoch_df.to_csv(csv_path, index=False)
    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_precisions, train_recalls, train_f1s,
                 val_precisions, val_recalls, val_f1s, val_labels, val_preds, model_name, output_dir)

def main():
    setup_default_logging(log_path=os.path.join(project_root, 'train.log'))
    args, args_text = _parse_args()

    random_seed(args.seed, 0)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tqdm.write(f'Training with a single process on 1 {args.device}.')

    # Define models to train
    models_to_train = [
        'cct_7_7x2_224',
                       # 'cct_14_7x2_224'
    ]

    # Validate dataset
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    validate_dataset(train_dir, val_dir, class_names)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = SafeImageFolder(train_dir, transform=transform)
    val_dataset = SafeImageFolder(val_dir, transform=val_transform)

    if train_dataset.invalid_files:
        tqdm.write(f"Warning: {len(train_dataset.invalid_files)} invalid files in train dataset")
    if val_dataset.invalid_files:
        tqdm.write(f"Warning: {len(val_dataset.invalid_files)} invalid files in val dataset")

    # Train each model
    for model_name in models_to_train:
        train_model(model_name, args, args_text, train_dir, val_dir, train_dataset, val_dataset)

if __name__ == '__main__':
    main()
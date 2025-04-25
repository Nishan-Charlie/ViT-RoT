import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import MobileViTForImageClassification, MobileViTImageProcessor

from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

import traceback
import logging
from torch.cuda.amp import autocast, GradScaler

# Custom ImageFolder to handle missing files
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
            print(f"Error loading {path}: {e}")
            self.invalid_files.append(path)
            dummy = torch.zeros(3, 224, 224) if self.transform is None else self.transform(Image.new('RGB', (224, 224)))
            return dummy, -1  # -1 indicates invalid sample

    def __len__(self):
        return len(self.samples)

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
            print(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model ...')
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

# Training function with metrics, early stopping, and mixed precision
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping, variant, output_dir, scaler):
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precisions, train_recalls, train_f1s = [], [], []
    val_precisions, val_recalls, val_f1s = [], [], []
    epoch_summaries = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            images, labels = images[valid_mask].to(device), labels[valid_mask].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / total if total > 0 else float('inf')
        train_acc = 100 * correct / total if total > 0 else 0
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                valid_mask = labels != -1
                if not valid_mask.any():
                    continue
                images, labels = images[valid_mask].to(device), labels[valid_mask].to(device)
                with autocast():
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        epoch_summary = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        epoch_summaries.append(epoch_summary)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save epoch summaries to CSV
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    epoch_df = pd.DataFrame(epoch_summaries)
    csv_path = os.path.join(summaries_dir, f'epoch_summary_{variant}.csv')
    epoch_df.to_csv(csv_path, index=False)
    print(f"Saved epoch summaries to {csv_path}")

    # Load best models weights
    model.load_state_dict(torch.load(early_stopping.path))

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 train_precisions, train_recalls, train_f1s,
                 val_precisions, val_recalls, val_f1s,
                 val_labels, val_preds, variant, output_dir)

    return model

# Plotting function with models name in confusion matrix
def plot_metrics(train_losses, val_losses, train_accs, val_accs,
                 train_precisions, train_recalls, train_f1s,
                 val_precisions, val_recalls, val_f1s,
                 val_labels, val_preds, variant, output_dir):
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'loss_plot_{variant}.png'))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'accuracy_plot_{variant}.png'))
    plt.close()

    # Precision/Recall/F1 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_precisions, 'b-', label='Val Precision')
    plt.plot(epochs, val_recalls, 'r-', label='Val Recall')
    plt.plot(epochs, val_f1s, 'g-', label='Val F1')
    plt.title('Validation Precision, Recall, and F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'prf1_plot_{variant}.png'))
    plt.close()

    # Confusion matrix with models name
    model_display_name = {
        'xx_small': 'MobileViT-XX-Small',
        'x_small': 'MobileViT-X-Small',
        'small': 'MobileViT-Small'
    }.get(variant, 'MobileViT')
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_display_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{variant}.png'))
    plt.close()

# Inference function
def predict_image(model, image_path, processor, device, class_names):
    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading test image {image_path}: {e}")
        return None, None
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        with autocast():
            outputs = model(**inputs).logits
            probs = torch.softmax(outputs, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class_idx].item()

    return class_names[pred_class_idx], pred_prob

# Setup logging to file and console
def setup_logging(variant, output_dir):
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f'training_{variant}.log')
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

# Log models parameters
def log_model_stats(model, model_name, logger):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model {model_name}: {params:.2f}M parameters")

# Main execution
def main(args):
    output_dir = 'outputs_mobilevit'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    variants = [
        ('xx_small', 'apple/mobilevit-xx-small'),
        ('x_small', 'apple/mobilevit-x-small'),
        ('small', 'apple/mobilevit-small')
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 11
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

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

    try:
        validate_dataset(train_dir, val_dir, class_names)
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        traceback.print_exc()
        return

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

    try:
        train_dataset = SafeImageFolder(train_dir, transform=transform)
        val_dataset = SafeImageFolder(val_dir, transform=val_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return

    if train_dataset.invalid_files:
        print(f"Warning: {len(train_dataset.invalid_files)} invalid files in train dataset: {train_dataset.invalid_files[:5]}")
    if val_dataset.invalid_files:
        print(f"Warning: {len(val_dataset.invalid_files)} invalid files in val dataset: {val_dataset.invalid_files[:5]}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if sorted(train_dataset.classes) != sorted(class_names):
        print(f"Warning: Dataset classes {train_dataset.classes} differ from {class_names}")

    for variant, model_name in variants:
        print(f"\nStarting training for {model_name} (variant: {variant})")

        logger = setup_logging(variant, output_dir)
        logger.info(f"Starting training for {model_name}")

        try:
            processor = MobileViTImageProcessor.from_pretrained(model_name)
            model = MobileViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            log_model_stats(model, model_name, logger)
        except Exception as e:
            logger.error(f"Error loading model or processor for {model_name}: {e}")
            traceback.print_exc()
            continue

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scaler = GradScaler()
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(models_dir, f'mobilevit_{variant}_best.pth'))

        try:
            model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping, variant, output_dir, scaler)
        except Exception as e:
            logger.error(f"Error during training {model_name}: {e}")
            traceback.print_exc()
            continue

        try:
            torch.save(model.state_dict(), os.path.join(models_dir, f"mobilevit_{variant}_final.pth"))
        except Exception as e:
            logger.error(f"Error saving final model for {model_name}: {e}")
            traceback.print_exc()
            continue

        test_image_path = args.test_image
        pred_class, pred_prob = predict_image(model, test_image_path, processor, device, class_names)
        if pred_class is not None:
            logger.info(f"Predicted class: {pred_class}, Probability: {pred_prob:.4f}")

        logger.info(f"Completed training for {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileViT variants for tomato leaf disease classification")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument('--data_dir', type=str,
                        default="dataset/tomato_leaf_dataset",
                        help="Path to the dataset root directory")
    parser.add_argument('--test_image', type=str,
                        default="C:/Users/nisha/OneDrive/Desktop/test_image.jpg",
                        help="Path to the test image for inference")

    args = parser.parse_args()
    main(args)
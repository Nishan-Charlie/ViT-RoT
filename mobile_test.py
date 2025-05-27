#!/usr/bin/env python3
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import MobileViTForImageClassification
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
import random

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            dummy = self.transform(Image.new('RGB', (224, 224)))
            return dummy, -1  # -1 indicates invalid sample

    def __len__(self):
        return len(self.samples)

# Validate dataset
def validate_dataset(test_dir, class_names):
    print("Validating test dataset...")
    total_images = 0
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Directory {test_dir} does not exist")
    for class_name in class_names:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} does not exist")
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"test/{class_name}: {len(files)} images found")
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
        raise ValueError("No valid images found in test dataset")
    print(f"Test dataset validation complete: {total_images} images found.")

# Evaluate model on test set
def evaluate_model(model, test_loader, device, model_type, variant, output_dir, class_names):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_type}-{variant}"):
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            images, labels = images[valid_mask].to(device), labels[valid_mask].to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total if total > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Plot confusion matrix
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    model_display_name = {
        'mobilevit_xx_small': 'MobileViT-XX-Small',
        'mobilevit_x_small': 'MobileViT-X-Small',
        'mobilevit_small': 'MobileViT-Small'
    }.get(f"{model_type.lower()}_{variant}", f"{model_type}-{variant}")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12}, cbar=True)
    plt.title(f'Confusion Matrix for {model_display_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_type.lower()}_{variant}.png'), dpi=300)
    plt.close()

    # Save confusion matrix as CSV
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = 'True Label'
    cm_df.columns.name = 'Predicted Label'
    cm_csv_path = os.path.join(summaries_dir, f'confusion_matrix_{model_type.lower()}_{variant}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to {cm_csv_path}")

    # Plot class-wise accuracy for MobileViT-Small
    if variant == 'small':
        plot_classwise_accuracy(cm_csv_path, output_dir, class_names)

    return {
        'model_variant': f"{model_type}-{variant}",
        'model': model_type,
        'variant': variant,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Plot class-wise accuracy for MobileViT-Small
def plot_classwise_accuracy(csv_path, output_dir, class_names):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return

    try:
        cm_df = pd.read_csv(csv_path, index_col='True Label')
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if cm_df.shape != (11, 11):
        print(f"Error: Expected 11x11 matrix in {csv_path}, got {cm_df.shape}")
        return

    cm = cm_df.to_numpy()
    row_sums = cm.sum(axis=1)
    accuracies = np.zeros(len(class_names))
    for i in range(len(class_names)):
        accuracies[i] = (cm[i, i] / row_sums[i] * 100) if row_sums[i] > 0 else 0.0
    samples = row_sums

    # Format class names for plotting (replace underscores with spaces)
    display_class_names = [name.replace('_', ' ') for name in class_names]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(class_names))
    bar_width = 0.35
    ax1.bar(x - bar_width/2, samples, bar_width, color='forestgreen', alpha=0.7)
    ax1.set_xlabel('Classes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=14, fontweight='bold', color='forestgreen')
    ax1.tick_params(axis='y', labelcolor='forestgreen', labelsize=12, width=2, labelright=False, labelleft=True, which='both', direction='in')
    ax1.tick_params(axis='x', labelsize=12, width=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_class_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')

    ax2 = ax1.twinx()
    ax2.bar(x + bar_width/2, accuracies, bar_width, color='red', alpha=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=12, width=2, labelleft=False, labelright=True, which='both', direction='in')
    ax2.set_ylim(0, 100)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    plt.title('Class-wise Accuracy and Sample Counts for MobileViT-Small', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'classwise_accuracy_mobilevit_small.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class-wise accuracy plot to {output_path}")

    print("\nClass-wise Accuracies for MobileViT-Small:")
    for cls, acc in zip(display_class_names, accuracies):
        print(f"{cls}: {acc:.2f}%")

# Setup logging
def setup_logging(output_dir):
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'test_mobilevit_log.txt')
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

# Main execution
def main(args):
    set_seed(42)
    output_dir = 'outputs_test'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    logger = setup_logging(output_dir)
    logger.info("Starting MobileViT models evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 11
    data_dir = args.data_dir
    test_dir = os.path.join(data_dir, "val")

    class_names = [
        "Bacterial_spot", "Early_blight", "healthy", "Late_blight", "Leaf_Mold",
        "powdery_mildew", "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
        "Target_Spot", "Tomato_mosaic_virus", "Tomato_Yellow_Leaf_Curl_Virus"
    ]

    try:
        validate_dataset(test_dir, class_names)
    except Exception as e:
        logger.error(f"Test dataset validation failed: {e}")
        traceback.print_exc()
        return

    models = [
        {
            'type': 'MobileViT',
            'variants': [
                ('xx_small', 'apple/mobilevit-xx-small', 'outputs_mobilevit/models/mobilevit_xx_small_best.pth'),
                ('x_small', 'apple/mobilevit-x-small', 'outputs_mobilevit/models/mobilevit_x_small_best.pth'),
                ('small', 'apple/mobilevit-small', 'outputs_mobilevit/models/mobilevit_small_best.pth')
            ]
        }
    ]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    metrics_list = []

    for model_config in models:
        model_type = model_config['type']
        for variant, model_name, model_path in model_config['variants']:
            logger.info(f"Evaluating {model_type}-{variant} ({model_name})")
            try:
                model = MobileViTForImageClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                if not os.path.exists(model_path):
                    logger.warning(f"Model weights not found at {model_path}, skipping")
                    continue
                model.load_state_dict(torch.load(model_path))
                model.to(device)

                test_dataset = SafeImageFolder(test_dir, transform=transform)
                if test_dataset.invalid_files:
                    logger.warning(f"{len(test_dataset.invalid_files)} invalid files in test dataset: {test_dataset.invalid_files[:5]}")
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                metrics = evaluate_model(
                    model, test_loader, device, model_type, variant, output_dir, class_names
                )
                metrics_list.append(metrics)
                logger.info(f"{model_type}-{variant}: Accuracy: {metrics['accuracy']:.2f}%, "
                            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                            f"F1: {metrics['f1_score']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {model_type}-{variant}: {e}")
                traceback.print_exc()
                continue

    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(summaries_dir, 'test_metrics_mobilevit.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved test metrics to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MobileViT models on tomato leaf disease test dataset")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument('--data_dir', type=str,
                        default="dataset/tomato_leaf_dataset",
                        help="Path to the dataset root directory")
    args = parser.parse_args()
    main(args)
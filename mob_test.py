#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import MobileViTForImageClassification
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
from torch.cuda.amp import autocast
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Set random seed for reproducibility
def set_seed(seed=42):
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
            dummy = torch.zeros(3, 224, 224) if self.transform is None else self.transform(Image.new('RGB', (224, 224)))
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
        for f in tqdm(files, desc=f"Validating {class_name}"):
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

# Perform statistical significance tests
def perform_statistical_tests(model_correctness, model_names, summaries_dir, logger):
    """
    Perform Friedman test and Nemenyi post-hoc test on model performance.

    Args:
        model_correctness (dict): Dictionary mapping model variant to list of binary correctness (1=correct, 0=incorrect).
        model_names (list): List of model variant names (e.g., ['xx_small', 'x_small', 'small']).
        summaries_dir (str): Directory to save statistical test results.
        logger: Logger object for logging results.

    Returns:
        dict: Results of statistical tests (Friedman p-value and Nemenyi p-values).
    """
    try:
        # Verify correctness arrays have the same length
        lengths = [len(model_correctness[variant]) for variant in model_names]
        if len(set(lengths)) != 1:
            logger.error(f"Inconsistent correctness array lengths: {lengths}")
            return {'Friedman_p_value': None, 'Nemenyi_p_values': None}
        n_samples = lengths[0]
        logger.info(f"Performing statistical tests with {n_samples} samples")

        # Prepare data for Friedman test
        correctness_matrix = np.array([model_correctness[variant] for variant in model_names]).T  # Shape: (n_samples, n_models)

        # Friedman test
        stat, p_value = friedmanchisquare(*[correctness_matrix[:, i] for i in range(len(model_names))])
        logger.info(f"Friedman Test: chi-squared = {stat:.4f}, p-value = {p_value:.6f}")
        test_results = {'Friedman_p_value': p_value}

        # Nemenyi test if Friedman is significant
        if p_value < 0.05:
            logger.info("Friedman test significant, performing Nemenyi post-hoc test...")
            nemenyi_result = sp.posthoc_nemenyi_friedman(correctness_matrix)
            nemenyi_result.index = model_names
            nemenyi_result.columns = model_names
            logger.info("Nemenyi Post-Hoc Test p-values:\n" + str(nemenyi_result))

            # Save Nemenyi results to CSV
            nemenyi_csv_path = os.path.join(summaries_dir, 'nemenyi_test_p_values.csv')
            nemenyi_result.to_csv(nemenyi_csv_path)
            logger.info(f"Saved Nemenyi test results to {nemenyi_csv_path}")

            test_results['Nemenyi_p_values'] = nemenyi_result.to_dict()
        else:
            logger.info("Friedman test not significant, skipping Nemenyi test.")
            test_results['Nemenyi_p_values'] = None

        # Save test results to text file
        stats_file = os.path.join(summaries_dir, 'statistical_test_results.txt')
        with open(stats_file, 'w') as f:
            f.write("Statistical Significance Test Results\n")
            f.write("==================================\n")
            f.write(f"Friedman Test: chi-squared = {stat:.4f}, p-value = {p_value:.6f}\n")
            if p_value < 0.05:
                f.write("\nNemenyi Post-Hoc Test p-values:\n")
                f.write(str(nemenyi_result) + "\n")
            else:
                f.write("No significant differences found (Friedman test p-value >= 0.05).\n")
        logger.info(f"Saved statistical test results to {stats_file}")

        return test_results
    except Exception as e:
        logger.error(f"Error performing statistical tests: {e}")
        traceback.print_exc()
        return {'Friedman_p_value': None, 'Nemenyi_p_values': None}

# Evaluate model on test set
def validate_model(model, data_loader, criterion, device, class_names, variant, output_dir):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds, val_labels = [], []
    correctness = []  # Track per-sample correctness for statistical tests

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Validating MobileViT-{variant}"):
            valid_mask = labels != -1  # Filter out invalid samples
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
            # Record correctness (1 if correct, 0 if incorrect)
            correctness.extend((predicted == labels).cpu().numpy().astype(int))

    val_loss = val_loss / val_total if val_total > 0 else float('inf')
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='weighted', zero_division=0
    )

    # Compute confusion matrix
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
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_validation_{variant}.png'))
    plt.close()

    # Save confusion matrix as CSV
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = 'True Label'
    cm_df.columns.name = 'Predicted Label'
    cm_csv_path = os.path.join(summaries_dir, f'confusion_matrix_{variant}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to {cm_csv_path}")

    # Compile metrics
    metrics = {
        'model_variant': f"MobileViT-{variant}",
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    }

    print(f"Validation Results for {model_display_name}:")
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
    print(f"Confusion matrix saved to {os.path.join(plots_dir, f'confusion_matrix_validation_{variant}.png')}")

    return metrics, correctness

# Setup logging
def setup_logging(output_dir):
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'validation_log.txt')
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

def main_validation(args):
    """
    Main function to run validation on all three MobileViT models, save metrics to a CSV file,
    and perform statistical significance tests.

    Args:
        args: Command-line arguments with data_dir, batch_size, output_dir, etc.
    """
    set_seed(42)
    logger = setup_logging(args.output_dir)
    logger.info("Starting MobileViT models validation")

    # Define class names
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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms (same as in original code)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dir = os.path.join(args.data_dir, "val")
    try:
        validate_dataset(test_dir, class_names)
        test_dataset = SafeImageFolder(test_dir, transform=val_transform)
        logger.info(f"Class to index mapping: {test_dataset.class_to_idx}")
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        traceback.print_exc()
        return

    if test_dataset.invalid_files:
        logger.warning(f"{len(test_dataset.invalid_files)} invalid files in test dataset: {test_dataset.invalid_files[:5]}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define model configurations
    model_configs = [
        ('xx_small', 'apple/mobilevit-xx-small', 'outputs_mobilevit/models/mobilevit_xx_small_best.pth'),
        ('x_small', 'apple/mobilevit-x-small', 'outputs_mobilevit/models/mobilevit_x_small_best.pth'),
        ('small', 'apple/mobilevit-small', 'outputs_mobilevit/models/mobilevit_small_best.pth')
    ]

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    metrics_list = []
    model_correctness = {}
    model_names = []

    # Validate each model
    for variant, model_name, model_path in model_configs:
        logger.info(f"Validating MobileViT-{variant} ({model_name})")
        try:
            # Initialize model
            model = MobileViTForImageClassification.from_pretrained(
                model_name,
                num_labels=len(class_names),
                ignore_mismatched_sizes=True
            )
            # Load fine-tuned weights
            if not os.path.exists(model_path):
                logger.warning(f"Model weights not found at {model_path}, skipping")
                continue
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Verify state dict keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            if missing_keys or unexpected_keys:
                logger.warning(f"Missing keys in state dict for MobileViT-{variant}: {missing_keys}")
                logger.warning(f"Unexpected keys in state dict for MobileViT-{variant}: {unexpected_keys}")
            # Load state dict
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.error(f"Failed to load state dict for MobileViT-{variant}: {e}")
                continue
            model.to(device)

            # Run validation
            metrics, correctness = validate_model(model, test_loader, criterion, device, class_names, variant, output_dir)
            metrics_list.append(metrics)
            model_correctness[variant] = correctness
            model_names.append(variant)

        except Exception as e:
            logger.error(f"Error validating MobileViT-{variant}: {e}")
            traceback.print_exc()
            continue

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(summaries_dir, 'mobilevit_validation_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved validation metrics to {csv_path}")

    # Perform statistical significance tests
    if len(model_correctness) >= 2:  # Need at least 2 models for comparison
        stat_test_results = perform_statistical_tests(model_correctness, model_names, summaries_dir, logger)
        # Add Friedman p-value to metrics CSV
        metrics_df['Friedman_p_value'] = stat_test_results['Friedman_p_value']
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Updated metrics CSV with Friedman p-value at {csv_path}")
    else:
        logger.warning("Insufficient models validated for statistical tests (need at least 2).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all MobileViT models on test dataset")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for validation")
    parser.add_argument('--data_dir', type=str,
                        default="dataset/tomato_leaf_dataset",
                        help="Path to the dataset root directory")
    parser.add_argument('--output_dir', type=str,
                        default="outputs_mobilevit",
                        help="Directory to save validation outputs")
    args = parser.parse_args()
    main_validation(args)
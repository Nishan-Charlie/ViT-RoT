#!/usr/bin/env python3
import torch
from torchvision import datasets
from torchvision import transforms
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image
import os
import sys
import numpy as np
import pandas as pd
import argparse
import traceback
import logging
import random
import time

# Add project root to sys.path for custom module imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'CCT/src'))

try:
    from CCT.src.cct import cct_7_7x2_224, cct_14_7x2_224
except ImportError as e:
    raise ImportError(f"Failed to import src.cct: {e}")

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Image Dataset for a single image
class SingleImageDataset:
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __getitem__(self, index):
        try:
            image = Image.open(self.image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {self.image_path}: {e}")
            return torch.zeros(3, 224, 224) if self.transform is None else self.transform(Image.new('RGB', (224, 224)))

    def __len__(self):
        return 1

# Collect all images from the test dataset
def collect_all_images(test_dir, class_names):
    print("Collecting all images from test dataset...")
    valid_images = []
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} does not exist")
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for f in files:
            file_path = os.path.join(class_path, f)
            try:
                Image.open(file_path).verify()
                valid_images.append(file_path)
            except Exception as e:
                print(f"Error: Corrupted image {file_path}: {e}")
    if not valid_images:
        raise ValueError("No valid images found in test dataset")
    print(f"Collected {len(valid_images)} images")
    return valid_images

# Evaluate model on a single image and return inference time
def evaluate_model(model, image_path, transform, device, model_type, processor=None):
    dataset = SingleImageDataset(image_path, transform=transform)
    image = dataset[0].unsqueeze(0).to(device)  # Add batch dimension

    # Measure inference time
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        if model_type == "MobileViT":
            pil_image = transforms.ToPILImage()(image.squeeze(0).cpu())
            inputs = processor([pil_image], return_tensors="pt").to(device)
            _ = model(**inputs).logits
        else:
            _ = model(image)
    inference_time = time.time() - start_time
    return inference_time

# Setup logging
def setup_logging(output_dir):
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'test_log.txt')
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
    # Set random seed
    set_seed(42)

    output_dir = 'outputs_test'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    logger = setup_logging(output_dir)
    logger.info("Starting inference time evaluation for all images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 11

    data_dir = args.data_dir
    test_dir = os.path.join(data_dir, "val")

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

    # Collect all images
    try:
        image_paths = collect_all_images(test_dir, class_names)
    except Exception as e:
        logger.error(f"Failed to collect images: {e}")
        traceback.print_exc()
        return

    num_images = len(image_paths)
    logger.info(f"Total images to process: {num_images}")

    # Model configurations
    models = [
        {
            'type': 'CCT',
            'variants': [
                ('small', 'cct_7_7x2_224', 'outputs_cct/cct_7_7x2_224-20250425-123049/models/cct_7_7x2_224_best.pth'),
                ('base', 'cct_14_7x2_224', 'outputs_cct/cct_14_7x2_224-20250425-074018/models/cct_14_7x2_224_best.pth'),
            ]
        },
        {
            'type': 'EfficientViT',
            'variants': [
                ('b0', 'efficientvit_b0.r224_in1k', 'outputs_efficientvit/efficientvit_b0_best.pth'),
                ('b2', 'efficientvit_b2.r224_in1k', 'outputs_efficientvit/efficientvit_b2_best.pth'),
                ('m5', 'efficientvit_m5.r224_in1k', 'outputs_efficientvit/efficientvit_m5_best.pth'),
            ]
        },
        {
            'type': 'MobileViT',
            'variants': [
                ('xx_small', 'apple/mobilevit-xx-small', 'outputs_mobilevit/models/mobilevit_xx_small_best.pth'),
                ('x_small', 'apple/mobilevit-x-small', 'outputs_mobilevit/models/mobilevit_x_small_best.pth'),
                ('small', 'apple/mobilevit-small', 'outputs_mobilevit/models/mobilevit_small_best.pth')
            ]
        },
        {
            'type': 'Swin',
            'variants': [
                ('tiny', 'swin_tiny_patch4_window7_224', 'outputs_swin/swin_tiny_best.pth'),
                ('small', 'swin_small_patch4_window7_224', 'outputs_swin/swin_small_best.pth'),
                ('base', 'swin_base_patch4_window7_224', 'outputs_swin/swin_base_best.pth')
            ]
        },
        {
            'type': 'ViT',
            'variants': [
                ('tiny', 'vit_tiny_patch16_224', 'outputs_vit/vit_tiny_best.pth'),
                ('small', 'vit_small_patch16_224', 'outputs_vit/vit_small_best.pth'),
                ('base', 'vit_base_patch16_224', 'outputs_vit/vit_base_best.pth')
            ]
        },
        {
            'type': 'ConvNeXt',
            'variants': [
                ('tiny', 'convnext_tiny', 'outputs_convnext/convnext_tiny_best.pth'),
                ('small', 'convnext_small', 'outputs_convnext/convnext_small_best.pth'),
            ]
        }
    ]

    metrics_list = []

    for model_config in models:
        model_type = model_config['type']
        for variant, model_name, model_path in model_config['variants']:
            logger.info(f"Evaluating {model_type}-{variant} ({model_name})")

            try:
                # Initialize model and transform
                if model_type == "MobileViT":
                    processor = MobileViTImageProcessor.from_pretrained(model_name)
                    model = MobileViTForImageClassification.from_pretrained(
                        model_name,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif model_type == "CCT":
                    if model_name == 'cct_7_7x2_224':
                        model = cct_7_7x2_224(
                            pretrained=False,
                            progress=False,
                            img_size=224,
                            positional_embedding='learnable',
                            num_classes=num_classes
                        )
                    elif model_name == 'cct_14_7x2_224':
                        model = cct_14_7x2_224(
                            pretrained=False,
                            progress=False,
                            img_size=224,
                            positional_embedding='learnable',
                            num_classes=num_classes
                        )
                    else:
                        raise ValueError(f"Unsupported CCT model: {model_name}")
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    data_config = resolve_model_data_config(model)
                    transform = create_transform(**data_config, is_training=False)

                if not os.path.exists(model_path):
                    logger.warning(f"Model weights not found at {model_path}, skipping")
                    continue

                model.load_state_dict(torch.load(model_path))
                model.to(device)

                # Evaluate inference time for each image
                inference_times = []
                for image_path in image_paths:
                    try:
                        inference_time = evaluate_model(
                            model, image_path, transform, device, model_type,
                            processor=processor if model_type == "MobileViT" else None
                        )
                        inference_times.append(inference_time)
                    except Exception as e:
                        logger.warning(f"Error processing {image_path} for {model_type}-{variant}: {e}")
                        continue

                # Calculate total and mean inference time
                total_inference_time = sum(inference_times) if inference_times else 0.0
                mean_inference_time = np.mean(inference_times) if inference_times else 0.0
                metrics_list.append({
                    'model_variant': f"{model_type}-{variant}",
                    'total_inference_time': total_inference_time,
                    'mean_inference_time': mean_inference_time
                })
                logger.info(f"{model_type}-{variant}: Total Inference Time: {total_inference_time:.4f}s, "
                            f"Mean Inference Time: {mean_inference_time:.4f}s ({len(inference_times)} images processed)")

            except Exception as e:
                logger.error(f"Error evaluating {model_type}-{variant}: {e}")
                traceback.print_exc()
                continue

    # Save metrics to CSV
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(summaries_dir, 'inference_times_all_images.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved inference times to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure inference times of models on all images in test dataset")
    parser.add_argument('--data_dir', type=str,
                        default="dataset/tomato_leaf_dataset",
                        help="Path to the dataset root directory")

    args = parser.parse_args()
    main(args)
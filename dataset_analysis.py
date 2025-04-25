import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

# Generate 11 unique colors using tab20 colormap
colormap = cm.get_cmap('tab20', len(class_names))
colors = [mcolors.rgb2hex(colormap(i)) for i in range(len(class_names))]


def count_images(data_dir, class_names):
    """
    Count images per class in train and val splits.
    Returns a DataFrame with train and val counts.
    """
    counts = {'train': {}, 'val': {}}
    total_images = 0
    valid_extensions = ('.jpg', '.png', '.jpeg')

    for split in ['train', 'val']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Error: Directory {split_dir} does not exist")
            raise FileNotFoundError(f"Directory {split_dir} does not exist")

        for class_name in class_names:
            class_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_path} does not exist")
                counts[split][class_name] = 0
                continue

            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(class_path, f))
            ]
            count = len(image_files)
            counts[split][class_name] = count
            total_images += count
            print(f"{split}/{class_name}: {count} images found")

        if not counts[split]:
            print(f"Error: No classes found in {split_dir}")
            raise ValueError(f"No classes found in {split_dir}")

    if total_images == 0:
        print("Error: No valid images found in dataset")
        raise ValueError("No valid images found in dataset")

    df = pd.DataFrame({
        'Class': class_names,
        'Train': [counts['train'][cls] for cls in class_names],
        'Val': [counts['val'][cls] for cls in class_names]
    })
    for _, row in df.iterrows():
        total = row['Train'] + row['Val']
        print(f"Total {row['Class']}: {total} images (Train: {row['Train']}, Val: {row['Val']})")
    print(f"Grand Total: {total_images} images found")
    return df


def plot_donut_train(df, class_names, output_dir, colors):
    """
    Create and save a donut chart for train split.
    """
    plt.figure(figsize=(12, 10), dpi=400)

    counts = df['Train']
    valid_indices = [i for i, count in enumerate(counts) if count > 0]
    valid_classes = [class_names[i] for i in valid_indices]
    valid_counts = [counts[i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

    if not valid_counts:
        print("Error: No valid counts to plot for train")
        return

    wedges, texts = plt.pie(
        valid_counts,
        labels=[str(count) for count in valid_counts],
        autopct=None,
        startangle=90,
        textprops={'fontsize': 20},
        wedgeprops={'width': 0.4},
        colors=valid_colors
    )

    plt.title('Class Distribution: Train Split [Donut]', fontsize=16, pad=20)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'plots', 'class_distribution_donut_train.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    print(f"Saved train donut chart to {plot_path}")
    plt.close()


def plot_donut_val(df, class_names, output_dir, colors):
    """
    Create and save a donut chart for val split.
    """
    plt.figure(figsize=(12, 10), dpi=400)

    counts = df['Val']
    valid_indices = [i for i, count in enumerate(counts) if count > 0]
    valid_classes = [class_names[i] for i in valid_indices]
    valid_counts = [counts[i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

    if not valid_counts:
        print("Error: No valid counts to plot for val")
        return

    wedges, texts = plt.pie(
        valid_counts,
        labels=[str(count) for count in valid_counts],
        autopct=None,
        startangle=90,
        textprops={'fontsize': 20},
        wedgeprops={'width': 0.4},
        colors=valid_colors
    )

    plt.title('Class Distribution: Validation Split [Donut]', fontsize=16, pad=20)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'plots', 'class_distribution_donut_val.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    print(f"Saved val donut chart to {plot_path}")
    plt.close()


def main():
    data_dir = 'dataset/tomato_leaf_dataset'
    output_dir = 'outputs_dataset'

    try:
        df = count_images(data_dir, class_names)

        # Save dataset distribution to CSV
        summaries_dir = os.path.join(output_dir, 'summaries')
        os.makedirs(summaries_dir, exist_ok=True)
        csv_path = os.path.join(summaries_dir, 'dataset_distribution.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved dataset distribution to {csv_path}")

        plot_donut_train(df, class_names, output_dir, colors)
        plot_donut_val(df, class_names, output_dir, colors)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
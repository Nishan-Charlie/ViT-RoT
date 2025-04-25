import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def read_confusion_matrix_csv(file_path):
    """
    Read a confusion matrix from a CSV file.
    Returns a numpy array of the confusion matrix normalized to percentages.
    """
    df = pd.read_csv(file_path)
    # Ensure the True Label column is the index
    if 'True Label' in df.columns:
        df.set_index('True Label', inplace=True)
    # Verify that columns match class_names
    if not all(col in df.columns for col in class_names) or not all(row in df.index for row in class_names):
        raise ValueError(f"CSV file {file_path} does not match expected class names")
    # Convert to numpy array, ensuring order matches class_names
    cm = df.loc[class_names, class_names].to_numpy()
    # Normalize to percentages (sum to 100% per row)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm = (cm / row_sums) * 100
    return cm


def plot_confusion_matrices(cms, titles, class_names, output_dir, plot_name='separate_confusion_matrices.png'):
    """
    Plot three confusion matrices in a single row using seaborn heatmap with specified style.
    """
    fig, axes = plt.subplots(1, 3, figsize=(36, 10), dpi=300)
    axes = axes.flatten()

    for idx, (cm, title) in enumerate(zip(cms, titles)):
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 14},
            cbar=True,
            square=True,
            ax=axes[idx]
        )
        axes[idx].set_xlabel("Predicted Labels", fontsize=16)
        axes[idx].set_ylabel("True Labels", fontsize=16)
        axes[idx].set_title(title, fontsize=14, pad=20)
        axes[idx].tick_params(axis='x', rotation=45, labelright=True)
        axes[idx].tick_params(axis='y', rotation=0)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'plots', plot_name)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices plot to {plot_path}")
    plt.close()


def main():
    input_dir = 'outputs_test/summaries'  # Directory containing CSV files
    output_dir = 'outputs_dataset'  # Directory to save the plot
    csv_files = [
        os.path.join(input_dir, 'confusion_matrix_swin_small.csv'),
        os.path.join(input_dir, 'confusion_matrix_convnext_small.csv'),
        os.path.join(input_dir, 'confusion_matrix_efficientvit_b0.csv')
    ]
    titles = [
        "Swin Small",
        "Convnext Small",
        "EfficientVit B0"
    ]

    try:
        # Verify all CSV files exist
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file {csv_file} does not exist")

        # Read and normalize confusion matrices
        cms = [read_confusion_matrix_csv(csv_file) for csv_file in csv_files]

        # Plot and save the confusion matrices
        plot_confusion_matrices(cms, titles, class_names, output_dir)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
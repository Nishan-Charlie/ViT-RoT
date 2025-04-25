import matplotlib.pyplot as plt
import numpy as np
import os

def plot_model_parameter_sizes(include_optional_models=True):
    models = [
        {
            'type': 'MobileViT',
            'color': '#1f77b4',
            'variants': [
                ('xx_small', 1.29),
                ('x_small', 2.29),
                ('small', 5.58)
            ]
        },
        {
            'type': 'Swin',
            'color': '#ff7f0e',
            'variants': [
                ('tiny', 28.3),
                ('small', 49.6),
                ('base', 87.8)
            ]
        },
        {
            'type': 'ViT',
            'color': '#2ca02c',
            'variants': [
                ('tiny', 5.7),
                ('small', 21.7),
                ('base', 85.8)
            ]
        },
        {
            'type': 'ConvNeXt',
            'color': '#d62728',
            'variants': [
                ('tiny', 28.6),
                ('small', 50.2),
                ('base', 88.6)
            ]
        }
    ]

    if include_optional_models:
        models.extend([
            {
                'type': 'CCT',
                'color': '#9467bd',
                'variants': [
                    ('tiny', 3.7),
                    ('small', 6.7),
                    ('base', 13.4),
                ]
            },
            {
                'type': 'EfficientViT',
                'color': '#8c564b',
                'variants': [
                    ('b0', 3.4),
                    ('b2', 6.1),
                    ('m5', 8.5)
                ]
            },
            {
                'type': 'MaxViT',
                'color': '#e377c2',
                'variants': [
                    ('tiny', 31.0),
                    ('small', 70.0),
                    ('base', 120.0)
                ]
            },
            {
                'type': 'CSwin',
                'color': '#17becf',
                'variants': [
                    ('tiny', 23.0),
                    ('small', 35.0),
                    ('base', 78.0)
                ]
            }
        ])

    model_names = []
    param_sizes = []
    colors = []
    positions = []
    group_positions = []
    current_pos = 0
    bar_width = 0.15
    group_gap = 0.6

    xtick_labels = []

    for model in models:
        num_variants = len(model['variants'])
        start_pos = current_pos + bar_width * num_variants / 2.0 - bar_width / 2.0
        group_positions.append(start_pos)
        for variant, params in model['variants']:
            model_names.append(model['type'])
            xtick_labels.append(variant.replace('_', '-'))
            param_sizes.append(params)
            colors.append(model['color'])
            positions.append(current_pos)
            current_pos += bar_width
        current_pos += group_gap

    plt.figure(figsize=(14, 8))
    bars = plt.bar(positions, param_sizes, width=bar_width, color=colors, edgecolor='black')

    plt.title('Parameter Sizes of Transformer Models', fontsize=16, pad=20)
    # plt.xlabel('Variants', fontsize=14)
    plt.ylabel('Parameters (Millions)', fontsize=14)
    plt.xticks(positions, xtick_labels, rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add parameter values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f'{height:.2f}M', ha='center', va='bottom', fontsize=9)

    # Add models group names below the x-axis
    y_min, _ = plt.ylim()
    y_offset = y_min - 11  # Adjust if needed
    for group_pos, model in zip(group_positions, models):
        plt.text(group_pos, y_offset, model['type'],
                 ha='center', va='top', fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model['color'], edgecolor='black', label=model['type']) for model in models]
    plt.legend(handles=legend_elements, title='Model Types', fontsize=11, title_fontsize=13)

    plt.tight_layout()

    output_dir = 'outputs_plots'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'model_parameter_sizes_grouped.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    print("\nModel Parameter Sizes (in Millions):")
    for model in models:
        print(f"{model['type']}:")
        for variant, params in model['variants']:
            print(f"  {variant.replace('_', '-')}: {params:.2f}M")

    plt.close()

if __name__ == "__main__":
    plot_model_parameter_sizes(include_optional_models=True)
# parameter_histogram_analysis.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import phantom_creator


def bounded_output_layer(x):
    # Use tanh for different distribution shapes
    # T1: Sharp peak around 1.0 with long tail
    t1_base = torch.tanh(x[:, 0]) * 0.8 + 1.2  # Peak around 1.2
    t1 = torch.where(x[:, 0] > 0, t1_base + torch.relu(x[:, 0]) * 0.5, t1_base)

    # T2: Heavy left skew - use exponential-like
    t2 = 0.05 + 0.15 * torch.exp(-torch.relu(x[:, 1]))

    # PD: Bimodal - use mixture approach
    pd_mode1 = 0.75 + 0.05 * torch.sigmoid(x[:, 2] - 1)
    pd_mode2 = 0.85 + 0.05 * torch.sigmoid(x[:, 2] + 1)
    pd_weight = torch.sigmoid(x[:, 2])
    pd = pd_weight * pd_mode2 + (1 - pd_weight) * pd_mode1

    return torch.stack([t1, t2, pd], dim=1)


def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data using the bounded output layer."""
    # Create random input (simulating MLP output before bounded layer)
    torch.manual_seed(42)  # For reproducibility
    random_input = torch.randn(num_samples, 3)  # Normal distribution input

    # Apply bounded output layer
    synthetic_params = bounded_output_layer(random_input)

    return synthetic_params.numpy()


def create_histograms(phantom_path, output_path, num_bins=50):
    """Create histograms for T1, T2, and PD parameter maps, plus synthetic comparison."""

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load phantom (assuming standard size for now, adjust as needed)
    Nx, Ny = 36, 36  # Adjust these to match your phantom size
    phantom, _ = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=1)

    # Get parameter maps
    T1_map = phantom.T1.squeeze().cpu().numpy()
    T2_map = phantom.T2.squeeze().cpu().numpy()
    PD_map = phantom.PD.squeeze().cpu().numpy()

    # Create brain mask (non-zero pixels)
    brain_mask = (T1_map > 0) & (T2_map > 0) & (PD_map > 0)

    # Extract brain values only
    T1_brain = T1_map[brain_mask]
    T2_brain = T2_map[brain_mask]
    PD_brain = PD_map[brain_mask]

    print(f"Brain pixels: {np.sum(brain_mask)}")
    print(f"T1 range: {T1_brain.min():.3f} - {T1_brain.max():.3f}")
    print(f"T2 range: {T2_brain.min():.3f} - {T2_brain.max():.3f}")
    print(f"PD range: {PD_brain.min():.3f} - {PD_brain.max():.3f}")

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(len(T1_brain))  # Same number of samples as brain pixels
    T1_synthetic = synthetic_data[:, 0]
    T2_synthetic = synthetic_data[:, 1]
    PD_synthetic = synthetic_data[:, 2]

    print(f"\nSynthetic data ranges:")
    print(f"T1 synthetic range: {T1_synthetic.min():.3f} - {T1_synthetic.max():.3f}")
    print(f"T2 synthetic range: {T2_synthetic.min():.3f} - {T2_synthetic.max():.3f}")
    print(f"PD synthetic range: {PD_synthetic.min():.3f} - {PD_synthetic.max():.3f}")

    # Create comparison histograms (Real vs Synthetic)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    parameters = [
        (T1_brain, T1_synthetic, 'T1', 'T1 Values', 'red'),
        (T2_brain, T2_synthetic, 'T2', 'T2 Values', 'blue'),
        (PD_brain, PD_synthetic, 'PD', 'PD Values', 'green')
    ]

    for i, (real_values, synth_values, param_name, xlabel, color) in enumerate(parameters):
        # Real data histogram (top row)
        axes[0, i].hist(real_values, bins=num_bins, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_xlabel(xlabel)
        axes[0, i].set_ylabel('Number of Pixels')
        axes[0, i].set_title(f'{param_name} - Real Data')
        axes[0, i].grid(True, alpha=0.3)

        # Add statistics for real data
        real_mean, real_std = np.mean(real_values), np.std(real_values)
        axes[0, i].text(0.6, 0.8, f'μ={real_mean:.3f}\nσ={real_std:.3f}',
                        transform=axes[0, i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Synthetic data histogram (bottom row)
        axes[1, i].hist(synth_values, bins=num_bins, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_xlabel(xlabel)
        axes[1, i].set_ylabel('Number of Pixels')
        axes[1, i].set_title(f'{param_name} - Synthetic Data')
        axes[1, i].grid(True, alpha=0.3)

        # Add statistics for synthetic data
        synth_mean, synth_std = np.mean(synth_values), np.std(synth_values)
        axes[1, i].text(0.6, 0.8, f'μ={synth_mean:.3f}\nσ={synth_std:.3f}',
                        transform=axes[1, i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        print(f"\n{param_name} Comparison:")
        print(f"  Real    - Mean: {real_mean:.3f}, Std: {real_std:.3f}")
        print(f"  Synthetic - Mean: {synth_mean:.3f}, Std: {synth_std:.3f}")

    plt.suptitle('Real vs Synthetic Parameter Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / 'real_vs_synthetic_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create overlaid comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (real_values, synth_values, param_name, xlabel, color) in enumerate(parameters):
        axes[i].hist(real_values, bins=num_bins, alpha=0.6, color=color, label='Real Data', edgecolor='black')
        axes[i].hist(synth_values, bins=num_bins, alpha=0.6, color='gray', label='Synthetic', edgecolor='black')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Number of Pixels')
        axes[i].set_title(f'{param_name} - Real vs Synthetic')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.suptitle('Overlaid Comparison: Real vs Synthetic Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / 'overlaid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create original visualizations as well
    create_original_histograms(T1_brain, T2_brain, PD_brain, output_path, num_bins)

    # Save comparison statistics
    with open(output_path / 'real_vs_synthetic_stats.txt', 'w') as f:
        f.write("REAL vs SYNTHETIC DATA COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of samples: {len(T1_brain)}\n\n")

        for real_values, synth_values, param_name, _, _ in parameters:
            f.write(f"{param_name} Statistics:\n")
            f.write(f"  REAL DATA:\n")
            f.write(f"    Mean: {np.mean(real_values):.4f}\n")
            f.write(f"    Std:  {np.std(real_values):.4f}\n")
            f.write(f"    Min:  {np.min(real_values):.4f}\n")
            f.write(f"    Max:  {np.max(real_values):.4f}\n")
            f.write(f"  SYNTHETIC DATA:\n")
            f.write(f"    Mean: {np.mean(synth_values):.4f}\n")
            f.write(f"    Std:  {np.std(synth_values):.4f}\n")
            f.write(f"    Min:  {np.min(synth_values):.4f}\n")
            f.write(f"    Max:  {np.max(synth_values):.4f}\n")
            f.write(f"  DIFFERENCE:\n")
            f.write(f"    Mean diff: {abs(np.mean(real_values) - np.mean(synth_values)):.4f}\n")
            f.write(f"    Std diff:  {abs(np.std(real_values) - np.std(synth_values)):.4f}\n\n")

    print(f"\nHistograms and comparison saved to: {output_path}")
    print("Files created:")
    print("  - real_vs_synthetic_histograms.png")
    print("  - overlaid_comparison.png")
    print("  - real_vs_synthetic_stats.txt")
    print("  - (plus original histogram files)")


def create_original_histograms(T1_brain, T2_brain, PD_brain, output_path, num_bins):
    """Create the original individual histograms."""
    # Create individual histograms
    parameters = [
        (T1_brain, 'T1', 'T1 Values', 'red'),
        (T2_brain, 'T2', 'T2 Values', 'blue'),
        (PD_brain, 'PD', 'PD Values', 'green')
    ]

    for values, param_name, xlabel, color in parameters:
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(values, bins=num_bins, alpha=0.7, color=color, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel('Number of Pixels')
        plt.title(f'{param_name} Histogram (Brain Pixels Only)')
        plt.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.text(0.7, 0.8, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nPixels: {len(values)}',
                 transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path / f'{param_name.lower()}_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Create combined histogram
    plt.figure(figsize=(15, 5))

    for i, (values, param_name, xlabel, color) in enumerate(parameters, 1):
        plt.subplot(1, 3, i)
        plt.hist(values, bins=num_bins, alpha=0.7, color=color, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel('Number of Pixels')
        plt.title(f'{param_name} Histogram')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.text(0.6, 0.8, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                 transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.suptitle('Parameter Maps Histograms (Brain Pixels Only)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / 'combined_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()


# ===== EXECUTION =====
if __name__ == "__main__":
    phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
    output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_72\parameter_histograms"

    create_histograms(
        phantom_path=phantom_path,
        output_path=output_path,
        num_bins=50  # Number of histogram bins
    )

    print("✅ Parameter histogram analysis complete!")
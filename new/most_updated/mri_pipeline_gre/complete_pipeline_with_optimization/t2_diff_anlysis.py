# timestep_correlation_analysis.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import pypulseq as pp
import phantom_creator
from simulate_and_process import simulate_and_process_mri


def create_perturbations(param_map, num_perturbations):
    """Create parameter perturbations."""
    factors = np.linspace(-1.0, 1.0, num_perturbations)
    np.random.shuffle(factors)
    perturbations = []

    for i, factor in enumerate(factors):
        perturbed = param_map * (1 + factor)
        perturbed = torch.clamp(perturbed, min=0.001)
        perturbations.append({'map': perturbed, 'factor': factor})

    return perturbations


def calc_l2_diff(original, perturbed, mask=None):
    """Calculate normalized L2 difference."""
    if mask is not None:
        orig, pert = original[mask], perturbed[mask]
    else:
        orig, pert = original.flatten(), perturbed.flatten()

    diff_norm = torch.norm(orig - pert, p=2)
    orig_norm = torch.norm(orig, p=2)
    return (diff_norm / orig_norm).item() if orig_norm > 1e-10 else 0.0


def analyze_parameter(param_name, param_gt, T1_gt, T2_gt, PD_gt, orig_images, seq_path, phantom_path,
                      coil_maps, grappa_weights, num_perturbations, output_path):
    """Analyze one parameter and create timestep plots."""
    print(f"\n=== {param_name} ANALYSIS ===")

    mask = param_gt > 0
    perturbations = create_perturbations(param_gt, num_perturbations)
    num_timesteps = orig_images.shape[0]

    # Collect data
    param_diffs, timestep_diffs = [], {t: [] for t in range(num_timesteps)}

    for i, pert in enumerate(perturbations):
        print(f"Processing {i + 1}/{num_perturbations} (factor={pert['factor']:.3f})")

        # Create perturbed phantom
        maps = {'T1': T1_gt, 'T2': T2_gt, 'PD': PD_gt}
        maps[param_name] = pert['map']

        pert_obj = phantom_creator.create_phantom_with_custom_parameters(
            T1_map=maps['T1'], T2_map=maps['T2'], PD_map=maps['PD'],
            Nread=param_gt.shape[0], Nphase=param_gt.shape[1],
            phantom_path=phantom_path, coil_maps=coil_maps).build()

        try:
            _, pert_images, _ = simulate_and_process_mri(pert_obj, seq_path, 34, grappa_weights_torch=grappa_weights)
        except:
            continue

        # Calculate differences
        param_diff = calc_l2_diff(param_gt, pert['map'], mask)
        param_diffs.append(param_diff)

        for t in range(num_timesteps):
            img_diff = calc_l2_diff(orig_images[t], pert_images[t], mask)
            timestep_diffs[t].append(img_diff)

    # Create plots and find high correlations
    plots_path = Path(output_path) / f'{param_name.lower()}_timestep_plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    high_corr_timesteps = []

    for t in range(num_timesteps):
        if len(timestep_diffs[t]) < 2:
            continue

        # Calculate correlation
        param_series = pd.Series(param_diffs[:len(timestep_diffs[t])])
        img_series = pd.Series(timestep_diffs[t])
        corr = param_series.corr(img_series)

        if abs(corr) > 0.9:
            high_corr_timesteps.append((t, corr))

        # Create plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(param_series, img_series, c=param_series.index, cmap='viridis', alpha=0.7)
        plt.xlabel(f'{param_name} Normalized L2 Difference')
        plt.ylabel('Image Normalized L2 Difference')
        plt.title(f'{param_name} vs Image Sensitivity - Timestep {t}\nCorrelation: {corr:.4f}')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Perturbation Index')
        plt.tight_layout()
        plt.savefig(plots_path / f'timestep_{t:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"High correlation timesteps (|corr| > 0.9): {high_corr_timesteps}")
    return high_corr_timesteps


def run_analysis(seq_path, phantom_path, output_path, parameters_to_analyze=['T2'], num_perturbations=50):
    """Main analysis function."""

    # Setup
    seq = pp.Sequence();
    seq.read(seq_path)
    Nx, Ny = int(seq.get_definition('Nx')), int(seq.get_definition('Ny'))
    phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=34)
    coil_maps = coil_maps.to("cuda")

    # Ground truth maps
    T1_gt = phantom.T1.squeeze().to("cuda")
    T2_gt = phantom.T2.squeeze().to("cuda")
    PD_gt = phantom.PD.squeeze().to("cuda")

    # Generate original images
    print("Generating reference images...")
    _, orig_images, grappa_weights = simulate_and_process_mri(phantom.build(), seq_path, 34)
    grappa_weights = grappa_weights.detach()

    # Analyze each parameter
    all_results = {}
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for param_name in parameters_to_analyze:
        param_gt = {'T1': T1_gt, 'T2': T2_gt, 'PD': PD_gt}[param_name]
        high_corr = analyze_parameter(param_name, param_gt, T1_gt, T2_gt, PD_gt, orig_images,
                                      seq_path, phantom_path, coil_maps, grappa_weights,
                                      num_perturbations, output_path)
        all_results[param_name] = high_corr

    # Write results to file
    results_file = output_path / 'high_correlation_timesteps.txt'
    with open(results_file, 'w') as f:
        f.write("HIGH CORRELATION TIMESTEPS (|correlation| > 0.9)\n")
        f.write("=" * 50 + "\n\n")

        for param, timesteps in all_results.items():
            f.write(f"{param}:\n")
            if timesteps:
                for timestep, corr in timesteps:
                    f.write(f"  Timestep {timestep:3d}: correlation = {corr:.4f}\n")
            else:
                f.write("  No high correlation timesteps found\n")
            f.write("\n")

    # Find timesteps with high correlation for ALL parameters
    if len(parameters_to_analyze) >= 2:
        # Get timestep numbers only (ignore correlation values)
        param_timesteps = {}
        for param, timesteps in all_results.items():
            param_timesteps[param] = set(t for t, corr in timesteps)

        # Find intersection of all parameter timesteps
        if len(param_timesteps) > 0:
            common_timesteps = set.intersection(*param_timesteps.values()) if param_timesteps else set()
        else:
            common_timesteps = set()

        # Write common timesteps to file
        with open(results_file, 'a') as f:
            f.write("=" * 50 + "\n")
            f.write("TIMESTEPS WITH HIGH CORRELATION FOR ALL PARAMETERS:\n")
            f.write("=" * 50 + "\n")
            if common_timesteps:
                f.write(f"Timesteps: {sorted(common_timesteps)}\n")
                f.write(f"Count: {len(common_timesteps)} timesteps\n\n")

                # Show details for each common timestep
                for t in sorted(common_timesteps):
                    f.write(f"Timestep {t}:\n")
                    for param, timesteps in all_results.items():
                        corr = next((corr for ts, corr in timesteps if ts == t), None)
                        if corr is not None:
                            f.write(f"  {param}: {corr:.4f}\n")
                    f.write("\n")
            else:
                f.write("No timesteps have high correlation for all parameters\n")

    # Final summary
    print(f"\n🎯 SUMMARY - HIGH CORRELATION TIMESTEPS (|corr| > 0.9):")
    for param, timesteps in all_results.items():
        print(f"{param}: {[t for t, corr in timesteps]}")

    if len(parameters_to_analyze) >= 2:
        print(f"\n🎯 TIMESTEPS WITH HIGH CORRELATION FOR ALL PARAMETERS:")
        if common_timesteps:
            print(f"Timesteps: {sorted(common_timesteps)}")
            print(f"Count: {len(common_timesteps)} timesteps")
        else:
            print("No timesteps have high correlation for all parameters")

    print(f"\nResults saved to: {results_file}")


# ===== EXECUTION =====
if __name__ == "__main__":
    seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_72\epi_gre_mrf_epi.seq"
    phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
    output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_72\timestep_analysis"

    run_analysis(
        seq_path=seq_path,
        phantom_path=phantom_path,
        output_path=output_path,
        parameters_to_analyze=['T1', 'T2', 'PD'],
        num_perturbations=200
    )
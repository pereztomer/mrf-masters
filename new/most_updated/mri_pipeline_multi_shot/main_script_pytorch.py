from data_loader_pytorch import load_data_torch
from epi_pipeline_torch import run_epi_pipeline_torch
import os
import torch

# GPU Configuration - First small change
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


base_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\6.6.25\2025-06-05_epi_multishot_ref_Nx192_Ny192_R3_part_fourier_reps1"
raw_data_path = os.path.join(base_path, "data.mat")
output_dir = os.path.join(base_path, "plots")
seq_file_path = os.path.join(base_path, "2025-06-05_epi_multishot_ref_Nx192_Ny192_R3_part_fourier_reps1.seq")
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

use_mr0_simulator = True  # Set to True to use MR0 simulator
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
use_phase_correction = True
use_grappa = True
show_plots = True
# MR0 coil configuration
num_coils = 34  # Number of coils
reference_signal_per_shot, raw_data_torch, seq = load_data_torch(
    raw_data_path,
    use_mr0=use_mr0_simulator,
    seq_file_path=seq_file_path,
    phantom_path=phantom_path,
    num_coils=num_coils,
    device=device
)

k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# Run EPI reconstruction pipeline
sos_image, data_xy, measured_traj_delay = run_epi_pipeline_torch(
    reference_signal_per_shot=reference_signal_per_shot,
    rawdata=raw_data_torch,
    device=device,
    seq=seq,
    use_phase_correction=use_phase_correction,
    use_grappa=use_grappa,
    show_plots=show_plots,
    output_dir=output_dir)

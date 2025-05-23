from data_loader import load_data
from epi_pipeline import run_epi_pipeline
import os

# Configuration
raw_data_path = r"C:\Users\perez\Desktop\test\epi\epi_data3.mat"
output_dir = r"C:\Users\perez\Desktop\test\epi\plots"  # Directory to save plots

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

use_mr0_simulator = True  # Set to True to use MR0 simulator
# seq_file = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3.seq';

# seq_file = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\sequences\2025-05-20_epi_Nx_192_Ny_192_part_fourier_factor_1_R3_repetetions_1.seq"

seq_file_path = r"/new/most_updated\sequences\2025-05-23_epi_Nx128_Ny128_part_fourier_repetitions_1.seq"
phantom_path = r"/new/most_updated\numerical_brain_cropped.mat"
use_phase_correction = True

# MR0 coil configuration
use_coil_maps = True  # Whether to add coil sensitivity maps
num_coils = 34  # Number of coils

# Load data (agnostic to source)
raw_data, seq = load_data(
    raw_data_path,
    use_mr0=use_mr0_simulator,
    seq_file_path=seq_file_path,
    phantom_path=phantom_path,
    use_coil_maps=use_coil_maps,
    num_coils=num_coils
)

# k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=traj_recon_delay)
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# Run EPI reconstruction pipeline
sos_image, data_xy, measured_traj_delay = run_epi_pipeline(
    rawdata=raw_data,
    seq=seq,
    use_phase_correction=use_phase_correction,
    show_plots=True,
    output_dir=output_dir
)

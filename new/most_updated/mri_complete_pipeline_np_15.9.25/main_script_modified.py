from data_loader import load_data
from epi_pipeline import run_epi_pipeline
import os

# Configuration
# base_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\25_5_25\2025-05-25_epi_Nx128_Ny128_repetitions_1"
# raw_data_path = os.path.join(base_path, "data.mat")
# output_dir = os.path.join(base_path, "plots")
# seq_file_path = os.path.join(base_path, "2025-05-25_epi_Nx128_Ny128_repetitions_1.seq")
# raw_data_path = ""
# output_dir = "temp2"
# seq_file_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\sequence_writing_code_es_epi\6.5.25_epi_time_series_with_inversion_spoiler_gradient_half_fourier.seq"


# base_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\6.6.25\2025-06-04_epi_Nx192_Ny192_R3_part_fourier_repetitions_1"
# raw_data_path = os.path.join(base_path, "data.mat")
# output_dir = os.path.join(base_path, "plots")
# seq_file_path = os.path.join(base_path, "2025-06-04_epi_Nx192_Ny192_R3_part_fourier_repetitions_1.seq")


base_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\26.6.25\epi_gre"
raw_data_path = os.path.join(base_path, "data.mat")
output_dir = os.path.join(base_path, "plots")
seq_file_path = os.path.join(base_path, "epi_gre_192.seq")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

use_mr0_simulator = False  # Set to True to use MR0 simulator
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
use_phase_correction = True

# MR0 coil configuration
num_coils = 34  # Number of coils

# Load data (agnostic to source)
raw_data, seq = load_data(
    raw_data_path,
    use_mr0=use_mr0_simulator,
    seq_file_path=seq_file_path,
    phantom_path=phantom_path,
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

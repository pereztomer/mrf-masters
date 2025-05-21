from data_loader import load_data
from epi_pipeline import run_epi_pipeline

# Configuration
raw_data_path = r"C:\Users\perez\Desktop\test\epi\epi_data2.mat"

use_mr0_simulator = True  # Set to True to use MR0 simulator
# seq_file = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3.seq';

# seq_file = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\progress\sequences\2025-05-20_epi_Nx_192_Ny_192_part_fourier_factor_1_R3_repetetions_1.seq"
seq_file = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\progress\sequences\2025-05-20_epi_Nx_192_Ny_192_part_fourier_factor_1_repetetions_1.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\progress\numerical_brain_cropped.mat"
use_phase_correction = True

# MR0 coil configuration
use_coil_maps = True  # Whether to add coil sensitivity maps
num_coils = 34         # Number of coils

# Load data (agnostic to source)
rawdata, ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing = load_data(
    raw_data_path,
    use_mr0=use_mr0_simulator,
    seq_file=seq_file,
    phantom_path=phantom_path,
    use_coil_maps=use_coil_maps,
    num_coils=num_coils
)

# Run EPI reconstruction pipeline
sos_image, data_xy, measured_traj_delay = run_epi_pipeline(
    rawdata,
    ktraj_adc,
    t_adc,
    use_phase_correction=use_phase_correction,
    show_plots=True
)

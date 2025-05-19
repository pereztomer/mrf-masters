from data_loader import load_data
from epi_pipeline import run_epi_pipeline

# Configuration
raw_data_path = r"C:\Users\perez\Desktop\test\epi\epi_data.mat"
use_mr0_simulator = False  # Set to True to use MR0 simulator
phantom_path = None  # Path to phantom if using MR0
use_phase_correction = True

# Load data (agnostic to source)
rawdata, ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing = load_data(
    raw_data_path,
    use_mr0=use_mr0_simulator,
    phantom_path=phantom_path
)

# Run EPI reconstruction pipeline
sos_image, data_xy, measured_traj_delay = run_epi_pipeline(
    rawdata,
    ktraj_adc,
    t_adc,
    use_phase_correction=use_phase_correction,
    show_plots=True
)

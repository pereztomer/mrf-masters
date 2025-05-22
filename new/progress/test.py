import pypulseq as pp
import numpy as np

# Define file paths
seq_file_path = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3.seq'
save_path = r'C:\Users\perez\Desktop\test\epi\epi_data2.mat'
data_file_path = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\meas_MID00028_FID17313_4_5_25_epi_time_series_with_inversion_spoiler_gradient_v3.dat'

# Read the sequence file
seq = pp.Sequence()
seq.read(seq_file_path, detect_rf_use=True)

# Set trajectory reconstruction delay
traj_recon_delay = 9.26044439e-07

# Calculate k-space trajectory - this is the equivalent function
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=traj_recon_delay)

# k_traj_adc : numpy.array
#             K-space trajectory sampled at `t_adc` timepoints.
#         k_traj : numpy.array
#             K-space trajectory of the entire pulse sequence.
#         t_excitation : List[float]
#             Excitation timepoints.
#         t_refocusing : List[float]
#             Refocusing timepoints.
#         t_adc : numpy.array
#             Sampling timepoints.
print("here")


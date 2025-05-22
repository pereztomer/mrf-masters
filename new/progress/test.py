import pypulseq as pp
# import numpy as np
#
# # Define file paths
# seq_file_path = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3.seq'
# save_path = r'C:\Users\perez\Desktop\test\epi\epi_data2.mat'
# data_file_path = r'C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\4.5.25_epi_time_series_with_inversion_spoiler_gradient_v3\meas_MID00028_FID17313_4_5_25_epi_time_series_with_inversion_spoiler_gradient_v3.dat'

# # Read the sequence file
# seq = pp.Sequence()
# seq.read(seq_file_path, detect_rf_use=True)
#
# # Set trajectory reconstruction delay
# traj_recon_delay = 9.26044439e-07
#
# # Calculate k-space trajectory - this is the equivalent function
# k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=traj_recon_delay)

import pypulseq as pp

seq_filename = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\progress\sequences\2025-05-22_epi_Nx192_Ny192_R3_part_fourier_repetitions_1.seq"



# Load the sequence
seq = pp.Sequence()
seq.read(seq_filename)

# Read individual definitions
fov = seq.get_definition('FOV')
name = seq.get_definition('Name')
resolution = seq.get_definition('Resolution')
matrix_size = seq.get_definition('MatrixSize [Nx, Ny, slice_number]')
time_steps = seq.get_definition('TimeSteps')
freq_encoding_steps = seq.get_definition('FrequencyEncodingSteps')
acceleration_factor = seq.get_definition('AccelerationFactor')
partial_fourier_factor = seq.get_definition('PartialFourierFactor')
echo_time = seq.get_definition('EchoTime')
repetition_time = seq.get_definition('RepetitionTime')
flip_angles = seq.get_definition('FlipAngles')
ny_sampled = seq.get_definition('NySampled')
ny_pre = seq.get_definition('NyPre')
ny_post = seq.get_definition('NyPost')

# Print all the definitions
print("Sequence Definitions:")
print(f"FOV: {fov}")
print(f"Name: {name}")
print(f"Resolution: {resolution}")
print(f"Matrix Size: {matrix_size}")
print(f"Time Steps: {time_steps}")
print(f"Frequency Encoding Steps: {freq_encoding_steps}")
print(f"Acceleration Factor: {acceleration_factor}")
print(f"Partial Fourier Factor: {partial_fourier_factor}")
print(f"Echo Time: {echo_time}")
print(f"Repetition Time: {repetition_time}")
print(f"Flip Angles: {flip_angles}")


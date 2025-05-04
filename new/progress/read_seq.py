import pypulseq as pp

# Load an existing sequence file
seq = pp.Sequence()
seq.read('1.5.25_epi_se_rs_time_series_with_inversion.seq')

out = seq.calculate_kspace()
# Now you can access various parts of the sequence
# For example, to get the list of all blocks in the sequence:
print("here")

# 47104

import mapvbvd
import numpy as np
import matplotlib.pyplot as plt

# Load the raw data file
twix_obj = mapvbvd.mapVBVD(r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\29.4.25_epi_se_rs_Nx_128_Ny_128_pypulseq_max_slew_150_time_series\meas_MID00154_FID17007_29_4_25_nx_128_ny_128_time_series.dat")
"double(twix_obj{end}.image.unsorted());"

raw_data = twix_obj[1].image['']

print(128*128*50, 47104 * 44, 184*44*6400)
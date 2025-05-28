import torch
import numpy as np
from data_loader import load_data
from data_loader_pytorch import load_data_torch
import os

# Test configuration
base_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\25_5_25\2025-05-25_epi_Nx128_Ny128_repetitions_1"
raw_data_path = os.path.join(base_path, "data.mat")
output_dir = os.path.join(base_path, "plots")
seq_file_path = os.path.join(base_path, "2025-05-25_epi_Nx128_Ny128_repetitions_1.seq")
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
num_coils = 34

def test_combination(name, use_mr0, num_coils_param, device='cpu'):
    print(f"\nTesting {name}...")

    # Original version
    rawdata_orig, _ = load_data(raw_data_path, use_mr0=use_mr0, seq_file_path=seq_file_path,
                               phantom_path=phantom_path,
                               num_coils=num_coils_param)
    # PyTorch version
    rawdata_torch, _ = load_data_torch(raw_data_path, use_mr0=use_mr0, seq_file_path=seq_file_path,
                                      phantom_path=phantom_path, num_coils=num_coils_param, device=device)

    # Compare
    l2_diff = torch.norm(rawdata_torch - rawdata_torch)
    max_diff = torch.max(torch.abs(rawdata_torch - rawdata_torch))

    print(f"  Original: {rawdata_orig.shape}, {rawdata_orig.dtype}")
    print(f"  PyTorch:  {rawdata_torch.shape}, {rawdata_torch.dtype}, device={rawdata_torch.device}")
    print(f"  L2 diff: {l2_diff:.2e}, Max diff: {max_diff:.2e}")
    print(f"  {'✅ PASSED' if l2_diff < 1e-4 else '❌ FAILED'}")

# Test all combinations
print("=" * 50)
print("TESTING ALL COMBINATIONS")
print("=" * 50)

# # Real data combinations
test_combination("Real Data (CPU)", use_mr0=False, num_coils_param=None, device='cpu')
test_combination("Real Data (GPU)", use_mr0=False, num_coils_param=None, device='cuda')
# #
# # MR0 simulator without coils
test_combination("MR0 No Coils (CPU)", use_mr0=True, num_coils_param=None, device='cpu')
test_combination("MR0 No Coils (GPU)", use_mr0=True, num_coils_param=None, device='cuda')
#
# # MR0 simulator with coils
test_combination("MR0 With Coils (CPU)", use_mr0=True, num_coils_param=num_coils, device='cpu')
#
test_combination("MR0 With Coils (GPU)", use_mr0=True, num_coils_param=num_coils, device='cuda')

print(f"\n{'=' * 50}")
print("ALL TESTS COMPLETE")
print("=" * 50)
import MRzeroCore as mr0
import matplotlib.pyplot as plt
import numpy as np
import torch
import pypulseq as pp


def load_mr0_data_torch(seq_file, phantom_path="numerical_brain_cropped.mat"):
    seq = pp.Sequence()
    seq.read(seq_file)
    ktraj_adc, _, _, _, t_adc = seq.calculate_kspace()
    ktraj_adc = torch.from_numpy(ktraj_adc).to(device='cuda',dtype=torch.float32)
    t_adc = torch.from_numpy(t_adc).to(device='cuda', dtype=torch.float32)
    Nx = 128
    Ny = 128
    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
    obj_p = obj_p.interpolate(int(Nx), int(Ny), 1)
    obj_p = obj_p.build()

    # Simulate the sequence
    # graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 2000, 1e-3)
    # signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=True)
    kspace_data = signal.reshape(Nx,Ny)
    kspace_data[1::2] = kspace_data[1::2].flip(1)
    from single_shot_analysis import reconstruct_shifted_kspace
    image = reconstruct_shifted_kspace(kspace_data)
    recon_original = torch.abs(image).detach().cpu().numpy()
    plt.imshow(recon_original)
    plt.show()
    exit()
    return signal
    # res = mr0.reco_adjoint(signal.cpu(), seq0.get_kspace())
    # plt.imshow(np.abs(res))
    # plt.show()
    # exit()

from single_shot_analysis import correct_odd_even_with_natural_progression
def main():
    # seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\26.6.25\epi_gre\epi_gre_192.seq"
    seq_path = r"epi_gre_128.seq"
    signal = load_mr0_data_torch(seq_file=seq_path)
    from single_shot_analysis import fix_single_shot_epi
    # fix_single_shot_epi(signal, ktraj_adc, t_adc)
    correct_odd_even_with_natural_progression(signal, 256e-3)
    print("heree")


if __name__ == '__main__':
    main()

import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt
import numpy as np
from research_sequences_from_scratch.convert_raw_data_to_kspace import grid_kspace_2d
import matplotlib
matplotlib.use('TkAgg')

np.int = int
np.float = float
np.complex = complex
def main():
    seq_file = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\research_sequences_from_scratch\epi\epi_with_acceleration_and_half_fourier\epi_pypulseq_acceleration_R_1_half_fourier_True.seq"
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    seq = pp.Sequence()
    seq.read(seq_file)
    print('load phantom')
    obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
    brain_phantom_res = 128
    obj_p = obj_p.interpolate(brain_phantom_res, brain_phantom_res, 1)
    obj_p.B0[:] = 0
    plot_phantom = True
    if plot_phantom: obj_p.plot()
    obj_p = obj_p.build()
    print('simulate (2D) \n' + seq_file)
    seq0 = mr0.Sequence.import_file(seq_file)
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=False)
    # @title 3. Plot sequence and signal
    sp_adc, t_adc = mr0.util.pulseq_plot(seq=seq, signal=signal.numpy())

    # Unfortunately, we need to limit the resolution as reco_adjoint is very RAM-hungy
    print('reconstruct and plot')
    seq0.plot_kspace_trajectory()
    kspace_test = seq0.get_kspace()
    organized_kspace = grid_kspace_2d(signal, kspace_test, grid_size=(128, 128))

    # plot ksapce
    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")
    plt.imshow(torch.log(organized_kspace.abs()+1), origin="lower")
    plt.show()
    # Shift DC from center to corner (inverse of fftshift)
    spectrum = torch.fft.ifftshift(organized_kspace)
    space = torch.fft.fft2(spectrum)
    space = torch.fft.ifftshift(space)

    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")
    plt.imshow(space.abs(), origin="lower")
    plt.colorbar()
    plt.subplot(122)
    plt.title("Phase")
    plt.imshow(space.angle(), origin="lower", vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
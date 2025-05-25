import MRzeroCore as mr0
import pypulseq as pp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import scipy.io as sio


def main():

    seq_file = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\sequence_writing_code\sequences\2025-05-25_epi_Nx192_Ny192_R3_part_fourier_repetitions_50_multi_shot_for_calibration.seq"

    # # for plotting graph of the signal evolotion
    # seq = pp.Sequence()
    # seq.read(seq_file)
    # signal = mr0.util.simulate_2d(seq)
    # seq.plot(plot_now=False)
    # mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
    # plt.show()
    # exit()
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat("numerical_brain_cropped.mat")
    obj_p = obj_p.build()
    # Simulate the sequence
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    exit()
    reco = mr0.reco_adjoint(signal, seq0.get_kspace(), resolution=(128, 128, 1), FOV=(0.22, 0.22, 1))
    plt.figure()
    plt.subplot(131)
    plt.title("Magnitude")
    plt.imshow(reco[:, :, 0].T.abs(), origin="lower")
    plt.colorbar()
    plt.subplot(132)
    plt.title("Phase")
    plt.imshow(reco[:, :, 0].T.angle(), origin="lower", vmin=-np.pi, vmax=np.pi)
    plt.colorbar()

    # Compute and display k-space from reconstructed image
    reco_img = reco[:, :, 0].T
    kspace_from_reco = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(reco_img)))
    plt.subplot(133)
    plt.title("K-space from reconstructed image")
    plt.imshow(np.abs(np.log(kspace_from_reco) + 1), cmap='gray')
    plt.colorbar()
    plt.show()


    # kspace_matrix, kx_grid, ky_grid = epi_resample_to_grid(signal, seq0.get_kspace(), Nx=128, Ny=128, method='cubic')
    # plt.imshow(np.abs(np.log(kspace_matrix)+1), cmap='gray')
    # plt.show()
    #
    # # Convert kspace_matrix to image domain and display
    # img_from_kspace = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace_matrix)))
    # plt.figure()
    # plt.title("Image from custom gridded k-space")
    # plt.imshow(np.abs(img_from_kspace), cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # # kspace = signal.reshape(184, 72)
    # # plt.imshow(torch.log(kspace.abs()+1))
    # # plt.show()

if __name__ == '__main__':
    main()

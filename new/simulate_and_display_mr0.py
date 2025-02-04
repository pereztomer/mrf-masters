import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt
import numpy as np
from convert_raw_data_to_kspace import grid_kspace_2d
import matplotlib

matplotlib.use('TkAgg')
from partial_fourier_recon import pocs_pf
from generate_coil_sensitivity_maps import generate_coil_maps, visualize_maps

np.int = int
np.float = float
np.complex = complex


def image_to_kspace(image):
    # Center FFT
    centered_image = torch.fft.ifftshift(image)
    # Compute 2D FFT
    kspace = torch.fft.fft2(centered_image)
    # Shift zero frequency components
    kspace_shifted = torch.fft.fftshift(kspace)
    return kspace_shifted


def kspace_to_image(kspace):
    """
    Convert k-space data to image domain using inverse 2D FFT
    """
    centered_kspace = torch.fft.ifftshift(kspace)
    image = torch.fft.ifft2(centered_kspace)
    image_shifted = torch.fft.fftshift(image)

    return image_shifted

def main():
    use_pocs = False
    seq_file = "epi_pypulseq_Nx_128_Ny_128_acceleration_R_4_half_fourier_True.seq"
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    seq = pp.Sequence()
    seq.read(seq_file)
    print('load phantom')
    obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
    brain_phantom_res = 128
    obj_p = obj_p.interpolate(brain_phantom_res, brain_phantom_res, 1)
    obj_p.B0[:] = 0

    coil_sensitivity_maps = generate_coil_maps((brain_phantom_res, brain_phantom_res), 8)
    # visualize_maps(coil_sensitivity_maps)
    # # if we want to insert the sensitivity maps into the object
    # coil_sensitivity_maps = torch.Tensor(np.expand_dims(coil_sensitivity_maps, axis=-1))
    # obj_p.coil_sens = coil_sensitivity_maps

    plot_phantom = True
    if plot_phantom: obj_p.plot()
    obj_p = obj_p.build()
    print('simulate (2D) \n' + seq_file)
    seq0 = mr0.Sequence.import_file(seq_file)
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=False)
    # @title 3. Plot sequence and signal
    # sp_adc, t_adc = mr0.util.pulseq_plot(seq=seq, signal=signal.numpy())

    # Unfortunately, we need to limit the resolution as reco_adjoint is very RAM-hungy
    print('reconstruct and plot')
    # seq0.plot_kspace_trajectory()
    kspace_test = seq0.get_kspace()
    organized_kspace = grid_kspace_2d(signal, kspace_test, crop_frequencies=False, half_fourier=True,
                                      grid_size=(brain_phantom_res, brain_phantom_res))
    organized_kspace = organized_kspace.type(torch.complex64)
    # plt.imshow(np.log(np.abs(organized_kspace) + 1))
    # plt.show()

    if use_pocs:
        organized_kspace = organized_kspace.numpy()
        organized_kspace = pocs_pf(organized_kspace, 10)
        organized_kspace = torch.tensor(organized_kspace)
        organized_kspace = organized_kspace.squeeze()

    # # stack the coil sensitivity maps and space after coil\
    # kspace_from_coil_stack = []
    # for idx, coil in enumerate(coil_sensitivity_maps):
    #     spectrum = torch.fft.ifftshift(organized_kspace)
    #     space = torch.fft.fft2(spectrum)
    #     space = torch.fft.ifftshift(space)
    #     space = space * coil
    #     organized_kspace_coil = image_to_kspace(space)
    #     kspace_from_coil_stack.append(organized_kspace_coil)
    #
    #     # plt.figure()
    #     # plt.subplot(131)
    #     # # plot kspace also
    #     # plt.title(f"Kspace_coil_{idx}")
    #     # plt.imshow(torch.log(organized_kspace_coil.abs() + 1), origin="lower")
    #     # plt.colorbar()
    #     #
    #     # plt.subplot(132)
    #     # plt.title(f"Magnitude_ coil_{idx}")
    #     # plt.imshow(space.abs(), origin="lower", cmap="gray")
    #     # plt.colorbar()
    #     # plt.subplot(133)
    #     # plt.title(f"Phase_coil_{idx}")
    #     # plt.imshow(space.angle(), origin="lower",cmap="gray", vmin=-np.pi, vmax=np.pi)
    #     # plt.colorbar()
    #     # plt.show()

    # # stack the coil sensitivity maps and space after coil
    # kspace_from_coil_stack = torch.stack(kspace_from_coil_stack)
    #
    # from pygrappa import grappa
    #
    # # GRAPPA reconstruction
    # res = grappa(kspace_from_coil_stack.numpy(), coil_sensitivity_maps, kernel_size=(5, 5), coil_axis=0)
    #
    # # display each kspace from each coil
    # for k_space in res:
    #     plt.figure()
    #     plt.subplot(141)
    #     plt.title("k-space Magnitude")
    #     plt.imshow(np.log(np.abs(k_space) + 1), origin="lower")
    #     plt.colorbar()
    #     plt.subplot(142)
    #     plt.title("k-space Phase")
    #     plt.imshow(np.angle(k_space), origin="lower", vmin=-np.pi, vmax=np.pi)
    #     plt.colorbar()
    #     plt.show()
    #
    #     # # plot the final image
    #     image = kspace_to_image(torch.Tensor(k_space))
    #     plt.subplot(143)
    #     plt.title("Image Magnitude")
    #     plt.imshow(image.abs(), origin="lower", cmap="gray")
    #     plt.colorbar()
    #     plt.subplot(144)
    #     plt.title("Image Phase")
    #     plt.imshow(image.angle(), origin="lower", cmap="gray", vmin=-np.pi, vmax=np.pi)
    #     plt.colorbar()
    #     plt.show()


    print("here")
    # # plot ksapce (if only coil exists)
    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")
    plt.imshow(torch.log(organized_kspace.abs() + 1), origin="lower")
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

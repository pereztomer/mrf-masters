import MRzeroCore as mr0
import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp
import torch
import scipy.io as sio

def main():
    # Load the sequence
    seq_file = r"sequences/2025-05-15_epi_Nx_128_Ny_128_part_fourier_factor_1_R1_repetetions_1.seq"
    seq = pp.Sequence()
    seq.read(seq_file)

    # Load the phantom
    obj_p = mr0.VoxelGridPhantom.load_mat("numerical_brain_cropped.mat")
    obj_p = obj_p.interpolate(128, 128, 1)


    # Define coil sensitivity maps
    num_coils = 8
    resolution = (128, 128, 1)
    fov = (0.22, 0.22, 1)

    # Generate synthetic coil sensitivity maps
    x = np.linspace(-1, 1, resolution[0])
    y = np.linspace(-1, 1, resolution[1])
    xx, yy = np.meshgrid(x, y)

    coil_maps = np.zeros((num_coils, *resolution), dtype=complex)
    for c in range(num_coils):
        # Use a Gaussian profile with phase variation for each coil
        phase_offset = np.exp(-1j * c * 2 * np.pi / num_coils)
        spatial_profile = np.exp(-(xx**2 + yy**2))
        coil_maps[c, :, :, 0] = phase_offset * spatial_profile

    # Set the coil maps in the phantom
    obj_p.coil_sens = torch.Tensor(coil_maps)
    obj_p = obj_p.build()
    # Simulate the sequence
    seq0 = mr0.Sequence.import_file(seq_file)
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=True)

    # Assuming `signal` is a torch tensor with shape (time_points, num_coils)
    time_points, num_coils = signal.shape

    # Calculate the number of phase encoding steps
    num_phase_encodings = time_points // 184

    # Verify that the total number of time points is correct
    assert time_points == 184 * num_phase_encodings, f"Unexpected number of time points: {time_points}"

    # Reshape the signal
    reshaped_signal = signal.view(184, num_coils, num_phase_encodings)
    signal_numpy = reshaped_signal.cpu().numpy()
    sio.savemat("simulated_signal.mat", {"signal": signal_numpy})
    print("Reshaped Signal Shape:", reshaped_signal.shape)  # Should be (184, 20, num_phase_encodings)
    exit()

    # Reconstruct the image
    reco = mr0.reco_adjoint(signal, seq0.get_kspace(), resolution=resolution, FOV=fov)

    # Plot the results
    plt.figure(figsize=(18, 6))

    # Magnitude
    plt.subplot(131)
    plt.title("Magnitude")
    plt.imshow(reco[:, :, 0].T.abs(), origin="lower", cmap='gray')
    plt.colorbar()

    # Phase
    plt.subplot(132)
    plt.title("Phase")
    plt.imshow(reco[:, :, 0].T.angle(), origin="lower", vmin=-np.pi, vmax=np.pi, cmap='twilight')
    plt.colorbar()

    # K-space
    reco_img = reco[:, :, 0].T
    kspace_from_reco = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(reco_img)))
    plt.subplot(133)
    plt.title("K-space from reconstructed image")
    plt.imshow(np.abs(np.log(kspace_from_reco) + 1), cmap='gray')
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()

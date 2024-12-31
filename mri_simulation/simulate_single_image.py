import numpy as np
import os
import MRzeroCore as mr0
import MRzeroCore
import matplotlib.pyplot as plt
from numpy import pi
import torch

def spiral_sampling3(kspace, num_points):
    """
    Spirally sample the given k-space tensor.

    Args:
        kspace (torch.Tensor): The k-space data as a complex tensor of shape [H, W].
        num_points (int): The number of points along the spiral trajectory.

    Returns:
        torch.Tensor: The spirally sampled k-space tensor.
    """
    # Get the dimensions of the k-space
    H, W = kspace.shape
    center = (H // 2, W // 2)

    # Create an empty mask for sampling
    mask = torch.zeros((H, W), dtype=torch.bool)

    # Generate a single continuous spiral trajectory with variable spacing
    theta_max = 12 * np.pi  # Increase the number of turns for a denser spiral
    a = 0.096  # Controls the rate at which the gap between turns increases
    b = 0.01 # Controls the density of points at the center
    for t in np.linspace(0, theta_max, num_points):
        # r = np.exp((a + b * t)) * t  # Radius grows with t, with denser points near the center
        r = np.exp(a * t)  # Radius grows with t, with denser points near the center
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))

        # Ensure the coordinates are within bounds
        if 0 <= x < H and 0 <= y < W:
            mask[x, y] = True

    # Apply the mask to the k-space to get the sampled k-space
    sampled_kspace = torch.zeros_like(kspace)
    sampled_kspace[mask] = kspace[mask]

    return sampled_kspace


import torch
from math import pi, cos, sin

def build_spiral_seq() -> mr0.Sequence:
    seq = mr0.Sequence()

    # Define the spiral parameters
    num_samples = 1024  # Total number of k-space samples
    num_turns = 10      # Number of turns in the spiral

    # Create a single repetition (only one excitation pulse)
    rep = seq.new_rep(1 + num_samples)

    # Set the 90-degree excitation pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 90 * pi / 180  # 90 degrees
    rep.pulse.phase = 0  # Phase is set to 0 for simplicity
    rep.event_time[0] = 2e-3  # Pulse duration

    # Calculate the gradient waveforms for spiral sampling
    gamma = 42.58e6  # Gyromagnetic ratio in Hz/T (for hydrogen)
    max_grad = 33e-3  # Maximum gradient in T/m

    for i in range(1, num_samples + 1):
        t = i * 0.08e-3  # Time increment, e.g., 80 microseconds
        angle = 2 * pi * num_turns * (t / (num_samples * 0.08e-3))  # Angular position in spiral

        # Calculate gradients for spiral trajectory
        g_x = max_grad * (t / (num_samples * 0.08e-3)) * cos(angle)
        g_y = max_grad * (t / (num_samples * 0.08e-3)) * sin(angle)

        rep.event_time[i] = 0.08e-3  # Readout sampling time
        rep.gradm[i, 0] = g_x
        rep.gradm[i, 1] = g_y
        rep.adc_usage[i] = 1
        rep.adc_phase[i] = pi  # Setting ADC phase

    return seq



def build_spiral_seq2() -> mr0.Sequence:
    seq = mr0.Sequence()

    # Define spiral parameters
    num_samples = 96  # Total number of k-space samples
    theta_max = 12 * pi  # Increase the number of turns for a denser spiral
    a = 0.096  # Controls the rate at which the gap between turns increases

    # Create a single repetition (only one excitation pulse)
    rep = seq.new_rep(1 + num_samples)

    # Set the 90-degree excitation pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 90 * pi / 180  # 90 degrees
    rep.pulse.phase = 0  # Phase is set to 0 for simplicity
    rep.event_time[0] = 2e-3  # Pulse duration

    # Set the ADC readout phase and duration
    adc_phase = pi

    # Loop over each point in the spiral
    for i, t in enumerate(torch.linspace(0, theta_max, num_samples)):
        # Compute the radius as a function of t (variable spacing)
        r = np.exp(a * t.item())  # Radius grows with t, exponential growth

        # Calculate gradient waveforms for spiral sampling in x and y directions
        g_x = r * cos(t.item())
        g_y = r * sin(t.item())

        print("g_x, g_y", g_x, g_y)
        # Set event time for readout sampling
        rep.event_time[i + 1] = 0.08e-4  # Readout sampling time

        # Set gradients for the x and y directions to create the spiral
        rep.gradm[i + 1, 0] = g_x
        rep.gradm[i + 1, 1] = g_y

        # Mark ADC usage for each sample point in the spiral
        rep.adc_usage[i + 1] = 1
        rep.adc_phase[i + 1] = adc_phase

    return seq


def build_spiral_seq3() -> mr0.Sequence:
    seq = mr0.Sequence()

    # Define spiral parameters
    num_samples = 1024  # Total number of k-space samples
    num_turns = 3       # Number of turns in the spiral
    max_radius_x = 100  # Maximum radius for k_x (from -30 to 100, total span is 130)
    max_radius_y = 30   # Maximum radius for k_y (from -30 to 30, total span is 60)

    # Create a single repetition (only one excitation pulse)
    rep = seq.new_rep(1 + num_samples)

    # Set the 90-degree excitation pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 90 * pi / 180  # 90 degrees
    rep.pulse.phase = 0  # Phase is set to 0 for simplicity
    rep.event_time[0] = 2e-3  # Pulse duration

    # Set the ADC readout phase and duration
    adc_phase = pi

    # Loop over each point in the spiral
    for i, t in enumerate(torch.linspace(0, 2 * pi * num_turns, num_samples)):
        # Compute radius as a function of t to create a smooth spiral
        r_fraction = i / num_samples  # Fraction of maximum radius, grows linearly from 0 to 1

        # Calculate x and y values in k-space (scale to fit within the specified bounds)
        g_x = r_fraction * max_radius_x * cos(t.item())
        g_y = r_fraction * max_radius_y * sin(t.item())

        # Set event time for readout sampling
        rep.event_time[i + 1] = 0.08e-3  # Readout sampling time

        # Set gradients for the x and y directions to create the spiral
        rep.gradm[i + 1, 0] = g_x
        rep.gradm[i + 1, 1] = g_y

        # Mark ADC usage for each sample point in the spiral
        rep.adc_usage[i + 1] = 1
        rep.adc_phase[i + 1] = adc_phase

    return seq


def build_seq() -> mr0.Sequence:
    seq = mr0.Sequence()

    enc = torch.randperm(64) - 32

    for i in range(64):
        rep = seq.new_rep(2 + 64 + 1)
        rep.pulse.usage = mr0.PulseUsage.EXCIT
        rep.pulse.angle = 7 * pi / 180
        rep.pulse.phase = 0.5 * 137.50776405 * (i ** 2 + i + 2) * pi / 180

        rep.event_time[0] = 2e-3  # Pulse
        rep.event_time[1] = 2e-3  # Rewinder
        rep.event_time[2:-1] = 0.08e-3  # Readout
        rep.event_time[-1] = 2e-3  # Spoiler

        rep.gradm[1, 0] = -33
        rep.gradm[2:-1, 0] = 1
        rep.gradm[-1, 0] = 96 - 31

        # Linear reordered phase encoding
        rep.gradm[1, 1] = i - 32
        # rep.gradm[1, 1] = i // 2 if i % 2 == 0 else -(i + 1) // 2
        # rep.gradm[1, 1] = enc[i]
        rep.gradm[-1, 1] = -rep.gradm[1, 1]

        rep.adc_usage[2:-1] = 1
        rep.adc_phase[2:-1] = pi - rep.pulse.phase

    return seq


def phantom_motion(time: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    time /= 0.712  # Sequence duration

    phi = 0.8 * time
    x = -0.03 * time ** 2
    y = 0 * time

    phi = torch.as_tensor(phi)
    cos = torch.cos(phi)
    sin = torch.sin(phi)

    # We can't construct tensors directly as this would remove gradients
    offset = torch.zeros(time.numel(), 3)
    offset[:, 0] = x
    offset[:, 1] = y

    rot = torch.zeros(time.numel(), 3, 3)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = sin
    rot[:, 1, 0] = -sin
    rot[:, 1, 1] = cos
    rot[:, 2, 2] = 1

    return rot, offset


def build_spiral_seq4() -> mr0.Sequence:
    seq = mr0.Sequence()

    # Target resolution
    resolution_x, resolution_y = 64, 64  # We need exactly 64x64 resolution in k-space

    # Number of samples to match resolution
    num_samples = resolution_x * resolution_y  # Total number of k-space samples for full coverage

    # Define spiral parameters
    num_turns = 3  # Number of turns in the spiral, can be adjusted
    max_radius_x = 32  # Radius to match half the resolution size for x direction
    max_radius_y = 32  # Radius to match half the resolution size for y direction

    # Create a single repetition (one excitation pulse and readout)
    rep = seq.new_rep(1 + num_samples)

    # Set the 90-degree excitation pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 90 * pi / 180  # 90 degrees
    rep.pulse.phase = 0  # Phase is set to 0 for simplicity
    rep.event_time[0] = 2e-3  # Pulse duration

    # Set the ADC readout phase and duration
    adc_phase = pi

    # Loop over each point to create a properly bounded spiral trajectory
    for i, t in enumerate(torch.linspace(0, 2 * pi * num_turns, num_samples)):
        # Compute radius in a linear function to keep the spiral bounded
        r_fraction = i / num_samples  # Fraction of max radius, grows linearly

        # Calculate k-space positions in x and y directions
        kx = r_fraction * max_radius_x * cos(t.item())
        ky = r_fraction * max_radius_y * sin(t.item())

        # Make sure kx and ky are in the desired bounds (-32, 32)
        g_x = kx
        g_y = ky

        # Set event time for readout sampling
        rep.event_time[i + 1] = 0.08e-3  # Readout sampling time

        # Set gradient waveforms to sample k-space at the computed (kx, ky)
        rep.gradm[i + 1, 0] = g_x
        rep.gradm[i + 1, 1] = g_y

        # Mark ADC usage at each k-space point
        rep.adc_usage[i + 1] = 1
        rep.adc_phase[i + 1] = adc_phase

    return seq
def main():
    folder_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos processed\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00002\numpy_files"
    # read all maps from the folder
    m0_map = torch.tensor(np.load(os.path.join(folder_path, "IMG-0004-00002_m0_map.npy")).T).unsqueeze(-1).expand(-1, -1, 2)
    t1_map = torch.tensor(np.load(os.path.join(folder_path, "IMG-0004-00002_t1_map.npy")).T).unsqueeze(-1).expand(-1, -1, 2)
    t2_map = torch.tensor(np.load(os.path.join(folder_path, "IMG-0004-00002_t2_map.npy")).T).unsqueeze(-1).expand(-1, -1, 2)

    m0_map = torch.flip(m0_map, dims=[1])
    t1_map = torch.flip(t1_map, dims=[1])
    t2_map = torch.flip(t2_map, dims=[1])
    # Build the default FLASH and show the kspace
    # seq = build_spiral_seq2()
    # seq.plot_kspace_trajectory()
    # exit()
    # seq = build_spiral_seq3()
    # seq.plot_kspace_trajectory()

    seq = build_seq()
    seq.plot_kspace_trajectory()
    # exit()
    # Until now, the sequence uses normalized grads: The simulation will adapt them
    # to the phantom size. If we want to hardcode a fixed FOV instead, we can do so:
    seq.normalized_grads = False
    for rep in seq:
        rep.gradm[:] /= 200e-3  # 200 mm FOV

    # convert all key from numpy to tensor

    t2_dash = torch.ones(m0_map.shape) * 0.1
    diffusion_map = torch.ones(m0_map.shape) * 0.001
    from MRzeroCore.phantom.voxel_grid_phantom import generate_B0_B1
    B0, B1 = generate_B0_B1(m0_map)
    coil_sensitivity = torch.ones(m0_map.shape)
    phantom = mr0.VoxelGridPhantom(PD=m0_map,
                                   T1=t1_map,
                                   T2=t2_map,
                                   T2dash=t2_dash,
                                   D=diffusion_map,
                                   B0=B0,
                                   B1=B1,
                                   coil_sens=coil_sensitivity,
                                   size=torch.tensor([0.192, 0.192, 0.192])
                                   )

    phantom = phantom.slices([1])
    phantom.plot()

    data = phantom.build()

    # data.phantom_motion = phantom_motion

    # Simulate the sequence

    graph = mr0.compute_graph(seq, data)
    signal = mr0.execute_graph(graph, seq, data)
    reco = mr0.reco_adjoint(signal, seq.get_kspace())

    reco_test = reco.abs().cpu()[:, :, 0]
    #min max normalizarion
    reco_test = (reco_test - reco_test.min()) / (reco_test.max() - reco_test.min())
    flipped_reco = torch.flip(reco_test, dims=[0])
    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")

    # plt.imshow(flipped_image, origin='lower', vmin=0, cmap='gray')
    # plt.imshow(reco_test, origin='lower', vmin=0)
    plt.imshow(reco_test)
    # plt.imshow(reco.abs().cpu()[:, :, 0].T, cmap='gray')
    plt.subplot(122)
    plt.title("Phase")
    plt.imshow(reco.angle().cpu()[:, :, 0].numpy(), vmin=-np.pi, vmax=np.pi, cmap="twilight")
    plt.show()
    plt.figure(figsize=(7, 5), dpi=120)
    graph.plot()
    plt.grid()
    plt.show()

    exit()
    # squise access dim
    reco = reco.squeeze(2)
    kspace = torch.fft.fft2(reco)

    # Optionally, you may want to shift the zero-frequency component to the center
    kspace_shifted = torch.fft.fftshift(kspace)


    sampled_kspace = spiral_sampling3(kspace_shifted, num_points=5000)

    # Plot the sampled k-space
    kspace_magnitude = torch.abs(sampled_kspace)
    kspace_log_magnitude = torch.log(1 + kspace_magnitude)

    plt.figure(figsize=(8, 8))
    plt.imshow(kspace_log_magnitude.numpy(), cmap='gray')
    plt.title('Radially Sampled K-Space (Log Magnitude)')
    plt.axis('off')
    plt.show()

    # Perform inverse FFT to get the image domain representation
    reconstructed_image = torch.fft.ifftshift(sampled_kspace)
    reconstructed_image = torch.fft.ifft2(reconstructed_image)

    # Calculate magnitude and phase of the reconstructed image
    image_magnitude = torch.abs(reconstructed_image)
    image_phase = torch.angle(reconstructed_image)

    # Plot the magnitude and phase of the reconstructed image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    # plt.imshow(image_magnitude.numpy().T,origin='lower', vmin=0)
    plt.imshow(image_magnitude.numpy(),cmap='gray')
    plt.title('Image (Magnitude)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # plt.imshow(image_phase.numpy().T,  origin='lower', vmin=-np.pi, vmax=np.pi, cmap="twilight")
    plt.imshow(image_phase.numpy(),  cmap="gray")
    plt.title('Image (Phase)')
    plt.axis('off')

    plt.show()




if __name__ == '__main__':
    main()

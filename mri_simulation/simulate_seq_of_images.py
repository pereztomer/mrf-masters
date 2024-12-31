import torch
import numpy as np
import os
from simulate_single_image import build_seq, spiral_sampling3
import matplotlib.pyplot as plt
import MRzeroCore as mr0
from MRzeroCore.phantom.voxel_grid_phantom import generate_B0_B1
from apply_registration_map import deform_image_torch
import cv2
import torchvision.transforms as transforms


def main():
    registration_folder_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos processed\registration_maps"
    dest_folder = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos processed\temp_folder"
    os.makedirs(dest_folder, exist_ok=True)
    print(dest_folder)
    masks_folder_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos processed\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00002\numpy_files"
    # read all maps from the folder
    m0_map = torch.tensor(np.load(os.path.join(masks_folder_path, "IMG-0004-00002_m0_map.npy"))).unsqueeze(-1).expand(-1, -1,2)
    t1_map = torch.tensor(np.load(os.path.join(masks_folder_path, "IMG-0004-00002_t1_map.npy"))).unsqueeze(-1).expand(-1, -1,2)
    t2_map = torch.tensor(np.load(os.path.join(masks_folder_path, "IMG-0004-00002_t2_map.npy"))).unsqueeze(-1).expand(-1, -1,2)

    seq = build_seq()

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

    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")
    plt.imshow(reco.abs().cpu()[:, :, 0], origin='lower', vmin=0)
    # plt.imshow(reco.abs().cpu()[:, :, 0].T, cmap='gray')
    plt.subplot(122)
    plt.title("Phase")
    plt.imshow(reco.angle().cpu()[:, :, 0], origin='lower', vmin=-np.pi, vmax=np.pi, cmap="twilight")
    plt.show()
    plt.figure(figsize=(7, 5), dpi=120)
    graph.plot()
    plt.grid()
    plt.show()

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
    plt.imshow(image_magnitude.numpy(), cmap='gray')
    plt.title('Image (Magnitude)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # plt.imshow(image_phase.numpy().T,  origin='lower', vmin=-np.pi, vmax=np.pi, cmap="twilight")
    plt.imshow(image_phase.numpy(), cmap="gray")
    plt.title('Image (Phase)')
    plt.axis('off')

    plt.show()

    image_magnitude = image_magnitude.unsqueeze(0)
    image_phase = image_phase.unsqueeze(0)
    resize = transforms.Resize((256, 256))

    image_magnitude = resize(image_magnitude)
    image_phase = resize(image_phase)

    registration_map_files = [os.path.join(registration_folder_path, f) for f in os.listdir(registration_folder_path) if "registration" in f and f.endswith('.npy')]

    registration_map_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for idx, deformation_path in enumerate(registration_map_files):
        deformation_map = torch.tensor(np.load(os.path.join(registration_folder_path, deformation_path)))
        # apply deformation map to the image
        deformed_magnitude = deform_image_torch(image_magnitude[0], deformation_map)
        deformed_phase = deform_image_torch(image_phase[0], deformation_map)
        # save the deformed image as png
        plt.imsave(os.path.join(dest_folder, f"deformed_magnitude_{idx}.png"), deformed_magnitude.numpy(), cmap='gray')
        plt.imsave(os.path.join(dest_folder, f"deformed_phase_{idx}.png"), deformed_phase.numpy(), cmap="twilight")

        # np.save(os.path.join(dest_folder, f"deformed_magnitude_{idx}"), deformed_magnitude.numpy())
        # np.save(os.path.join(dest_folder, f"deformed_phase_{idx}"), deformed_phase.numpy())


    video_filename = os.path.join(dest_folder, 'mri_signal_test.mp4')
    video_fps = 24  # Frames per second

    # Get list of image files
    image_files = [os.path.join(dest_folder, f) for f in sorted(os.listdir(dest_folder)) if f.endswith('.png') and 'deformed_magnitude' in f]

    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Read the first image to get the size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    print(f'Video saved as {video_filename}')

    video_filename = os.path.join(dest_folder, 'mri_phase.mp4')
    video_fps = 24  # Frames per second

    # Get list of image files
    image_files = [os.path.join(dest_folder, f) for f in sorted(os.listdir(dest_folder)) if
                   f.endswith('.png') and 'deformed_phase' in f]

    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Read the first image to get the size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    print(f'Video saved as {video_filename}')




if __name__ == '__main__':
    main()
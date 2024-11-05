import torch
import numpy as np
import matplotlib.pyplot as plt
from chaos_ds.find_countor import process_image
from select_points import select_random_points_within_contour
from backward_distortion_utils import create_transform_matrices, gaussian_weight, forward_transform
import os
import cv2
from tqdm import tqdm

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# PyTorch version of backward transformation for multiple time-distorted images in a batch
def backward_transform_batch(control_points, transformations_batch, q_prime_grid_batch, initial_guess_grid, sigma=1,
                             lr=2, num_iters=100):
    batch_size, height, width, _ = q_prime_grid_batch.shape

    # Flatten the grid of points (batch_size x height*width x 2)
    q_prime_flat_batch = q_prime_grid_batch.view(batch_size, -1, 2).to(device)
    q_batch = initial_guess_grid.view(batch_size, -1, 2).to(device).requires_grad_(True)

    # Use an optimizer (SGD or Adam)
    optimizer = torch.optim.Adam([q_batch], lr=lr)

    for _ in tqdm(range(num_iters)):
        optimizer.zero_grad()

        # Compute the objective function for all pixels in the batch
        loss = objective_function_batch(q_batch, control_points, transformations_batch, q_prime_flat_batch, sigma)

        # Backpropagate
        loss.backward()

        # Update q_batch
        optimizer.step()

    return q_batch.view(batch_size, height, width, 2).detach()


# Objective function for batch processing
def objective_function_batch(q_batch, control_points, transformations_batch, q_prime_flat_batch, sigma):
    """
    Vectorized objective function that computes the loss over a batch of images.

    Args:
        q_batch (torch.Tensor): Tensor of size (batch_size x height*width x 2), current guess for all points.
        control_points (torch.Tensor): Control points (k x 2).
        transformations_batch (list of torch.Tensor): List of 3x3 transformation matrices for the batch.
        q_prime_flat_batch (torch.Tensor): Flattened tensor of the transformed points (batch_size x height*width x 2).
        sigma (float): Sigma for the Gaussian weight.

    Returns:
        torch.Tensor: The loss (summed squared distances for each image in the batch).
    """
    q_estimated_batch = forward_transform_batch(control_points, transformations_batch, q_batch, sigma)

    return torch.sum((q_prime_flat_batch - q_estimated_batch) ** 2)


# Forward transformation for a batch of images
def forward_transform_batch(control_points, transformations_batch, q_batch, sigma=1):
    """
    Apply the forward transformation using Gaussian weighted control points to a batch of images.

    Args:
        control_points (torch.Tensor): Control points (k x 2).
        transformations_batch (list of torch.Tensor): List of 3x3 transformation matrices.
        q_batch (torch.Tensor): Tensor of size (batch_size x height*width x 2), current points.
        sigma (float): Sigma for the Gaussian weight.

    Returns:
        torch.Tensor: Transformed points (batch_size x height*width x 2).
    """
    k = control_points.shape[0]
    batch_size = q_batch.shape[0]

    # Compute weights for all pixels in the batch (batch_size x height*width x k)
    weights_batch = torch.stack([gaussian_weight_batch(q_batch, control_points[i], sigma) for i in range(k)], dim=-1)
    weights_batch /= weights_batch.sum(dim=-1, keepdim=True)  # Normalize weights (batch_size x height*width x k)

    # Convert q_batch to homogeneous coordinates (batch_size x height*width x 3)
    q_homogeneous_batch = torch.cat([q_batch, torch.ones(batch_size, q_batch.shape[1], 1, device=device)], dim=-1)

    # Apply the transformation to each control point in the batch
    transformed_q_batch = torch.zeros_like(q_homogeneous_batch)
    for i in range(k):
        transformed_q_batch += weights_batch[:, :, i:i + 1] * (
                    q_homogeneous_batch @ transformations_batch[:, i].permute(0, 2, 1))

    return transformed_q_batch[:, :, :2]  # Return only x and y coordinates


# Gaussian weight calculation for a batch
# Gaussian weight calculation for a batch
def gaussian_weight_batch(q_batch, p, sigma=1):
    # Ensure that p is a PyTorch tensor
    if isinstance(p, np.ndarray):
        p = torch.tensor(p, dtype=torch.float32).to(device)

    # Compute the distance between q_batch and control point p
    dist_batch = torch.norm(q_batch - p.unsqueeze(0).unsqueeze(1), dim=-1)
    return torch.exp(-0.5 * (dist_batch ** 2) / sigma ** 2)


# def create_time_varying_transformations(control_points, width, height, l):
#     center_x, center_y = width // 2, height // 2  # Assuming the center is at the middle of the image
#     parameter_sequences = []
#
#     for idx, point in enumerate(control_points):
#         # Calculate direction to move outwards (based on the position relative to the center)
#         move_x = point[0] - center_x  # Horizontal distance from the center
#         move_y = point[1] - center_y  # Vertical distance from the center
#
#         # Outward movement: moving away from the center at t=0
#         tx_start = move_x * 0.2  # 20% of the distance outward
#         ty_start = move_y * 0.2  # 20% of the distance outward
#
#         # Return to original position at the end
#         tx_end = 0  # At the end, the point returns to its original position
#         ty_end = 0  # Same for vertical movement
#
#         # Other transformation parameters can remain relatively static
#         sx_start, sy_start = 1.0, 1.0  # No scaling changes
#         theta_start = 0  # No rotation
#         shear_x_start, shear_y_start = 0, 0  # No shear
#
#         sx_end, sy_end = 1.0, 1.0  # Return to original scaling
#         theta_end = 0  # No rotation change
#         shear_x_end, shear_y_end = 0, 0  # Return to no shear
#
#         # Generate sequences over time
#         sx_seq = np.linspace(sx_start, sx_end, l)
#         sy_seq = np.linspace(sy_start, sy_end, l)
#         theta_seq = np.linspace(theta_start, theta_end, l)
#         shear_x_seq = np.linspace(shear_x_start, shear_x_end, l)
#         shear_y_seq = np.linspace(shear_y_start, shear_y_end, l)
#         tx_seq = np.linspace(tx_start, tx_end, l)
#         ty_seq = np.linspace(ty_start, ty_end, l)
#
#         parameter_sequences.append({
#             'sx': sx_seq,
#             'sy': sy_seq,
#             'theta': theta_seq,
#             'shear_x': shear_x_seq,
#             'shear_y': shear_y_seq,
#             'tx': tx_seq,
#             'ty': ty_seq
#         })
#
#     return parameter_sequences

def create_time_varying_transformations(control_points, width, height,wavelength, seq_length):
    """
    Generate smooth time-varying transformations for control points.

    Args:
        control_points (np.ndarray): Coordinates of control points (k x 2).
        width (int): Image width.
        height (int): Image height.
        wavelength (int): l is the distance over which the wave's shape repeats.
        seq_length (int): Number of time steps in the sequence.

    Returns:
        List of parameter sequences for each control point.
    """
    assert seq_length <= wavelength, "Sequence length must be shorter or equal than or equal to the wavelength."
    center_x, center_y = width // 2, height // 2  # Assume the center of the image
    parameter_sequences = []
    frequency = 2 * np.pi / wavelength  # Frequency for smooth oscillations (full period within the time steps)

    for idx, point in enumerate(control_points):

        # Calculate direction to move outwards (based on the position relative to the center)
        move_x = center_x - point[0]   # Horizontal distance from the center
        move_y = center_y - point[1]   # Vertical distance from the center
        # # take sign of move_x and move_y
        # dir_x = np.sign(move_x)
        # dir_y = np.sign(move_y)
        # Smooth periodic translation using sine wave
        tx_seq = move_x*0.6 * np.sin(frequency * np.arange(wavelength))[:seq_length]  # Smooth oscillation for X translation
        ty_seq = move_y*0.6 * np.sin(frequency * np.arange(wavelength))[:seq_length]  # Smooth oscillation for Y translation

        # Apply very subtle periodic scaling and no rotation/shear
        # sx_seq = 1.0 + 0.01 * np.sin(frequency * np.arange(l))  # Subtle scaling in X
        sx_seq = np.ones(wavelength)[:seq_length]
        # sy_seq = 1.0 + 0.01 * np.sin(frequency * np.arange(l))  # Subtle scaling in Y
        sy_seq = np.ones(wavelength)[:seq_length]
        theta_seq = np.zeros(wavelength)[:seq_length]  # No rotation
        shear_x_seq = np.zeros(wavelength)[:seq_length]  # No shear in X
        shear_y_seq = np.zeros(wavelength)[:seq_length]  # No shear in Y

        parameter_sequences.append({
            'sx': sx_seq,
            'sy': sy_seq,
            'theta': theta_seq,
            'shear_x': shear_x_seq,
            'shear_y': shear_y_seq,
            'tx': tx_seq,
            'ty': ty_seq
        })

    return parameter_sequences

def process_images_batch(m0_map, output_dir,seq_len, batch_size=4, l=250, sigma=25):
    """
    Process time-varying images in batches on GPU.
    """
    height, width = m0_map.shape

    # Process the image to get the contour mask and control points
    contour_mask = process_image(m0_map)
    control_points, _, _ = select_random_points_within_contour(m0_map, contour_mask)

    # Prepare parameter sequences for each control point over time
    parameter_sequences = create_time_varying_transformations(control_points,
                                                              width,
                                                              height,
                                                              seq_length=seq_len,
                                                              wavelength=l)

    # Process the full time series in chunks (batch by batch)
    for batch_start in range(0, seq_len, batch_size):
        current_batch_size = min(batch_size, seq_len - batch_start)  # Adjust for the last smaller batch
        transformations_batch = []

        # Prepare transformations for the current batch
        for t in range(current_batch_size):
            time_step = batch_start + t
            transformations = []
            for idx in range(len(control_points)):
                params = parameter_sequences[idx]
                sx = params['sx'][time_step]
                sy = params['sy'][time_step]
                theta = params['theta'][time_step]
                shear_x = params['shear_x'][time_step]
                shear_y = params['shear_y'][time_step]
                tx = params['tx'][time_step]
                ty = params['ty'][time_step]

                A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
                transformations.append(A)
            transformations_batch.append(torch.tensor(transformations, dtype=torch.float32).to(device))

        transformations_batch = torch.stack(transformations_batch)

        # Prepare pixel coordinates for the batch
        i_coords, j_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        q_prime_grid_batch = torch.stack((i_coords, j_coords), dim=-1).float().expand(current_batch_size, -1, -1,
                                                                                      -1).to(device)

        # Prepare initial guess grid for the batch
        initial_guess_grid = torch.tensor([height // 2, width // 2],
                                          dtype=torch.float32).expand(current_batch_size,height, width, 2).to(device)

        # Perform the backward transformation for the current batch
        recovered_q_batch = backward_transform_batch(control_points, transformations_batch, q_prime_grid_batch,
                                                     initial_guess_grid, sigma=sigma)

        # Process results and save images for each image in the current batch
        for batch_idx in range(current_batch_size):
            os.makedirs(output_dir, exist_ok=True)

            # Reconstruct the new image from the recovered points
            new_image = np.zeros(shape=m0_map.shape)
            recovered_q = recovered_q_batch[batch_idx].int().cpu().numpy()

            for i in range(height):
                for j in range(width):
                    q = recovered_q[i, j]
                    if 0 <= q[0] < height and 0 <= q[1] < width:
                        new_image[i, j] = m0_map[q[0], q[1]]

            # save recovered_q as npy
            np.save(f"{output_dir}/registration_map_{batch_start + batch_idx}.npy", recovered_q)
            # save image as numpy
            np.save(f"{output_dir}/distorted_m0_map_{batch_start + batch_idx}.npy", new_image)
            # # Save the final image
            # plt.imsave(f"{output_dir}/distorted_m0_map_{batch_start + batch_idx}.png", new_image, cmap='gray', vmin=0, vmax=1)

        print(f"Processed and saved images for batch starting from time step {batch_start}.")

    # video_filename = os.path.join(output_dir, 'distorted_m0_map_video.mp4')
    # video_fps = 24  # Frames per second
    #
    # # Get list of image files
    # image_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]
    #
    # image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #
    # # Read the first image to get the size
    # frame = cv2.imread(image_files[0])
    # height, width, layers = frame.shape
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))
    #
    # for image_file in image_files:
    #     frame = cv2.imread(image_file)
    #     video.write(frame)
    #
    # video.release()
    # print(f'Video saved as {video_filename}')


if __name__ == '__main__':
    # this is the full last motion simulation!
    # Single m0_map
    m0_map_path = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\chaos_ds\m0_map.npy"
    m0_map = np.load(m0_map_path)

    output_dir = r'C:\Users\perez\Desktop\masters\mri_research\datasets\distortion_dataset_8'
    process_images_batch(m0_map, output_dir,seq_len=100, batch_size=64, sigma=25)


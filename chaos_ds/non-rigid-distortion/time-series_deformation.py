import numpy as np
from chaos_ds.find_countor import process_image
from select_points import select_random_points_within_contour
from backward_distortion_utils import (
    create_transform_matrices,
    gaussian_weight,
    backward_transform,
    forward_transform
)
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import os
import cv2  # OpenCV library for video creation

np.random.seed(42)

def process_pixel(args):
    i, j, control_points, transformations_t, initial_guess, sigma, height, width, m0_map = args
    q_prime = np.array([i, j])
    q = backward_transform(control_points, transformations_t, q_prime, initial_guess, sigma=sigma)
    q = q.astype(int)
    if 0 <= q[0] < height and 0 <= q[1] < width:
        return (i, j, m0_map[q[0], q[1]])
    else:
        return (i, j, 0)

def main():
    # Load the M0 map
    map_path = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\chaos_ds\m0_map.npy"  # Update the path accordingly
    m0_map = np.load(map_path)

    # Process the image to get the contour mask
    contour_mask = process_image(m0_map)

    # Select random points within the contour
    control_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)
    height, width = m0_map.shape

    # Define time series length
    l = 250  # Number of time steps/images in the series

    # Define initial and final transformation parameters for each control point
    # parameter_sequences = []
    # for _ in range(len(control_points)):
    #     # Initial parameters
    #     sx_start = 0.9
    #     sy_start = 0.9
    #     theta_start = -np.pi / 6
    #     shear_x_start = -0.4
    #     shear_y_start = -0.4
    #     tx_start = 0
    #     ty_start = 0
    #
    #     # Final parameters (small change from the start)
    #     sx_end = sx_start + 0.4
    #     sy_end = sy_start + 0.4
    #     theta_end = np.pi / 6
    #     shear_x_end = 0.4
    #     shear_y_end = 0.4
    #     tx_end = 0
    #     ty_end = 0
    #
    #     # Generate sequences over time
    #     sx_seq = np.linspace(sx_start, sx_end, l)
    #     sy_seq = np.linspace(sy_start, sy_end, l)
    #     theta_seq = np.linspace(theta_start, theta_end, l)
    #     shear_x_seq = np.linspace(shear_x_start, shear_x_end, l)
    #     shear_y_seq = np.linspace(shear_y_start, shear_y_end, l)
    #     tx_seq = np.linspace(tx_start, tx_end, l)
    #     ty_seq = np.linspace(ty_start, ty_end, l)
    #
    #     parameter_sequences.append({
    #         'sx': sx_seq,
    #         'sy': sy_seq,
    #         'theta': theta_seq,
    #         'shear_x': shear_x_seq,
    #         'shear_y': shear_y_seq,
    #         'tx': tx_seq,
    #         'ty': ty_seq
    #     })
    center_x, center_y = width // 2, height // 2  # Assuming the center is at the middle of the image

    parameter_sequences = []
    for idx, point in enumerate(control_points):
        # Calculate direction to move outwards (based on the position relative to the center)
        move_x = point[0] - center_x  # Horizontal distance from the center
        move_y = point[1] - center_y  # Vertical distance from the center

        # Outward movement: moving away from the center at t=0
        tx_start = move_x * 0.2  # 10% of the distance outward
        ty_start = move_y * 0.2  # 10% of the distance outward

        # Return to original position at the end
        tx_end = 0  # At the end, the point returns to its original position
        ty_end = 0  # Same for vertical movement

        # Other transformation parameters can remain relatively static
        sx_start, sy_start = 1.0, 1.0  # No scaling changes
        theta_start = 0  # No rotation
        shear_x_start, shear_y_start = 0, 0  # No shear

        sx_end, sy_end = 1.0, 1.0  # Return to original scaling
        theta_end = 0  # No rotation change
        shear_x_end, shear_y_end = 0, 0  # Return to no shear

        # Generate sequences over time
        sx_seq = np.linspace(sx_start, sx_end, l)
        sy_seq = np.linspace(sy_start, sy_end, l)
        theta_seq = np.linspace(theta_start, theta_end, l)
        shear_x_seq = np.linspace(shear_x_start, shear_x_end, l)
        shear_y_seq = np.linspace(shear_y_start, shear_y_end, l)
        tx_seq = np.linspace(tx_start, tx_end, l)
        ty_seq = np.linspace(ty_start, ty_end, l)

        parameter_sequences.append({
            'sx': sx_seq,
            'sy': sy_seq,
            'theta': theta_seq,
            'shear_x': shear_x_seq,
            'shear_y': shear_y_seq,
            'tx': tx_seq,
            'ty': ty_seq
        })

    # Create directory to save images
    output_dir = 'output_images_7'
    os.makedirs(output_dir, exist_ok=True)

    # Generate the time series of images
    for t in range(l):
        print(f"Processing time step {t+1}/{l}")
        # For each control point, create the transformation matrix at time t
        transformations_t = []
        for idx in range(len(control_points)):
            params = parameter_sequences[idx]
            sx = params['sx'][t]
            sy = params['sy'][t]
            theta = params['theta'][t]
            shear_x = params['shear_x'][t]
            shear_y = params['shear_y'][t]
            tx = params['tx'][t]
            ty = params['ty'][t]

            A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
            transformations_t.append(A)

        # Prepare arguments for multiprocessing
        initial_guess = np.array([height // 2, width // 2])
        sigma = 25

        # Create a list of all pixel coordinates
        pixel_indices = [(i, j) for i in range(height) for j in range(width)]

        # Prepare arguments for each pixel
        args_list = [
            (i, j, control_points, transformations_t, initial_guess, sigma, height, width, m0_map)
            for i, j in pixel_indices
        ]

        # Calculate time
        start_time = time.time()

        # Use multiprocessing Pool to process pixels in parallel
        num_workers = cpu_count()
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_pixel, args_list)

        # Reconstruct the new image from results
        new_image = np.zeros((height, width))
        for i, j, value in results:
            new_image[i, j] = value

        plt.imsave(f"{output_dir}/distorted_m0_map_{t+1:03d}.png",
                   new_image,
                   cmap='gray',
                   vmin=0, vmax=1)

        # save the image
        print(f"Time taken for time step {t + 1}: {time.time() - start_time:.2f} seconds")

        # Compute the transformed control points at time t
        transformed_control_points = []
        for idx, p in enumerate(control_points):
            p = p.astype(float)
            transformed_p = forward_transform(control_points, transformations_t, p, sigma=sigma)
            transformed_control_points.append(transformed_p)
        transformed_control_points = np.array(transformed_control_points)

        # # Plot the distorted image with original and transformed control points
        # plt.figure()
        # plt.imshow(new_image, cmap='viridis')
        # plt.scatter(control_points[:, 1], control_points[:, 0], c='red', marker='o', label='Original Control Points')
        # plt.scatter(transformed_control_points[:, 1], transformed_control_points[:, 0], c='blue', marker='x', label='Transformed Control Points')
        # plt.legend()
        # plt.title(f'Distorted M0 Map with Control Points - Time Step {t+1}')

        # # Save the image to a file
        # output_filename = os.path.join(output_dir, f'distorted_m0_map_{t+1:03d}.png')
        # plt.savefig(output_filename)
        # plt.close()

    # After generating images, convert them into a video
    # Specify video parameters
    video_filename = 'distorted_m0_map_video_7.mp4'
    video_fps = 24  # Frames per second

    # Get list of image files
    image_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]

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

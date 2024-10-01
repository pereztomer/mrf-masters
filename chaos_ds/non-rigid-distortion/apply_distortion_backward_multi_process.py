import numpy as np
from chaos_ds.find_countor import process_image
# from chaos_ds.distortion import select_random_points_within_contour
from select_points import select_random_points_within_contour
from backward_distortion_utils import create_transform_matrices, gaussian_weight, backward_transform
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

np.random.seed(42)

def process_pixel(args):
    i, j, control_points, transformations, initial_guess, sigma, height, width, m0_map = args
    q_prime = np.array([i, j])
    q = backward_transform(control_points, transformations, q_prime, initial_guess, sigma=sigma)
    q = q.astype(int)
    if 0 <= q[0] < height and 0 <= q[1] < width:
        return (i, j, m0_map[q[0], q[1]])
    else:
        return (i, j, 0)

def main():
    map_path = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\chaos_ds\m0_map.npy"
    m0_map = np.load(map_path)

    contour_mask = process_image(m0_map)

    # Select random points within the contour and create 2D Gaussian distributions

    control_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)
    height, width = m0_map.shape

    # Define transformation parameters for each control point
    transformations = []
    for _ in range(len(control_points)):
        sx = np.random.uniform(0.99, 1.01)  # Random scaling factor for x
        sy = np.random.uniform(0.99, 1.01)  # Random scaling factor for y
        theta = np.random.uniform(-np.pi / 24, np.pi / 24)  # Random rotation angle
        shear_x = np.random.uniform(-0.1, 0.1)  # Random shearing factor for x
        shear_y = np.random.uniform(-0.1, 0.1)  # Random shearing factor for y
        tx = np.random.uniform(-5, 5)  # Random translation in x
        ty = np.random.uniform(-5, 5)  # Random translation in y

        print(f"Control point: sx={sx:.2f}, sy={sy:.2f}, theta={theta:.2f}, shear_x={shear_x:.2f}, shear_y={shear_y:.2f}, tx={tx:.2f}, ty={ty:.2f}")
        # Create the affine transformation matrix for this control point
        A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
        transformations.append(A)

    # Prepare arguments for multiprocessing
    initial_guess = np.array([height // 2, width // 2])
    sigma = 25

    # Create a list of all pixel coordinates
    pixel_indices = [(i, j) for i in range(height) for j in range(width)]

    # Prepare arguments for each pixel
    args_list = [
        (i, j, control_points, transformations, initial_guess, sigma, height, width, m0_map)
        for i, j in pixel_indices
    ]
    # calc time
    import time
    start = time.time()
    # Use multiprocessing Pool to process pixels in parallel
    num_workers = cpu_count()
    print(f"Using {num_workers} CPU cores for multiprocessing.")
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_pixel, args_list)

    # Reconstruct the new image from results
    new_image = np.zeros((height, width))
    for i, j, value in results:
        new_image[i, j] = value

    print(f"Time taken: {time.time() - start:.2f} seconds")
    plt.imshow(new_image, cmap='viridis')
    plt.title('Distorted M0 Map - pool')
    plt.show()


if __name__ == '__main__':
    main()

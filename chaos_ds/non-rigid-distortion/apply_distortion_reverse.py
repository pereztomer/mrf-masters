import numpy as np
from chaos_ds.find_countor import process_image
# from chaos_ds.distortion import select_random_points_within_contour
from backward_distortion_utils import create_transform_matrices, gaussian_weight, backward_transform
import matplotlib.pyplot as plt

np.random.seed(42)
def main():
    map_path = r"C:\Users\perez\Desktop\cargoseer\mrf-masters\chaos_ds\m0_map.npy"
    m0_map = np.load(map_path)

    contour_mask = process_image(m0_map)

    from select_points import select_random_points_within_contour
    # Select random points within the contour and create 2D Gaussian distributions
    control_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)
    height, width = m0_map.shape

    # Define transformation parameters for each control point
    transformations = []
    for i in range(len(control_points)):
        sx = np.random.uniform(0.9, 1.1)  # Random scaling factor for x
        sy = np.random.uniform(0.9, 1.1)  # Random scaling factor for y
        theta = np.random.uniform(-np.pi / 8, np.pi / 8)  # Random rotation angle
        shear_x = np.random.uniform(-0.2, 0.2)  # Random shearing factor for x
        shear_y = np.random.uniform(-0.2, 0.2)  # Random shearing factor for y
        tx = np.random.uniform(-10, 10)  # Random translation in x
        ty = np.random.uniform(-10, 10)  # Random translation in y

        # Create the affine transformation matrix for this control point
        A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
        transformations.append(A)

    import time
    start = time.time()


    # backward_transform(control_points, transformations, q_prime, initial_guess, sigma=1)
    new_image = np.zeros(shape=m0_map.shape)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            q_prime = np.array([i, j])
            # define initial guess as the middle of the image (height/2, width/2)
            initial_guess = np.array([height // 2, width // 2])
            q = backward_transform(control_points, transformations, q_prime, initial_guess, sigma=25)
            # convert to int
            q = q.astype(int)
            if 0 <= q[0] < height and 0 <= q[1] < width:
                new_image[q_prime[0], q_prime[1]] = m0_map[q[0], q[1]]

    print(f"Time taken: {time.time() - start:.2f} seconds")
    plt.imshow(new_image, cmap='viridis')
    plt.title('Distorted M0 Map')
    plt.show()



if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.ndimage import affine_transform
from torch import nn
# from chaos_ds.apply_affine_transformation import apply_affine_transformation_to_image
from chaos_ds.find_countor import (process_image,
                                   apply_manual_affine_transformation,
                                    interpolate_missing_values,
                                   apply_manual_affine_transformation_point)
import torch

# set seed
np.random.seed(42)
def select_random_points_within_contour(map_data, contour_mask, num_points=6):
    """
    Select random points from within the contour of the organ in the map,
    assign each a standard deviation, and create 2D Gaussian distributions.

    Parameters:
    - map_data: 2D numpy array representing the M0 map.
    - contour_mask: Binary mask indicating the region inside the contour.
    - num_points: Number of points to randomly select (default is 6).

    Returns:
    - selected_points: List of tuples (x, y) representing the selected points.
    - stds: List of tuples representing standard deviations (std_x, std_y) for each point.
    - gaussians: List of 2D Gaussian distributions based on the selected points and their stds.
    """
    # Get the indices of the points inside the contour
    inside_contour_indices = np.argwhere(contour_mask)

    # Randomly select num_points unique points from the contour
    selected_points = inside_contour_indices[np.random.choice(len(inside_contour_indices), num_points, replace=False)]

    # Assign a standard deviation to each selected point
    stds = np.random.uniform(1, 2, size=(num_points, 2))  # Example std range [1, 10] for both dimensions

    # Create 2D Gaussian distributions based on the selected points and their stds
    gaussians = []
    for i, (x, y) in enumerate(selected_points):
        mu = [y, x]  # Reversing x and y to align with image coordinates
        cov = np.diag(stds[i] ** 2)  # Covariance matrix for 2D Gaussian
        gaussian = multivariate_normal(mean=mu, cov=cov)
        gaussians.append(gaussian)

    return selected_points, stds, gaussians


def plot_gaussians_in_3d(selected_points, gaussians, map_shape):
    x, y = np.mgrid[0:map_shape[0], 0:map_shape[1]]
    pos = np.dstack((y, x))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (point, gaussian) in enumerate(zip(selected_points, gaussians)):
        z = gaussian.pdf(pos)  # Get the probability density for the 2D Gaussian

        # Plot the 3D surface for this Gaussian
        ax.plot_surface(y, x, z, rstride=3, cstride=3, cmap='viridis', edgecolor='none', alpha=0.7)
        ax.scatter(point[1], point[0], gaussian.pdf(point[::-1]), color='red', s=50, label=f'Point {i + 1}')

    ax.set_title('3D Plot of 2D Gaussian Distributions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    plt.show()

def apply_affine_transform(point, transformation_matrix):
    # Apply the affine transformation to the given point (x, y)
    point_homogeneous = np.array([point[0], point[1], 1])  # Convert to homogeneous coordinates
    transformed_point = np.dot(transformation_matrix, point_homogeneous)
    return transformed_point[:2]  # Return the x, y coordinates after transformation


def apply_affine_transformation_to_image(image, transformation_matrix):
    """
    Apply an affine transformation to the entire image.

    Parameters:
    - image: 2D numpy array representing the M0 map or any other image.
    - transformation_matrix: 3x3 affine transformation matrix.

    Returns:
    - transformed_image: The transformed image after applying the affine transformation.
    """
    # The affine_transform function requires a 2x2 matrix, so we extract the relevant part
    matrix = transformation_matrix[:2, :2]
    offset = transformation_matrix[:2, 2]

    # Apply the affine transformation
    transformed_image = affine_transform(image, matrix, offset=offset , order=1)
    return transformed_image

def main():
    m0_map = np.load('m0_map.npy')

    contour_mask = process_image(m0_map)

    # Select random points within the contour and create 2D Gaussian distributions
    selected_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)

    plot_gaussians_in_3d(selected_points, gaussians, m0_map.shape)

    transformation_matrices = []
    transformed_points = []

    for i, point in enumerate(selected_points):
        s_x = np.random.uniform(0.8, 1.2) # Scaling factor along x-axis
        s_y = np.random.uniform(0.8, 1.2)  # Scaling factor along y-axis
        theta = np.random.uniform(-np.pi/16, np.pi/16)


        shear_x = 0.02  # Shearing factor along x-axis
        shear_y = 0.02  # Shearing factor along y-axis
        t_x = np.random.uniform(-5, 5)  # Translation along x-axis
        t_y = np.random.uniform(-5, 5)  # Translation along y-axis

        scaling_matrix = np.array([
            [s_x, 0, 0],
            [0, s_y, 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        shearing_matrix = np.array([
            [1, shear_x, 0],
            [shear_y, 1, 0],
            [0, 0, 1]
        ])

        translation_matrix = np.array([
            [1, 0, t_x],
            [0, 1, t_y],
            [0, 0, 1]
        ])

        matrices = {"rotation_matrix": rotation_matrix,
                    "scaling_matrix": scaling_matrix,
                    "shearing_matrix": shearing_matrix,
                    "translation_matrix": translation_matrix}
        print(f"values of affine transformation:s_x, s_y: {s_x, s_y}")
        print(f"values of affine transformation:theta: {np.degrees(theta)}")
        print(f"values of affine transformation:t_x, t_y: {t_x, t_y}\n\n")
        res = apply_manual_affine_transformation(m0_map, matrices)
        plt.imshow(res)
        plt.show()
        transformation_matrices.append(matrices)

    new_image = np.zeros(shape=m0_map.shape)
    new_points = []
    for i in range(m0_map.shape[0]):
        for j in range(m0_map.shape[1]):
            # find distance to all extracted points
            point = np.array([i, j])

            point_transformed_via_all_transformations = [apply_manual_affine_transformation_point(m0_map, point, matrices) for matrices in transformation_matrices]

            distances = np.linalg.norm(selected_points - point, axis=1)
            # transform the distances to a probability distribution
            # distances = np.exp(distances)
            # normalize the distances
            percentages = distances / np.sum(distances)

            # percentages = np.array([gaussian.pdf(point) for gaussian in gaussians])

            # Filter out the None values and the corresponding distances
            filtered_points = []
            filtered_distances = []

            for point, dist in zip(point_transformed_via_all_transformations, percentages):
                if point is not None:
                    filtered_points.append(point)
                    filtered_distances.append(dist)

            # Convert back to numpy arrays if needed
            filtered_points = np.array(filtered_points)
            filtered_distances = np.array(filtered_distances)

            if len(filtered_points) > 0:
                new_sample_location = (filtered_points.T @ filtered_distances).astype(int)
                # new_sample_location = np.exp(np.array(np.log(point_transformed_via_all_transformations).T @ percentages)).astype(int)
                new_image[new_sample_location[0], new_sample_location[1]] = m0_map[i, j]

    # new_image = interpolate_missing_values(new_image)

    plt.imshow(new_image, cmap='viridis')
    plt.title('Distorted M0 Map')
    plt.show()


if __name__ == "__main__":
    main()

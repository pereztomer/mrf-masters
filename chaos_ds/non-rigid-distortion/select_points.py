import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from chaos_ds.find_countor import process_image


def select_random_points_within_contour(map_data, contour_mask, num_points=6):
    """
    Select points from within the contour of the organ in the map,
    ensuring they are spread out over the contour, assign each a standard deviation,
    and create 2D Gaussian distributions.

    Parameters:
    - map_data: 2D numpy array representing the M0 map.
    - contour_mask: Binary mask indicating the region inside the contour.
    - num_points: Number of points to select (default is 6).

    Returns:
    - selected_points: List of tuples (x, y) representing the selected points.
    - stds: List of tuples representing standard deviations (std_x, std_y) for each point.
    - gaussians: List of 2D Gaussian distributions based on the selected points and their stds.
    """
    # Get the indices of the points inside the contour
    inside_contour_indices = np.argwhere(contour_mask)  # Shape: (N, 2)

    # Perform KMeans clustering to partition the contour into num_points clusters
    kmeans = KMeans(n_clusters=num_points, random_state=42)
    kmeans.fit(inside_contour_indices)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # For each cluster center, find the closest actual point in the contour
    selected_points = []
    for center in cluster_centers:
        # Compute distances to all contour points
        distances = np.linalg.norm(inside_contour_indices - center, axis=1)
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        # Add the closest point to the selected points
        selected_points.append(inside_contour_indices[closest_index])

    selected_points = np.array(selected_points)  # Convert to NumPy array

    # Assign a standard deviation to each selected point
    stds = np.random.uniform(1, 2, size=(num_points, 2))  # Example std range [1, 2] for both dimensions

    # Create 2D Gaussian distributions based on the selected points and their stds
    gaussians = []
    for i, (x, y) in enumerate(selected_points):
        mu = [y, x]  # Reversing x and y to align with image coordinates
        cov = np.diag(stds[i] ** 2)  # Covariance matrix for 2D Gaussian
        gaussian = multivariate_normal(mean=mu, cov=cov)
        gaussians.append(gaussian)

    return selected_points, stds, gaussians

def main():
    map_path = r"C:\Users\perez\Desktop\cargoseer\mrf-masters\chaos_ds\m0_map.npy"
    m0_map = np.load(map_path)
    contour_mask = process_image(m0_map)
    control_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)

    import matplotlib.pyplot as plt

    # Plot the contour mask
    plt.imshow(contour_mask, cmap='gray')
    # Plot the selected points
    selected_points_array = np.array(control_points)
    plt.scatter(selected_points_array[:, 1], selected_points_array[:, 0], c='red', marker='x')
    plt.title('Selected Control Points over Contour Mask')
    plt.show()
if __name__ == '__main__':
    main()
import numpy as np

np.random.seed(42)

# Function to create the transformation matrices
def create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height):
    # Scaling matrix
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

    # Shearing matrix
    H = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ])

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Translation to center of image matrix
    T_center = np.array([
        [1, 0, -width / 2],
        [0, 1, -height / 2],
        [0, 0, 1]
    ])

    # Translation back from center of image matrix
    T_center_inv = np.array([
        [1, 0, width / 2],
        [0, 1, height / 2],
        [0, 0, 1]
    ])

    # Translation matrix
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    # Combine all the transformations: Center -> Rotate -> Scale -> Shear -> Translate -> Back to origin
    A = T_center_inv @ R @ T_center @ T @ H @ S
    return A


# Function to compute the Gaussian weight for each control point
def gaussian_weight(q, p, sigma=1):
    dist = np.linalg.norm(q - p)
    return np.exp(-0.5 * (dist ** 2) / sigma ** 2)


# Function to compute the final transformed point
def forward_transform(control_points, transformations, q, sigma=1):
    """
    Perform the non-rigid forward transformation.

    Args:
    control_points (np.ndarray): Control points (k x 2).
    transformations (list of np.ndarray): List of 3x3 transformation matrices.
    q (np.ndarray): The point to transform (2x1).
    sigma (float): Sigma for the Gaussian weight.

    Returns:
    np.ndarray: Transformed point q' (2x1).
    """
    k = control_points.shape[0]

    # Compute Gaussian weights for all control points
    weights = np.array([gaussian_weight(q, control_points[i], sigma) for i in range(k)])

    # Normalize the weights
    weights /= np.sum(weights)

    # Compute the final transformed point as a weighted sum of the transformations
    q_homogeneous = np.array([q[0], q[1], 1])  # Convert to homogeneous coordinates
    transformed_q = np.zeros(3)  # Initialize the result in homogeneous coordinates

    for i in range(k):
        transformed_q += weights[i] * (transformations[i] @ q_homogeneous)

    # Return the transformed point, discarding the homogeneous component
    return transformed_q[:2]


# Example usage
if __name__ == "__main__":
    # Define control points (k = 3)
    control_points = np.array([[1, 1], [5, 5], [9, 1]])

    # Define the width and height of the image
    width = 10
    height = 10

    # Define transformation parameters for each control point
    transformations = []
    for i in range(len(control_points)):
        sx = np.random.uniform(0.8, 1.2)  # Random scaling factor for x
        sy = np.random.uniform(0.8, 1.2)  # Random scaling factor for y
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random rotation angle
        shear_x = np.random.uniform(-0.2, 0.2)  # Random shearing factor for x
        shear_y = np.random.uniform(-0.2, 0.2)  # Random shearing factor for y
        tx = np.random.uniform(-5, 5)  # Random translation in x
        ty = np.random.uniform(-5, 5)  # Random translation in y

        # Create the affine transformation matrix for this control point
        A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
        transformations.append(A)

    # Define a point to be transformed
    q = np.array([4, 3])

    # Perform the forward transformation
    q_prime = forward_transform(control_points, transformations, q, sigma=1)

    # Print the result
    print(f"Original point: {q}")
    print(f"Transformed point: {q_prime}")

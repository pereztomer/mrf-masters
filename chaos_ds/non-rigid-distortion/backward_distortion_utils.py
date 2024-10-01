import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
# Same function as in the forward transform to create transformation matrices
def create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

    H = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ])

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    T_center = np.array([
        [1, 0, -width / 2],
        [0, 1, -height / 2],
        [0, 0, 1]
    ])

    T_center_inv = np.array([
        [1, 0, width / 2],
        [0, 1, height / 2],
        [0, 0, 1]
    ])

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    A = T_center_inv @ R @ T_center @ T @ H @ S
    return A


# Gaussian weight for each control point
def gaussian_weight(q, p, sigma=1):
    dist = np.linalg.norm(q - p)
    return np.exp(-0.5 * (dist ** 2) / sigma ** 2)


# Forward transformation function
def forward_transform(control_points, transformations, q, sigma=1):
    k = control_points.shape[0]
    weights = np.array([gaussian_weight(q, control_points[i], sigma) for i in range(k)])
    weights /= np.sum(weights)

    q_homogeneous = np.array([q[0], q[1], 1])  # Convert to homogeneous coordinates
    transformed_q = np.zeros(3)  # Initialize in homogeneous coordinates

    for i in range(k):
        transformed_q += weights[i] * (transformations[i] @ q_homogeneous)

    return transformed_q[:2]


# Objective function for the backward transformation (minimization)
def objective_function(q, control_points, transformations, q_prime, sigma):
    # Compute the forward transformation for the current guess of q
    q_estimated = forward_transform(control_points, transformations, q, sigma)
    # Compute the squared distance between q_prime and q_estimated
    return np.linalg.norm(q_prime - q_estimated) ** 2


# Backward transformation function (optimization)
def backward_transform(control_points, transformations, q_prime, initial_guess, sigma=1):
    """
    Perform the backward transformation (inverse non-rigid transformation).

    Args:
    control_points (np.ndarray): Control points (k x 2).
    transformations (list of np.ndarray): List of 3x3 transformation matrices.
    q_prime (np.ndarray): The transformed point (2x1).
    initial_guess (np.ndarray): Initial guess for the point q (2x1).
    sigma (float): Sigma for the Gaussian weight.

    Returns:
    np.ndarray: The recovered original point q (2x1).
    """
    # Use scipy's minimize function to find the point q that minimizes the objective function
    result = minimize(objective_function, initial_guess, args=(control_points, transformations, q_prime, sigma),
                      method='BFGS')

    # Return the optimized point q
    return result.x


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

    # Define a transformed point (q_prime) to be recovered
    q_prime = np.array([8.34279749, 1.97156448])

    # Provide an initial guess for the original point q
    initial_guess = np.array([4, 3])  # You can provide a rough estimate

    # Perform the backward transformation
    recovered_q = backward_transform(control_points, transformations, q_prime, initial_guess, sigma=1)

    # Print the result
    print(f"Transformed point (q'): {q_prime}")
    print(f"Recovered original point (q): {recovered_q}")

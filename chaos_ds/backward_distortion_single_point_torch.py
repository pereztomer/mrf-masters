import torch
import numpy as np

# Create transformation matrices using PyTorch
def create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height):
    # Ensure all values are PyTorch tensors
    sx = torch.tensor(sx, dtype=torch.float32)
    sy = torch.tensor(sy, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    shear_x = torch.tensor(shear_x, dtype=torch.float32)
    shear_y = torch.tensor(shear_y, dtype=torch.float32)
    tx = torch.tensor(tx, dtype=torch.float32)
    ty = torch.tensor(ty, dtype=torch.float32)

    S = torch.tensor([
        [sx, torch.tensor(0.0), torch.tensor(0.0)],
        [torch.tensor(0.0), sy, torch.tensor(0.0)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    H = torch.tensor([
        [torch.tensor(1.0), shear_x, torch.tensor(0.0)],
        [shear_y, torch.tensor(1.0), torch.tensor(0.0)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), torch.tensor(0.0)],
        [torch.sin(theta), torch.cos(theta), torch.tensor(0.0)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    T_center = torch.tensor([
        [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(-width / 2)],
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-height / 2)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    T_center_inv = torch.tensor([
        [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(width / 2)],
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(height / 2)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    T = torch.tensor([
        [torch.tensor(1.0), torch.tensor(0.0), tx],
        [torch.tensor(0.0), torch.tensor(1.0), ty],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]
    ], dtype=torch.float32)

    A = T_center_inv @ R @ T_center @ T @ H @ S
    return A


# Gaussian weight for each control point using PyTorch
def gaussian_weight(q, p, sigma=1):
    dist = torch.norm(q - p)
    return torch.exp(-0.5 * (dist ** 2) / sigma ** 2)


# Forward transformation using PyTorch
def forward_transform(control_points, transformations, q, sigma=1):
    k = control_points.shape[0]
    weights = torch.tensor([gaussian_weight(q, control_points[i], sigma) for i in range(k)], dtype=torch.float32)
    weights /= torch.sum(weights)

    q_homogeneous = torch.tensor([q[0], q[1], 1], dtype=torch.float32)  # Convert to homogeneous coordinates
    transformed_q = torch.zeros(3)  # Initialize in homogeneous coordinates

    for i in range(k):
        transformed_q += weights[i] * (transformations[i] @ q_homogeneous)

    return transformed_q[:2]


# Objective function for optimization using PyTorch
def objective_function(q, control_points, transformations, q_prime, sigma):
    # Compute the forward transformation for the current guess of q
    q_estimated = forward_transform(control_points, transformations, q, sigma)
    # Compute the squared distance between q_prime and q_estimated
    return torch.norm(q_prime - q_estimated) ** 2


# Backward transformation function (optimization) using PyTorch
def backward_transform(control_points, transformations, q_prime, initial_guess, sigma=1, lr=1e-2, num_iters=500):
    """
    Perform the backward transformation (inverse non-rigid transformation) using PyTorch's gradient descent.

    Args:
        control_points (torch.Tensor): Control points (k x 2).
        transformations (list of torch.Tensor): List of 3x3 transformation matrices.
        q_prime (torch.Tensor): The transformed point (2x1).
        initial_guess (torch.Tensor): Initial guess for the point q (2x1).
        sigma (float): Sigma for the Gaussian weight.
        lr (float): Learning rate for optimization.
        num_iters (int): Number of optimization iterations.

    Returns:
        torch.Tensor: The recovered original point q (2x1).
    """
    # Initialize the point q as a torch tensor and set requires_grad=True to compute gradients
    q = torch.tensor(initial_guess, dtype=torch.float32, requires_grad=True)

    # Use an optimizer (SGD or Adam)
    optimizer = torch.optim.Adam([q], lr=lr)

    for _ in range(num_iters):
        optimizer.zero_grad()  # Clear gradients
        loss = objective_function(q, control_points, transformations, q_prime, sigma)  # Compute loss

        # Ensure loss is a scalar and requires_grad
        if not loss.requires_grad:
            loss = loss.requires_grad_()

        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update q

    return q.detach().numpy()  # Return the optimized point q


# Example usage
if __name__ == "__main__":
    # Define control points (k = 3)
    control_points = torch.tensor([[1, 1], [5, 5], [9, 1]], dtype=torch.float32)

    # Define the width and height of the image
    width = 10
    height = 10

    # Define transformation parameters for each control point
    transformations = []
    for i in range(len(control_points)):
        sx = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # Random scaling factor for x
        sy = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # Random scaling factor for y
        theta = torch.FloatTensor(1).uniform_(-np.pi / 4, np.pi / 4).item()  # Random rotation angle
        shear_x = torch.FloatTensor(1).uniform_(-0.2, 0.2).item()  # Random shearing factor for x
        shear_y = torch.FloatTensor(1).uniform_(-0.2, 0.2).item()  # Random shearing factor for y
        tx = torch.FloatTensor(1).uniform_(-5, 5).item()  # Random translation in x
        ty = torch.FloatTensor(1).uniform_(-5, 5).item()  # Random translation in y

        # Create the affine transformation matrix for this control point
        A = create_transform_matrices(sx, sy, theta, shear_x, shear_y, tx, ty, width, height)
        transformations.append(A)

    # Define a transformed point (q_prime) to be recovered
    q_prime = torch.tensor([8.34279749, 1.97156448], dtype=torch.float32)

    # Provide an initial guess for the original point q
    initial_guess = torch.tensor([4, 3], dtype=torch.float32)

    # Perform the backward transformation using PyTorch
    recovered_q = backward_transform(control_points, transformations, q_prime, initial_guess, sigma=1)

    # Print the result
    print(f"Transformed point (q'): {q_prime}")
    print(f"Recovered original point (q): {recovered_q}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from chaos_ds.find_countor import process_image
from backward_distortion_utils import create_transform_matrices, gaussian_weight, backward_transform
from tqdm import tqdm

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# PyTorch version of backward transformation for the entire image with vectorization
def backward_transform_2d(control_points, transformations, q_prime_grid, initial_guess_grid, sigma=1, lr=0.1,
                          num_iters=2000):
    height, width, _ = q_prime_grid.shape

    # Flatten the grid of points (height*width x 2)
    q_prime_flat = q_prime_grid.view(-1, 2).to(device)

    # Create q as a leaf tensor with requires_grad=True
    # q = initial_guess_grid.to(device).clone().detach().requires_grad_(True).view(-1, 2)
    q = initial_guess_grid.view(-1, 2).to(device).requires_grad_(True)

    # Use an optimizer (SGD or Adam)
    optimizer = torch.optim.Adam([q], lr=lr)

    for _ in tqdm(range(num_iters)):
        optimizer.zero_grad()  # Clear gradients

        # Compute the objective function for all pixels at once
        loss = objective_function(q, control_points, transformations, q_prime_flat, sigma)

        # Perform backpropagation
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update q

    return q.view(height, width, 2).detach()  # Return the optimized points reshaped


# Vectorized objective function for the entire image
def objective_function(q, control_points, transformations, q_prime_flat, sigma):
    """
    Vectorized objective function that computes the loss over all pixels at once.

    Args:
        q (torch.Tensor): Tensor of size (height*width x 2), current guess for all points.
        control_points (torch.Tensor): Control points (k x 2).
        transformations (list of torch.Tensor): List of 3x3 transformation matrices.
        q_prime_flat (torch.Tensor): Flattened tensor of the transformed points (height*width x 2).
        sigma (float): Sigma for the Gaussian weight.

    Returns:
        torch.Tensor: The loss (summed squared distances between q_prime and q_estimated).
    """
    # Perform forward transformation for all pixels in a vectorized manner
    q_estimated = forward_transform(control_points, transformations, q, sigma)

    # Compute the squared distance between q_prime_flat and q_estimated
    return torch.sum((q_prime_flat - q_estimated) ** 2)


# Vectorized forward transformation using PyTorch
def forward_transform(control_points, transformations, q, sigma=1):
    """
    Apply the forward transformation using Gaussian weighted control points to all pixels at once.

    Args:
        control_points (torch.Tensor): Control points (k x 2).
        transformations (list of torch.Tensor): List of 3x3 transformation matrices.
        q (torch.Tensor): Tensor of size (height*width x 2), current points.
        sigma (float): Sigma for the Gaussian weight.

    Returns:
        torch.Tensor: Transformed points (height*width x 2).
    """
    k = control_points.shape[0]

    # Compute weights for all pixels (height*width x k)
    weights = torch.stack([gaussian_weight(q, control_points[i], sigma) for i in range(k)], dim=-1)
    weights /= weights.sum(dim=-1, keepdim=True)  # Normalize weights (height*width x k)

    # Convert q to homogeneous coordinates (height*width x 3)
    q_homogeneous = torch.cat([q, torch.ones(q.shape[0], 1, device=device)], dim=-1)

    # Apply the transformation to each control point in a vectorized manner
    transformed_q = torch.zeros_like(q_homogeneous)
    for i in range(k):
        transformed_q += weights[:, i:i + 1] * (q_homogeneous @ transformations[i].T)

    return transformed_q[:, :2]  # Return only the x and y coordinates


# Vectorized Gaussian weight calculation using PyTorch
def gaussian_weight(q, p, sigma=1):
    """
    Compute the Gaussian weights for all pixels at once.

    Args:
        q (torch.Tensor): Tensor of size (height*width x 2), current points.
        p (torch.Tensor): Control point (2,).
        sigma (float): Sigma for the Gaussian weight.

    Returns:
        torch.Tensor: Weights for all pixels (height*width).
    """
    dist = torch.norm(q - p, dim=-1)  # Compute distance between q and control point p for all pixels
    return torch.exp(-0.5 * (dist ** 2) / sigma ** 2)


def main():
    map_path = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\chaos_ds\m0_map.npy"
    m0_map = np.load(map_path)

    contour_mask = process_image(m0_map)

    from select_points import select_random_points_within_contour
    # Select random points within the contour and create 2D Gaussian distributions
    control_points, stds, gaussians = select_random_points_within_contour(m0_map, contour_mask)
    height, width = m0_map.shape

    # Convert control points to PyTorch tensors and move to device (GPU/CPU)
    control_points = torch.tensor(control_points, dtype=torch.float32).to(device)

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

    # Convert transformations to PyTorch tensors and move to the device
    transformations = [torch.tensor(A, dtype=torch.float32).to(device) for A in transformations]

    # Create grid of pixel coordinates (q_prime_grid)
    i_coords, j_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    q_prime_grid = torch.stack((i_coords, j_coords), dim=-1).float().to(device)  # Shape: (height, width, 2)

    # Define initial guess for each pixel (using the center of the image)
    initial_guess_grid = torch.tensor([height // 2, width // 2], dtype=torch.float32).expand(height, width, 2).to(
        device)

    # Perform the backward transformation for the entire 2D grid
    recovered_q_grid = backward_transform_2d(control_points, transformations, q_prime_grid, initial_guess_grid,
                                             sigma=25)

    # Reconstruct the new image from the recovered points
    new_image = np.zeros(shape=m0_map.shape)
    recovered_q_grid = recovered_q_grid.int().cpu().numpy()  # Convert back to NumPy for indexing

    for i in range(height):
        for j in range(width):
            q = recovered_q_grid[i, j]
            if 0 <= q[0] < height and 0 <= q[1] < width:
                new_image[i, j] = m0_map[q[0], q[1]]

    plt.imshow(new_image, cmap='viridis')
    plt.title('Distorted M0 Map')
    plt.show()


if __name__ == '__main__':
    main()

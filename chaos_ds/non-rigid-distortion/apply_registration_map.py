import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import torch

def deform_image(image, registration_map):
    """
    Load an image and a registration map, deform the image according to the map, and save the result.

    Args:
        image_path (str): Path to the input image.
        registration_map_path (str): Path to the registration map (.npy file).
        output_path (str): Path to save the deformed image.
    """

    # Initialize the deformed image with the same shape as the original
    deformed_image = np.zeros_like(image)

    # Deform the image according to the registration map
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            target_coords = registration_map[i, j]
            target_x, target_y = int(target_coords[0]), int(target_coords[1])

            # Ensure the target coordinates are within bounds
            if 0 <= target_x < height and 0 <= target_y < width:
                deformed_image[i, j] = image[target_x, target_y]

    return deformed_image


def deform_image_torch(image, registration_map):
    """
    Deform an image according to the registration map using PyTorch on the GPU.

    Args:
        image (torch.Tensor): Input image tensor of shape (H, W) on the GPU.
        registration_map (torch.Tensor): Registration map tensor of shape (H, W, 2) on the GPU.

    Returns:
        torch.Tensor: Deformed image tensor of shape (H, W).
    """
    # Ensure the coordinates in registration_map are integers and within bounds
    height, width = image.shape
    target_x = registration_map[..., 0].clamp(0, height - 1).long()
    target_y = registration_map[..., 1].clamp(0, width - 1).long()

    # Use advanced indexing to sample from the input image based on the registration map
    deformed_image = image[target_x, target_y]

    return deformed_image

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # testing for the registration:
    original_map = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\chaos_ds\m0_map.npy"
    # registration_map_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\distortion_dataset_8\registration_map_22.npy"
    # output_path = r"deformed_image.png"

    folder_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\registration_files"
    # take all files the end with .npy:
    # registration_map_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    registration_map_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if "registration" in f and f.endswith('.npy')]

    registration_map_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # take all files that end with .png:
    # image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if "distorted" in f and f.endswith('.npy')]
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # map files via their number

    for im_path, registration_path in zip(image_files, registration_map_files):
        image = np.load(original_map)
        registration_map = np.load(registration_path)  # Load registration map
        # convert to torch tensors
        image_torch = torch.tensor(image, dtype=torch.float32).to(device)
        registration_map_torch = torch.tensor(registration_map, dtype=torch.long).to(device)
        deformed_image_torch = deform_image_torch(image_torch, registration_map_torch)
        # convert to numpy
        deformed_image_from_torch = deformed_image_torch.cpu().numpy()
        deformed_image = deform_image(image, registration_map)
        # calc diff
        deformed_from_torch = np.load(im_path)
        diff = np.linalg.norm(deformed_image - deformed_from_torch)
        print(f"Mean absolute difference: {diff}")
        diff_2 = np.linalg.norm(deformed_image_from_torch - deformed_from_torch)
        print(f"Mean absolute difference: {diff_2}")
        print("\n\n\n")

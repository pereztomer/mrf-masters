import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
# import pydicom
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial import cKDTree


def process_image(image, plot=False):
    # Apply a threshold to create a binary image
    threshold_value = filters.threshold_otsu(image)
    binary_image = image > threshold_value

    # Remove small objects (noise) and fill holes
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=500)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=500)

    # Find contours (optional for visualization)
    contours = measure.find_contours(cleaned_image, level=0.8)
    if plot:
        # Display the original image with contours (optional)
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='viridis')
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        plt.title('Contours of the Organ')
        plt.axis('off')
        plt.show()

    return cleaned_image

def find_countors_cv2(image):
    # Apply a threshold to create a binary image
    threshold_value = filters.threshold_otsu(image)
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert binary image to uint8 for OpenCV operations
    binary_image = binary_image.astype(np.uint8)

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Fill holes by performing a dilation (expansion) to expand the foreground
    expanded_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Remove small objects by performing an erosion to shrink the expanded regions
    cleaned_image = cv2.erode(expanded_image, kernel, iterations=1)

    # Find contours (optional for visualization)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cleaned_image

def apply_affine_transformation_to_image_torch(image, transformation_matrix):
    """
    Apply an affine transformation to the entire image using PyTorch.

    Parameters:
    - image: 2D numpy array representing the M0 map or any other image.
    - transformation_matrix: 3x3 affine transformation matrix.

    Returns:
    - transformed_image: The transformed image after applying the affine transformation.
    """
    # Convert the image to a torch tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    # Convert the 3x3 transformation matrix to 2x3 for torch (remove the last row)
    transformation_matrix = transformation_matrix[:2, :]

    # Convert the transformation matrix to a torch tensor and add batch dimension
    transformation_matrix = torch.from_numpy(transformation_matrix).float().unsqueeze(0)

    # Generate the grid for sampling
    grid = F.affine_grid(transformation_matrix, image_tensor.size(), align_corners=False)

    # Apply the grid sampling to get the transformed image
    transformed_image_tensor = F.grid_sample(image_tensor, grid, align_corners=False)

    # Remove batch and channel dimensions and convert to numpy array
    transformed_image = transformed_image_tensor.squeeze().numpy()

    return transformed_image


def apply_manual_affine_transformation(image, matrices):
    """
    Manually apply an affine transformation to an image.

    Parameters:
    - image: 2D numpy array representing the image.
    - transformation_matrix: 3x3 affine transformation matrix.

    Returns:
    - transformed_image: The transformed image as a 2D numpy array.
    """
    # Get the dimensions of the original image
    height, width = image.shape

    center_width, center_height = (width // 2, height // 2)
    center_translation_matrix = np.array([[1, 0, -center_width],
                                        [0, 1, -center_height],
                                        [0, 0, 1]])
    center_inverse_translation_matrix = np.array([[1, 0, center_width],
                                                [0, 1, center_height],
                                                [0, 0, 1]])

    new_image = np.zeros(shape=image.shape)
    for y in range(height):
        for x in range(width):
            original_point = np.array([x, y, 1])
            # apply all transformations
            rotation_matrix = matrices["rotation_matrix"]
            transformed_point = center_inverse_translation_matrix @ rotation_matrix @ center_translation_matrix @ original_point
            #iterate throuw a dict of matrices without rotation matirx
            for key in matrices:
                if key != "rotation_matrix":
                    transformation_matrix = matrices[key]
                    transformed_point = transformation_matrix @ transformed_point

            transformed_point = transformed_point[:2].astype(int)
            if 0<=transformed_point[0]<height and 0<=transformed_point[1]<width:
                new_image[transformed_point[1], transformed_point[0]] = image[y, x]

    new_image = interpolate_missing_values(new_image)
    return new_image



def apply_manual_affine_transformation_point(image, point, matrices):
    """
    Manually apply an affine transformation to an image.

    Parameters:
    - image: 2D numpy array representing the image.
    - transformation_matrix: 3x3 affine transformation matrix.

    Returns:
    - transformed_image: The transformed image as a 2D numpy array.
    """
    # Get the dimensions of the original image
    height, width = image.shape

    center_width, center_height = (width // 2, height // 2)
    center_translation_matrix = np.array([[1, 0, -center_width],
                                        [0, 1, -center_height],
                                        [0, 0, 1]])
    center_inverse_translation_matrix = np.array([[1, 0, center_width],
                                                [0, 1, center_height],
                                                [0, 0, 1]])

    x,y = point
    original_point = np.array([x, y, 1])
    # apply all transformations
    rotation_matrix = matrices["rotation_matrix"]
    transformed_point = center_inverse_translation_matrix @ rotation_matrix @ center_translation_matrix @ original_point
    #iterate throuw a dict of matrices without rotation matirx
    for key in matrices:
        if key != "rotation_matrix":
            transformation_matrix = matrices[key]
            transformed_point = transformation_matrix @ transformed_point

    transformed_point = transformed_point[:2].astype(int)
    if 0<=transformed_point[0]<height and 0<=transformed_point[1]<width:
        return transformed_point
    return None


def interpolate_missing_values(new_image):
    """
    Interpolates missing values in the transformed image using the original image.

    Parameters:
    - image: The original 2D numpy array.
    - new_points: The new transformed coordinates.
    - new_image: The transformed image with some missing values.

    Returns:
    - new_image: The transformed image with missing values filled in.
    """

    # Step 1: Find the object contour in the new image
    object_contour_mask = find_countors_cv2(new_image)

    # Step 2: Create a mask for the missing values in the new_image
    missing_mask = (new_image == 0) & (object_contour_mask > 0)

    # Step 3: Get the coordinates of missing points and available points
    missing_indices = np.array(np.where(missing_mask)).T
    available_indices = np.array(np.where(~missing_mask)).T

    # Step 4: Build a KD-Tree for fast nearest-neighbor lookup
    tree = cKDTree(available_indices)

    # Step 5: For each missing point, find the nearest available point
    _, nearest_indices = tree.query(missing_indices, k=1)

    # Step 6: Interpolate missing values
    for i, index in enumerate(missing_indices):
        nearest_point = available_indices[nearest_indices[i]]
        new_image[index[0], index[1]] = new_image[nearest_point[0], nearest_point[1]]

    return new_image


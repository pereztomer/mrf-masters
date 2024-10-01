import os
import pydicom
import matplotlib.pyplot as plt
from glob import glob
import random
import numpy as np
import matplotlib.patches as mpatches
from PIL import Image
random.seed(42)


# Function to extract labels from the segmentation mask based on ranges
def extract_labels(label_image):
    label_values = {
        "Liver": (55, 70),
        "Right kidney": (110, 135),
        "Left kidney": (175, 200),
        "Spleen": (240, 255)
    }

    labels = {}

    # Create a mask for each organ
    combined_mask = np.zeros_like(label_image, dtype=bool)
    for organ, (low, high) in label_values.items():
        mask = (label_image >= low) & (label_image <= high)
        labels[organ] = mask
        combined_mask = combined_mask | mask

    # Background is everything that is not in the combined mask
    labels["Background"] = ~combined_mask

    return labels


# Function to create an overlay image with colored masks
def create_overlay_image(dicom_image, labels):
    overlay = np.zeros((*dicom_image.shape, 3), dtype=np.uint8)
    color_map = {
        "Liver": (255, 0, 0),  # Red
        "Right kidney": (0, 255, 0),  # Green
        "Left kidney": (0, 0, 255),  # Blue
        "Spleen": (255, 255, 0),  # Yellow
        "Background": (0, 0, 0), # Black
        "Fat": (255, 255, 255)  # White
    }

    for organ, mask in labels.items():
        overlay[mask] = color_map[organ]

    # plot the overlay on the dicom image

    # Plot the overlay on the dicom image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(dicom_image, cmap='gray')
    ax.imshow(overlay, alpha=0.5)
    ax.set_title('Overlay Image on top of mri image')
    ax.axis('off')

    # Create legend
    patches = [mpatches.Patch(color=np.array(color_map[organ]) / 255, label=organ) for organ in color_map]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.show()
    return overlay, color_map


# Function to load and display image with overlay and legend
def load_and_display_image_with_label(dicom_path, label_path):
    # Load DICOM image
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_image = dicom_data.pixel_array

    # Load label image
    label_image = np.array(Image.open(label_path))

    # Extract labels
    labels = extract_labels(label_image)

    # Create overlay image
    overlay, color_map = create_overlay_image(dicom_image, labels)

    # Display images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(dicom_image, cmap='gray')
    ax[0].set_title('DICOM Image')
    ax[0].axis('off')

    ax[1].imshow(label_image, cmap='gray')
    ax[1].set_title('Label Image')
    ax[1].axis('off')

    ax[2].imshow(dicom_image, cmap='gray')
    ax[2].imshow(overlay, alpha=0.5)
    ax[2].set_title('Overlay Image')
    ax[2].axis('off')

    # Create legend
    patches = [mpatches.Patch(color=np.array(color_map[organ]) / 255, label=organ) for organ in color_map]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.show()

def load_one_image_per_class(base_path):
    # List all class directories
    class_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for class_dir in class_dirs:
        class_path = os.path.join(base_path, class_dir, 'T1DUAL')

        samples_path = os.path.join(class_path, 'DICOM_anon', 'InPhase')
        labels_files = os.path.join(class_path, 'Ground')
        # List all files in the T1DUAL directory
        samples_path = glob(os.path.join(samples_path, '*.dcm'))
        labels_files = glob(os.path.join(labels_files, '*.png'))

        # select random number from 0 to len(samples_path)
        random_index = random.randint(0, len(samples_path)-1)
        # Load the first DICOM file found
        single_dicom_file_path = samples_path[random_index]
        single_labels_file_path = labels_files[random_index]

        load_and_display_image_with_label(single_dicom_file_path, single_labels_file_path)



def main():
    # Replace with the actual path to your dataset
    base_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR"
    load_one_image_per_class(base_path)


if __name__ == '__main__':
    main()


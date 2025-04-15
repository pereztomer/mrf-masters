import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')


def read_dicom(file_path):
    """Read a DICOM file"""
    return pydicom.dcmread(file_path)


def save_images(dicom_path, save_dir):
    """
    Read DICOM file and save slices and k-space images

    Parameters:
    dicom_path: Path to the DICOM file
    save_dir: Directory to save images
    """
    # Read the DICOM file
    ds = read_dicom(dicom_path)
    img = ds.pixel_array

    # Create main save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"Main directory: {save_dir}")

    # Process and save slices
    for i in range(0, len(img)):
        # Create slice-specific folder
        slice_folder = os.path.join(save_dir, f"slice_{i}")
        os.makedirs(slice_folder, exist_ok=True)

        # Calculate k-space
        kspace = np.fft.fftshift(np.fft.fft2(img[i]))
        k_display = np.log(np.abs(kspace) + 1)

        # Save original slice
        plt.figure(figsize=(8, 8))
        plt.imshow(img[i], cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        slice_filename = os.path.join(slice_folder, "slice.png")
        plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        # Save k-space
        plt.figure(figsize=(8, 8))
        plt.imshow(k_display, cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        kspace_filename = os.path.join(slice_folder, "kspace.png")
        plt.savefig(kspace_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        print(f"Saved slice {i} in folder: {slice_folder}")


if __name__ == "__main__":
    # File paths
    # dicom_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\general sequence scan of phantoms\Dicoms_2_phantoms\SER00010\IMG00001.dcm"
    # save_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\general sequence scan of phantoms\Dicoms_2_phantoms\SER00010\IMG00001"
    dicom_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\general sequence scan of phantoms\Dicoms_2_phantoms\SER00002\IMG00001.dcm"
    save_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\general sequence scan of phantoms\Dicoms_2_phantoms\SER00002\IMG00001"
    # Save the images
    save_images(dicom_path, save_dir)
    print("Processing complete.")
import matplotlib.pyplot as plt
import pydicom
from read_dicom_from_dirs import extract_labels, create_overlay_image
from PIL import Image

from find_countor import process_image
import numpy as np
import MRzeroCore as mr0
from skimage import measure
import json

def main():
    obj_p = mr0.VoxelGridPhantom.brainweb(r"C:\Users\perez\Desktop\phantom\subject05.npz")

    fig, axes = plt.subplots(2, 7, figsize=(16, 8))
    slice = 64
    # images = [obj_p.B0[:,slice], obj_p.B1[0][:,slice], obj_p.D[:,slice], obj_p.PD[:,slice], obj_p.T1[:,slice], obj_p.T2[:,slice], obj_p.T2dash[:,slice]]
    images = {"B0": obj_p.B0[:, :, slice], "B1": obj_p.B1[0][:, :, slice], "D": obj_p.D[:, :, slice],
              "PD": obj_p.PD[:, :, slice],
              "T1": obj_p.T1[:, :, slice], "T2": obj_p.T2[:, :, slice], "T2_dash": obj_p.T2dash[:, :, slice]}

    W, H = 256, 256
    import cv2
    for map_type, im in images.items():
        im = im.numpy()
        im_resized = cv2.resize(im.real, (W, H)) + 1j * cv2.resize(im.imag, (W, H))
        images[map_type] = im_resized

    brain_mask = images['PD'] > 0

    for i, (label, img) in enumerate(images.items()):
        img = np.abs(img)  # Handle complex data

        # Image
        ax_img = axes[0, i]
        im = ax_img.imshow(img, cmap='gray')
        ax_img.set_title(label)
        ax_img.axis('off')
        plt.colorbar(im, ax=ax_img)

        # Histogram
        ax_hist = axes[1, i]
        ax_hist.hist(img[brain_mask].flatten(), bins=50, color='blue', alpha=0.7)
        ax_hist.set_title(f'{label} Histogram')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    main()

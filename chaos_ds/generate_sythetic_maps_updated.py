import numpy as np
import matplotlib.pyplot as plt
import pydicom
from read_dicom_from_dirs import extract_labels, create_overlay_image
from PIL import Image

from find_countor import process_image
# Function to generate T1, T2, and M0 maps
import numpy as np
import MRzeroCore as mr0

"""
####################################
This script generates synthetic T1, T2, and M0 maps from a set of labeled masks and an dicom image
####################################
"""



def generate_maps(labels, default_values, unknown_image, background_mask):
    shape = next(iter(labels.values())).shape  # Get the shape from one of the masks
    t1_map = np.zeros(shape, dtype=np.float32)
    t2_map = np.zeros(shape, dtype=np.float32)
    m0_map = np.zeros(shape, dtype=np.float32)

    # Typical ranges for T1, T2 (you may adjust these based on your specific needs)
    t1_range = (300, 2000)  # Example: from 300 ms to 2000 ms
    t2_range = (20, 300)  # Example: from 20 ms to 300 ms
    m0_range = (0.5, 1.5)  # Example: from 0.5 to 1.5 relative proton density

    # Normalize the unknown image to the typical ranges
    def stretch_image_to_range(image, target_range):
        min_val, max_val = np.min(image), np.max(image)
        stretched = target_range[0] + (image - min_val) * (target_range[1] - target_range[0]) / (max_val - min_val)
        return stretched

    stretched_t1 = stretch_image_to_range(unknown_image, t1_range)
    stretched_t2 = stretch_image_to_range(unknown_image, t2_range)
    stretched_m0 = stretch_image_to_range(unknown_image, m0_range)

    for organ, mask in labels.items():
        if organ == "Background":
            continue
        t1_map[mask] = default_values[organ]['T1']
        t2_map[mask] = default_values[organ]['T2']
        m0_map[mask] = default_values[organ]['M0']

    # Fill in the areas with no mask using the stretched unknown image
    no_mask = np.ones(shape, dtype=bool)
    for mask in labels.values():
        no_mask &= ~mask  # Combine all masks to find unmasked areas

    no_mask &= background_mask

    t1_map[no_mask] = stretched_t1[no_mask]
    t2_map[no_mask] = stretched_t2[no_mask]
    m0_map[no_mask] = stretched_m0[no_mask]

    return t1_map, t2_map, m0_map


def main():
    import MRzeroCore as mr0


    obj_p = mr0.VoxelGridPhantom.brainweb(r"C:\Users\perez\Desktop\phantom\subject05.npz")

    fig, axes = plt.subplots(2, 7, figsize=(16, 8))
    slice = 64
    # images = [obj_p.B0[:,slice], obj_p.B1[0][:,slice], obj_p.D[:,slice], obj_p.PD[:,slice], obj_p.T1[:,slice], obj_p.T2[:,slice], obj_p.T2dash[:,slice]]
    images = [obj_p.B0[:,:,slice], obj_p.B1[0][:,:,slice], obj_p.D[:,:,slice], obj_p.PD[:,:,slice], obj_p.T1[:,:,slice], obj_p.T2[:,:,slice], obj_p.T2dash[:,:,slice]]
    # images = [obj_p.B0[slice], obj_p.B1[0][slice], obj_p.D[slice], obj_p.PD[slice], obj_p.T1[slice], obj_p.T2[slice], obj_p.T2dash[slice]]
    labels = ['B0', 'B1', 'D', 'PD', 'T1', 'T2', 'T2dash']
    brain_mask = (obj_p.PD[:,:,slice] > 0).numpy()

    for i, (img, label) in enumerate(zip(images, labels)):
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

    single_dicom_inphase_file_path = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00016.dcm"
    single_labels_file_path = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\Ground\IMG-0004-00046.png"

    # Load DICOM image
    dicom_inphase_data = pydicom.dcmread(single_dicom_inphase_file_path)
    dicom_inphase_image = dicom_inphase_data.pixel_array
    # min-max normalization for dicom_inphase_image

    abdomen_mask = process_image(dicom_inphase_image)

    # Load label image
    label_image = np.array(Image.open(single_labels_file_path))

    # Extract labels
    labels = extract_labels(label_image)

    # add to labels a new mask for fat (values above 500 in the dicom_inphase_image)
    labels["Fat"] = dicom_inphase_image > 500
    # remove key backgorund from labels
    labels.pop("Background", None)
    labels['abdomen_mask'] = abdomen_mask

    n_masks = len(labels)
    fig, axes = plt.subplots(1, n_masks, figsize=(4 * n_masks, 4))

    for ax, (name, mask) in zip(axes, labels.items()):
        ax.imshow(mask, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    dicom_inphase_image = (dicom_inphase_image - np.min(dicom_inphase_image)) / (
                np.max(dicom_inphase_image) - np.min(dicom_inphase_image))
    plt.imshow(dicom_inphase_image, cmap='gray')
    plt.title("Dicom Inphase Image")
    plt.show()
    # overlay, color_map = create_overlay_image(dicom_inphase_image, labels)

    # Default values for T1, T2, and M0
    default_values = {
        "Liver": {"T1": 500 / 500, "T2": 43 / 56, "M0": 1},
        "Right kidney": {"T1": 650 / 500, "T2": 58 / 56, "M0": 0.9},
        "Left kidney": {"T1": 650 / 500, "T2": 58 / 56, "M0": 0.9},
        "Spleen": {"T1": 200 / 500, "T2": 61 / 56, "M0": 1},
    }

    from skimage import exposure
    t1_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask], obj_p.T1[:,:,slice].numpy()[brain_mask])
    t1_abdomen = np.zeros_like(dicom_inphase_image)
    t1_abdomen[abdomen_mask] = t1_matched_vals

    t2_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask], obj_p.T2[:,:,slice].numpy()[brain_mask])
    t2_abdomen = np.zeros_like(dicom_inphase_image)
    t2_abdomen[abdomen_mask] = t2_matched_vals

    pd_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask], obj_p.PD[:,:,slice].numpy()[brain_mask])
    pd_abdomen = np.zeros_like(dicom_inphase_image)
    pd_abdomen[abdomen_mask] = pd_matched_vals


    # Display the maps with histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    maps = [t1_abdomen, t2_abdomen, pd_abdomen]
    labels = ['T1 Map', 'T2 Map', 'M0 Map']

    for i, (map_data, label) in enumerate(zip(maps, labels)):
        # Image
        ax_img = axes[0, i]
        im = ax_img.imshow(map_data, cmap='hot', interpolation='nearest')
        ax_img.set_title(label)
        ax_img.axis('off')
        plt.colorbar(im, ax=ax_img)

        # Histogram
        ax_hist = axes[1, i]
        ax_hist.hist(np.abs(map_data).flatten(), bins=50, color='blue', alpha=0.7)
        ax_hist.set_title(f'{label} Histogram')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    # save all maps

    np.savez(
        r"C:\Users\perez\Desktop\phantom\abdominal_phantom\maps.npz",
        t1=t1_abdomen,
        t2=t2_abdomen,
        m0=pd_abdomen
    )

# Example usage
if __name__ == "__main__":
    main()

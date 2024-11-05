import numpy as np
import matplotlib.pyplot as plt
import pydicom
from read_dicom_from_dirs import extract_labels, create_overlay_image
from PIL import Image

from find_countor import process_image
# Function to generate T1, T2, and M0 maps
import numpy as np

"""
####################################
This script generates synthetic T1, T2, and M0 maps from a set of labeled masks and an unknown image!!!!!!!!!!!!!!!!
####################################
"""
def generate_maps(labels, default_values, unknown_image, background_mask):
    shape = next(iter(labels.values())).shape  # Get the shape from one of the masks
    t1_map = np.zeros(shape, dtype=np.float32)
    t2_map = np.zeros(shape, dtype=np.float32)
    m0_map = np.zeros(shape, dtype=np.float32)

    # Typical ranges for T1, T2 (you may adjust these based on your specific needs)
    t1_range = (300, 2000)  # Example: from 300 ms to 2000 ms
    t2_range = (20, 300)    # Example: from 20 ms to 300 ms
    m0_range = (0.5, 1.5)   # Example: from 0.5 to 1.5 relative proton density

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
    single_dicom_inphase_file_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00016.dcm"
    single_dicom_outphase_file_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\OutPhase\IMG-0004-00016.dcm"
    single_labels_file_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\Ground\IMG-0004-00046.png"


    # Load DICOM image
    dicom_inphase_data = pydicom.dcmread(single_dicom_inphase_file_path)
    dicom_inphase_image = dicom_inphase_data.pixel_array
    # min-max normalization for dicom_inphase_image

    background_mask = process_image(dicom_inphase_image)

    # Load label image
    label_image = np.array(Image.open(single_labels_file_path))

    # Extract labels
    labels = extract_labels(label_image)

    # add to labels a new mask for fat (values above 500 in the dicom_inphase_image)
    labels["Fat"] = dicom_inphase_image > 500
    # remove key backgorund from labels
    labels.pop("Background", None)
    # Create overlay image

    dicom_inphase_image = (dicom_inphase_image - np.min(dicom_inphase_image)) / (np.max(dicom_inphase_image) - np.min(dicom_inphase_image))
    plt.imshow(dicom_inphase_image, cmap='gray')
    plt.show()
    # overlay, color_map = create_overlay_image(dicom_inphase_image, labels)

    # Default values for T1, T2, and M0
    default_values = {
        "Liver": {"T1": 500, "T2": 43, "M0": 1},
        "Right kidney": {"T1": 650, "T2": 58, "M0": 0.9},
        "Left kidney": {"T1": 650, "T2": 58, "M0": 0.9},
        "Spleen": {"T1": 200, "T2": 61, "M0": 1},
        "Fat": {"T1": 260, "T2": 85, "M0": 0.5}
    }

    t1_map, t2_map, m0_map = generate_maps(labels, default_values,dicom_inphase_image, background_mask)

    # Display the maps
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(t1_map, cmap='hot', interpolation='nearest')
    ax[0].set_title('T1 Map')
    ax[0].axis('off')

    ax[1].imshow(t2_map, cmap='hot', interpolation='nearest')
    ax[1].set_title('T2 Map')
    ax[1].axis('off')

    ax[2].imshow(m0_map, cmap='hot', interpolation='nearest')
    ax[2].set_title('M0 Map')
    ax[2].axis('off')

    plt.show()
    # save all maps
    np.save("t1_map.npy", t1_map)
    np.save("t2_map.npy", t2_map)
    np.save("m0_map.npy", m0_map)



# Example usage
if __name__ == "__main__":
    main()
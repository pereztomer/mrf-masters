import os
import numpy as np
import pydicom
from PIL import Image
from read_dicom_from_dirs import extract_labels
from find_countor import process_image
import matplotlib.pyplot as plt

def generate_maps(labels, default_values, unknown_image, background_mask):
    shape = next(iter(labels.values())).shape
    t1_map = np.zeros(shape, dtype=np.float32)
    t2_map = np.zeros(shape, dtype=np.float32)
    m0_map = np.zeros(shape, dtype=np.float32)

    t1_range = (300, 2000)
    t2_range = (20, 300)
    m0_range = (0.5, 1.5)

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

    no_mask = np.ones(shape, dtype=bool)
    for mask in labels.values():
        no_mask &= ~mask

    no_mask &= background_mask

    t1_map[no_mask] = stretched_t1[no_mask]
    t2_map[no_mask] = stretched_t2[no_mask]
    m0_map[no_mask] = stretched_m0[no_mask]

    return t1_map, t2_map, m0_map

def process_and_save_maps(dicom_inphase_image, labels, default_values, background_mask, output_dir, filename_base):
    background_map = labels["Background"]
    labels.pop("Background", None)
    t1_map, t2_map, m0_map = generate_maps(labels, default_values, dicom_inphase_image, background_mask)
    labels["Background"] = background_map
    np_files_path = os.path.join(output_dir, "numpy_files")
    os.makedirs(np_files_path, exist_ok=True)
    np.save(os.path.join(np_files_path, f"{filename_base}_t1_map.npy"), t1_map)
    np.save(os.path.join(np_files_path, f"{filename_base}_t2_map.npy"), t2_map)
    np.save(os.path.join(np_files_path, f"{filename_base}_m0_map.npy"), m0_map)

    # save t1,t2 and m0 mask as an image
    image_maps_path = os.path.join(output_dir, "image_maps")
    os.makedirs(image_maps_path, exist_ok=True)
    t1_map_as_image = (t1_map - np.min(t1_map)) / (np.max(t1_map) - np.min(t1_map))
    t2_map_as_image = (t2_map - np.min(t2_map)) / (np.max(t2_map) - np.min(t2_map))
    m0_map_as_image = (m0_map - np.min(m0_map)) / (np.max(m0_map) - np.min(m0_map))
    # save as image
    plt.imsave(os.path.join(image_maps_path, f"{filename_base}_t1_map.png"), t1_map_as_image, cmap='magma')
    plt.imsave(os.path.join(image_maps_path, f"{filename_base}_t2_map.png"), t2_map_as_image, cmap='viridis')
    plt.imsave(os.path.join(image_maps_path, f"{filename_base}_m0_map.png"), m0_map_as_image, cmap='magma')

    # Save the labels (ground truth)
    original_segmentation_path = os.path.join(output_dir, "original_segmentation")
    os.makedirs(original_segmentation_path, exist_ok=True)
    save_labels(labels, original_segmentation_path, filename_base)

def save_labels(labels, output_dir, filename_base):
    """Saves each organ mask as a separate file."""
    for organ, mask in labels.items():
        label_output_path = os.path.join(output_dir, f"{filename_base}_{organ}_mask.png")
        mask = mask.astype(np.uint8) * 255
        Image.fromarray(mask).save(label_output_path)
        # np.save(label_output_path, mask)

def traverse_and_process_dataset(dataset_dir, output_dir):
    default_values = {
        "Liver": {"T1": 500, "T2": 43, "M0": 1},
        "Right kidney": {"T1": 650, "T2": 58, "M0": 0.9},
        "Left kidney": {"T1": 650, "T2": 58, "M0": 0.9},
        "Spleen": {"T1": 200, "T2": 61, "M0": 1},
        "Fat": {"T1": 260, "T2": 85, "M0": 0.5}
    }

    # Train_Sets\MR
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "MR" in file_path and file.endswith(".dcm"):
                dicom_file_path = os.path.join(root, file)
                label_root = root.replace("InPhase", "").replace("OutPhase", "")
                label_image_path = os.path.join(label_root,file)
                label_image_path = label_image_path.replace("DICOM_anon", "Ground").replace(".dcm", ".png")
                if os.path.exists(label_image_path):
                    dicom_inphase_data = pydicom.dcmread(dicom_file_path)
                    dicom_inphase_image = dicom_inphase_data.pixel_array
                    dicom_inphase_image = (dicom_inphase_image - np.min(dicom_inphase_image)) / (np.max(dicom_inphase_image) - np.min(dicom_inphase_image))

                    background_mask = process_image(dicom_inphase_image)

                    label_image = np.array(Image.open(label_image_path))
                    labels = extract_labels(label_image)

                    labels["Fat"] = dicom_inphase_image > 500
                    # labels.pop("Background", None)

                    # Create mirrored output directory
                    relative_path = os.path.relpath(root, dataset_dir)
                    output_subdir = os.path.join(output_dir, relative_path)
                    filename_base = os.path.splitext(file)[0]
                    output_subdir = os.path.join(output_subdir,filename_base)

                    process_and_save_maps(dicom_inphase_image, labels, default_values, background_mask, output_subdir, filename_base)

def main(dataset_dir, output_dir):
    traverse_and_process_dataset(dataset_dir, output_dir)


if __name__ == "__main__":
    # Replace these paths with the actual dataset path and destination path
    dataset_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset - Copy"
    output_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos processed"

    main(dataset_dir, output_dir)
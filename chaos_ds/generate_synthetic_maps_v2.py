import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.io import savemat
from opensimplex import OpenSimplex
import pydicom
import matplotlib.pyplot as plt
from skimage import measure
from read_dicom_from_dirs import extract_labels
from PIL import Image
from scipy.ndimage import label as scipy_label
from scipy.ndimage import label as scipy_label, binary_closing
import os
import MRzeroCore as mr0
from cv2 import morphologyEx, MORPH_CLOSE, getStructuringElement, MORPH_ELLIPSE
from skimage import io, filters, measure, morphology


def random_multi_peak_dist(perlin_field, n_peaks=2):
    peaks = np.sort(np.random.rand(n_peaks))
    weights = np.random.dirichlet(np.ones(n_peaks))
    result = np.zeros_like(perlin_field)
    for peak, weight in zip(peaks, weights):
        sigma = np.random.uniform(0.05, 0.15)
        result += weight * np.exp(-((perlin_field - peak) ** 2) / (2 * sigma ** 2))
    return result / result.max()


def create_perlin_maps(dicom_path, labels_png_path, mat_output_path, labels_npy_output_path, seed=42,
                       plot_output_path=None, plot=False):
    # Load DICOM
    dicom = pydicom.dcmread(dicom_path).pixel_array
    dicom = (dicom - dicom.min()) / (dicom.max() - dicom.min())

    # Extract abdomen_mask
    threshold_value = filters.threshold_otsu(dicom)
    binary_image = dicom > threshold_value
    abdomen_mask = morphology.remove_small_holes(binary_image, area_threshold=2000)

    label_image = np.array(Image.open(labels_png_path))
    labels = extract_labels(label_image)
    labels.pop("Background", None)
    labels['abdomen_mask'] = abdomen_mask

    non_black_pct = (label_image != 0).sum() / label_image.size * 100
    if non_black_pct < 4:
        print("Mask too small, skipping")
        return
    os.makedirs(os.path.dirname(mat_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_npy_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    print(f"Non-black: {non_black_pct:.2f}%")

    # Create regions based on intensity levels
    dicom_norm = (dicom * 255).astype(np.uint8)
    # levels_percentiles = np.percentile(dicom_norm[abdomen_mask], [18,36,60,75,90])
    levels_percentiles = np.percentile(dicom_norm[abdomen_mask], [36,75,90])
    levels = []
    # Process each intensity level
    level_ranges = [(levels_percentiles[i - 1] if i > 0 else 0, levels_percentiles[i]) for i in range(len(levels_percentiles))] + [(levels_percentiles[-1], 255)]
    for level_idx, (level_min, level_max) in enumerate(level_ranges):
        # Create binary mask for this intensity range
        level_mask = (dicom_norm >= level_min) & (dicom_norm < level_max)

        kernel = getStructuringElement(MORPH_ELLIPSE, (7, 7))
        level_mask = morphologyEx(level_mask.astype(np.uint8), MORPH_CLOSE, kernel)
        level_mask = level_mask.astype(bool)

        # Apply to entire level
        level_mask = level_mask & abdomen_mask
        if not level_mask.any():
            continue
        levels.append(level_mask)

    t1_map = np.zeros_like(dicom)
    t2_map = np.zeros_like(dicom)
    pd_map = np.zeros_like(dicom)

    t1_range, t2_range, pd_range = (0, 4.1), (0, 1.6), (0.4, 1.0)
    noise_gen = OpenSimplex(seed=seed)

    # Generate Perlin noise once
    h, w = dicom.shape
    perlin = np.array([[noise_gen.noise2(x * 0.05, y * 0.05) for x in range(w)] for y in range(h)])
    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min() + 1e-8)

    # Process each intensity level
    for level_mask in levels:
        # Random center value for this level
        t1_val = t1_range[0] + np.random.rand() * (t1_range[1] - t1_range[0])
        t2_val = t2_range[0] + np.random.rand() * (t2_range[1] - t2_range[0])
        pd_val = pd_range[0] + np.random.rand() * (pd_range[1] - pd_range[0])

        # Add smooth gradient with Perlin noise
        level_perlin = perlin[level_mask]
        t1_map[level_mask] = t1_val + (level_perlin - 0.5) * (t1_range[1] - t1_range[0]) * 0.05
        t2_map[level_mask] = t2_val + (level_perlin - 0.5) * (t2_range[1] - t2_range[0]) * 0.05
        pd_map[level_mask] = pd_val + (level_perlin - 0.5) * (pd_range[1] - pd_range[0]) * 0.05



    for region_name in labels.keys():
        if region_name == "abdomen_mask":
            continue

        region = labels[region_name]

        t1_val = t1_range[0] + np.random.rand() * (t1_range[1] - t1_range[0])
        t2_val = t2_range[0] + np.random.rand() * (t2_range[1] - t2_range[0])
        pd_val = pd_range[0] + np.random.rand() * (pd_range[1] - pd_range[0])

        # Add smooth gradient with Perlin noise
        level_perlin = perlin[region]
        t1_map[region] = t1_val + (level_perlin - 0.5) * (t1_range[1] - t1_range[0]) * 0.05
        t2_map[region] = t2_val + (level_perlin - 0.5) * (t2_range[1] - t2_range[0]) * 0.05
        pd_map[region] = pd_val + (level_perlin - 0.5) * (pd_range[1] - pd_range[0]) * 0.05


    t1_map[~abdomen_mask] = 0
    t2_map[~abdomen_mask] = 0
    pd_map[~abdomen_mask] = 0

    # Crop
    props = measure.regionprops(abdomen_mask.astype(int))
    if props:
        min_row, min_col, max_row, max_col = props[0].bbox
        padding = 0
        min_row, min_col = max(0, min_row - padding), max(0, min_col - padding)
        max_row, max_col = min(h, max_row + padding), min(w, max_col + padding)

        # Make square
        crop_h = max_row - min_row
        crop_w = max_col - min_col
        max_dim = max(crop_h, crop_w)
        pad_h = (max_dim - crop_h) // 2
        pad_w = (max_dim - crop_w) // 2

        t1_map = t1_map[min_row:max_row, min_col:max_col]
        t2_map = t2_map[min_row:max_row, min_col:max_col]
        pd_map = pd_map[min_row:max_row, min_col:max_col]

        t1_map = np.pad(t1_map, ((pad_h, max_dim - crop_h - pad_h), (pad_w, max_dim - crop_w - pad_w)), mode='constant',
                        constant_values=0)
        t2_map = np.pad(t2_map, ((pad_h, max_dim - crop_h - pad_h), (pad_w, max_dim - crop_w - pad_w)), mode='constant',
                        constant_values=0)
        pd_map = np.pad(pd_map, ((pad_h, max_dim - crop_h - pad_h), (pad_w, max_dim - crop_w - pad_w)), mode='constant',
                        constant_values=0)

        for key in labels.keys():
            labels[key] = labels[key][min_row:max_row, min_col:max_col]
            labels[key] = np.pad(labels[key], ((pad_h, max_dim - crop_h - pad_h), (pad_w, max_dim - crop_w - pad_w)),
                                 mode='constant', constant_values=0)

        for idx in range(len(levels)):
            temp_level = levels[idx][min_row:max_row, min_col:max_col]
            levels[idx] = np.pad(temp_level, ((pad_h, max_dim - crop_h - pad_h), (pad_w, max_dim - crop_w - pad_w)),
                                 mode='constant', constant_values=0)

    W, H = t1_map.shape
    obj_p = mr0.VoxelGridPhantom.brainweb(r"C:\Users\perez\Desktop\phantom\subject05.npz")
    slice_num = 64
    B0_map = obj_p.B0[:, :, slice_num].numpy()
    B0_map = cv2.resize(B0_map.real, (W, H)) + 1j * cv2.resize(B0_map.imag, (W, H))

    B1_map = obj_p.B1[0][:, :, slice_num].numpy()
    B1_map = cv2.resize(B1_map.real, (W, H)) + 1j * cv2.resize(B1_map.imag, (W, H))

    if plot_output_path:
        n_level_rows = (len(levels) + 4) // 5  # ceil division

        fig, axes = plt.subplots(2 + n_level_rows, 5, figsize=(16, 4 * (2 + n_level_rows)))

        # Row 0: DICOM, abdomen_mask, T1, T2, PD
        axes[0, 0].imshow(dicom, cmap='gray')
        axes[0, 0].set_title('DICOM')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(labels['abdomen_mask'], cmap='gray')
        axes[0, 1].set_title('Abdomen Mask')
        axes[0, 1].axis('off')

        vmin_t1, vmax_t1 = t1_range
        vmin_t2, vmax_t2 = t2_range
        vmin_pd, vmax_pd = pd_range

        for i, (m, name, vmin, vmax) in enumerate(
                [(t1_map, 'T1', vmin_t1, vmax_t1), (t2_map, 'T2', vmin_t2, vmax_t2), (pd_map, 'PD', vmin_pd, vmax_pd)]):
            im = axes[0, i + 2].imshow(m, cmap='hot', vmin=vmin, vmax=vmax)
            axes[0, i + 2].set_title(name)
            axes[0, i + 2].axis('off')
            plt.colorbar(im, ax=axes[0, i + 2])

        # Row 1: Histograms + labels
        for i, (m, name) in enumerate([(t1_map, 'T1'), (t2_map, 'T2'), (pd_map, 'PD')]):
            axes[1, i + 1].hist(m[m > 0].flatten(), bins=50, alpha=0.7)
            axes[1, i + 1].set_title(f'{name} Hist')

        axes[1, 0].imshow(np.array(Image.open(labels_png_path)), cmap='gray')
        axes[1, 0].set_title('Ground Truth Labels')
        axes[1, 0].axis('off')

        # Rows 2+: Intensity levels (5 per row)
        for idx, level_mask in enumerate(levels):
            row = 2 + (idx // 5)
            col = idx % 5
            axes[row, col].imshow(level_mask.astype(np.uint8)*255, cmap='gray')
            axes[row, col].set_title(f'Level {idx}\n[{int(level_min)}-{int(level_max)}]')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=100)
        if plot:
            plt.show()
    
    
    stacked = np.stack([pd_map,
                        t1_map,
                        t2_map,
                        B0_map,  # index 3
                        B1_map], axis=-1)
    savemat(mat_output_path, {'cropped_brain': stacked})
    np.save(labels_npy_output_path, labels)


def process_dataset(ground_dir, inphase_dir, output_dir, seed=42, plot=False):
    ground_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])

    for ground_file in ground_files:
        # Match DICOM by name (remove suffix if needed)
        base_name = ground_file.replace('.png', '')
        dicom_file = None
        for f in os.listdir(inphase_dir):
            if base_name in f and f.endswith('.dcm'):
                dicom_file = f
                break

        if dicom_file is None:
            print(f"Skipping {ground_file} - no matching DICOM")
            continue

        dicom_path = os.path.join(inphase_dir, dicom_file)
        labels_png_path = os.path.join(ground_dir, ground_file)
        mat_output = os.path.join(output_dir, base_name, f"{base_name}.mat")
        npy_output = os.path.join(output_dir, base_name, f"{base_name}_labels.npy")
        plot_output = os.path.join(output_dir, base_name, f"{base_name}_plot.png")

        print(f"Processing {base_name}...")
        create_perlin_maps(dicom_path, labels_png_path, mat_output, npy_output, seed=seed, plot_output_path=plot_output,
                           plot=plot)




def main():
    # process_dataset(
    #     r"C:\Users\perez\Desktop\data_from_local\Ground",
    #     r"C:\Users\perez\Desktop\data_from_local\InPhase",
    #     r"C:\Users\perez\Desktop\data_from_local\Output",
    #     seed=42,
    #     plot=False
    # )
    ground_dir = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\Ground"
    inphase_dir = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase"
    output_dir = r"C:\Users\perez\Desktop\abdomen_phantoms_2"
    process_dataset(
        ground_dir,
        inphase_dir,
        output_dir,
        seed=42,
        plot=True
    )


if __name__ == '__main__':
    main()
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.io import savemat
from opensimplex import OpenSimplex
import pydicom
import matplotlib.pyplot as plt
from skimage import measure
from find_countor import process_image


def create_perlin_maps(dicom_path, labels_png_path, mat_output_path, labels_npy_output_path, seed=42, plot=False):
    # Load DICOM
    dicom = pydicom.dcmread(dicom_path).pixel_array
    dicom = (dicom - dicom.min()) / (dicom.max() - dicom.min())

    # Extract contours and labels
    abdomen_mask = process_image(dicom)
    label_image = np.array(Image.open(labels_png_path))
    labels = extract_labels(label_image)
    labels.pop("Background", None)
    labels['abdomen_mask'] = abdomen_mask

    edges = cv2.Canny((dicom * 255).astype(np.uint8), 50, 150)
    dist = distance_transform_edt(~edges.astype(bool)).astype(float)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

    # Perlin noise
    h, w = dicom.shape
    noise = OpenSimplex(seed=seed)
    perlin = np.array([[noise.noise2d(x * 0.05, y * 0.05) for x in range(w)] for y in range(h)])
    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min())

    # Create maps
    t1_range, t2_range, pd_range = (500, 2000), (30, 150), (0.5, 1.0)
    modulated = perlin * dist_norm

    t1_map = t1_range[0] + modulated * (t1_range[1] - t1_range[0])
    t2_map = t2_range[0] + modulated * (t2_range[1] - t2_range[0])
    pd_map = pd_range[0] + modulated * (pd_range[1] - pd_range[0])

    t1_map[~abdomen_mask] = 0
    t2_map[~abdomen_mask] = 0
    pd_map[~abdomen_mask] = 0

    # Crop
    props = measure.regionprops(abdomen_mask.astype(int))
    if props:
        min_row, min_col, max_row, max_col = props[0].bbox
        padding = 10
        min_row, min_col = max(0, min_row - padding), max(0, min_col - padding)
        max_row, max_col = min(h, max_row + padding), min(w, max_col + padding)

        t1_map = t1_map[min_row:max_row, min_col:max_col]
        t2_map = t2_map[min_row:max_row, min_col:max_col]
        pd_map = pd_map[min_row:max_row, min_col:max_col]

        for key in labels.keys():
            labels[key] = labels[key][min_row:max_row, min_col:max_col]

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, (m, name) in zip(axes[0], [(t1_map, 'T1'), (t2_map, 'T2'), (pd_map, 'PD')]):
            im = ax.imshow(m, cmap='hot')
            ax.set_title(name)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        for ax, (m, name) in zip(axes[1], [(t1_map, 'T1'), (t2_map, 'T2'), (pd_map, 'PD')]):
            ax.hist(m[m > 0].flatten(), bins=50, alpha=0.7)
            ax.set_title(f'{name} Hist')
        plt.tight_layout()
        plt.show()

    # stacked = np.stack([pd_map, t1_map, t2_map, np.zeros_like(t1_map), np.zeros_like(t1_map)], axis=-1)
    # savemat(mat_output_path, {'cropped_brain': stacked})
    # np.save(labels_npy_output_path, labels)



def main():
    single_dicom_inphase_file_path = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00016.dcm"
    single_labels_file_path = r"C:\Users\perez\Desktop\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\Ground\IMG-0004-00046.png"

    create_perlin_maps(single_dicom_inphase_file_path, r"output.mat", r"labels.npy", seed=42, plot=True)
    create_perlin_maps(single_dicom_inphase_file_path, single_labels_file_path, r"output.mat", r"labels.npy", seed=42, plot=True)


if __name__ == '__main__':
    main()

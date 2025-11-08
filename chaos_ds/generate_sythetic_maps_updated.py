import matplotlib.pyplot as plt
import pydicom
from read_dicom_from_dirs import extract_labels, create_overlay_image
from PIL import Image

from find_countor import process_image
import numpy as np
import MRzeroCore as mr0
from skimage import measure
import json

def labels_to_coco(labels, save_path):
    """Convert labels dict to COCO format and save"""

    def mask_to_polygon(mask):
        polygons = []
        for contour in measure.find_contours(mask, 0.5):
            polygon = np.flip(contour, axis=1).flatten().tolist()
            if len(polygon) > 4:
                polygons.append(polygon)
        return polygons

    def get_bbox(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return [0, 0, 0, 0]
        return [int(cols.min()), int(rows.min()), int(cols.max() - cols.min()), int(rows.max() - rows.min())]

    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'abdomen', 'supercategory': ''}]
    }

    image_id, ann_id = 1, 1

    for label_name, mask in labels.items():
        polygons = mask_to_polygon(mask)

        coco_data['images'].append({
            'id': image_id,
            'file_name': f'{label_name}.png',
            'height': mask.shape[0],
            'width': mask.shape[1]
        })

        for polygon in polygons:
            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': 1,
                'segmentation': [polygon],
                'area': float(np.sum(mask)),
                'bbox': get_bbox(mask),
                'iscrowd': 0
            })
            ann_id += 1

        image_id += 1

    with open(save_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

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
    # remove key background from labels
    labels.pop("Background", None)
    labels['abdomen_mask'] = abdomen_mask

    labels_to_coco(labels, r"C:\Users\perez\Desktop\phantom\abdominal_phantom_annotations.json")
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

    from skimage import exposure
    t1_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask],
                                                images['T1'][brain_mask])
    t1_abdomen = np.zeros_like(dicom_inphase_image)
    t1_abdomen[abdomen_mask] = t1_matched_vals

    t2_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask],
                                                images['T2'][brain_mask])
    t2_abdomen = np.zeros_like(dicom_inphase_image)
    t2_abdomen[abdomen_mask] = t2_matched_vals

    pd_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask],
                                                images['PD'][brain_mask])
    pd_abdomen = np.zeros_like(dicom_inphase_image)
    pd_abdomen[abdomen_mask] = pd_matched_vals

    t2_dash_matched_vals = exposure.match_histograms(dicom_inphase_image[abdomen_mask],
                                                     images['T2_dash'][brain_mask])
    t2_dash_abdomen = np.zeros_like(dicom_inphase_image)
    t2_dash_abdomen[abdomen_mask] = t2_dash_matched_vals

    # Display the maps with histograms
    maps = [t1_abdomen, t2_abdomen, pd_abdomen, t2_dash_abdomen]
    fig, axes = plt.subplots(2, len(maps), figsize=(15, 10))
    labels = ['T1 Map', 'T2 Map', 'M0 Map', 'T2 dash Map']

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

    constant_d = np.zeros((H, W), dtype=np.complex128)
    constant_d[brain_mask] = 5.0 + 3.0j  # specific complex value everywhere mask is True

    from scipy.io import savemat

    stacked = np.stack([
        pd_abdomen,  # index 0
        t1_abdomen,  # index 1
        t2_abdomen,  # index 2
        images['B0'],  # index 3
        images['B1']  # index 4
    ], axis=-1)

    savemat(
        r"C:\Users\perez\Desktop\phantom\abdominal_phantom.mat",
        {
            'cropped_brain': stacked,
        }
    )


# Example usage
if __name__ == "__main__":
    main()

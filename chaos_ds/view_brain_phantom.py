import numpy as np
import MRzeroCore as mr0
import matplotlib.pyplot as plt


def main():
    obj_p = mr0.VoxelGridPhantom.brainweb(r"C:\Users\perez\Desktop\phantom\subject05.npz")

    for slice_number in range(128):
        images = {"B0": obj_p.B0[:, :, slice_number],
                  "B1": obj_p.B1[0][:, :, slice_number],
                  "D": obj_p.D[:, :, slice_number],
                  "PD": obj_p.PD[:, :, slice_number],
                  "T1": obj_p.T1[:, :, slice_number],
                  "T2": obj_p.T2[:, :, slice_number],
                  "T2_dash": obj_p.T2dash[:, :, slice_number]}

        brain_mask = images['PD'] > 0

        fig, axes = plt.subplots(3, 7, figsize=(16, 8))

        for i, (label, img) in enumerate(images.items()):
            img = np.abs(img)

            ax_img = axes[0, i]
            im = ax_img.imshow(img, cmap='gray')
            ax_img.set_title(label)
            ax_img.axis('off')
            plt.colorbar(im, ax=ax_img)

            ax_hist = axes[1, i]
            ax_hist.hist(img[brain_mask].flatten(), bins=50, color='blue', alpha=0.7)
            ax_hist.set_title(f'{label} Histogram')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Frequency')

        axes[2, 0].set_title('brain mask')
        axes[2, 0].imshow(brain_mask, cmap='gray')
        axes[2, 0].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
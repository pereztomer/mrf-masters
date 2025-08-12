import MRzeroCore as mr0

import time
import pypulseq as pp

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np

def display_seq(seq_file):
    # for plotting graph of the signal evolotion
    seq = pp.Sequence()
    seq.read(seq_file)
    signal = mr0.util.simulate_2d(seq)
    # seq.plot(plot_now=False)
    # mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
    # plt.show()
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat("numerical_brain_cropped.mat")
    obj_p = obj_p.build()
    # Simulate the sequence
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)
    kspace = seq0.get_kspace()
    kspace_2d = kspace[:, :2]

    reco = mr0.reco_adjoint(signal, seq0.cuda().get_kspace(), resolution=(108, 108, 1), FOV=(0.22, 0.22, 1))
    mag_image = torch.abs(reco).detach().cpu().numpy()
    return mag_image


def main():
    # Your two images
    inversion_mag_image = display_seq(seq_file=r"epi_gre_mrf_epi_inversion.seq")
    no_inversion_map_image = display_seq(seq_file=r"epi_gre_mrf_epi_no_inversion.seq")

    # Shared scale for the first two images
    vmin = min(inversion_mag_image.min(), no_inversion_map_image.min())
    vmax = max(inversion_mag_image.max(), no_inversion_map_image.max())

    # Difference image (with symmetric scale)
    diff_img = inversion_mag_image - no_inversion_map_image
    diff_max = np.max(np.abs(diff_img))

    # Plot side-by-side + difference
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(inversion_mag_image, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Image 1")
    axes[0].axis('off')

    im1 = axes[1].imshow(no_inversion_map_image, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("Image 2")
    axes[1].axis('off')

    im2 = axes[2].imshow(diff_img, cmap='bwr', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title("Difference")
    axes[2].axis('off')

    # Colorbar for Image 1 & 2
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # (left, bottom, width, height)
    fig.colorbar(im0, cax=cbar_ax1, label="Intensity")

    # Colorbar for difference
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label="Difference")

    plt.subplots_adjust(wspace=0.05)
    plt.show()


if __name__ == '__main__':
    main()

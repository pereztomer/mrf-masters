import MRzeroCore as mr0
import pypulseq as pp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def main():
    seq_file = r"sequences/2025-05-14_epi_Nx_192_Ny_192_part_fourier_factor_1_repetetions_1.seq"
    seq = pp.Sequence()
    seq.read(seq_file)
    signal = mr0.util.simulate_2d(seq)
    seq.plot(plot_now=False)
    mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
    plt.show()

    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat("numerical_brain_cropped.mat")
    obj_p = obj_p.build()
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=True)

    # kspace = signal.reshape(184, 72)
    # plt.imshow(torch.log(kspace.abs()+1))
    # plt.show()
    reco = mr0.reco_adjoint(signal, seq0.get_kspace(), resolution=(192, 192, 1), FOV=(0.22, 0.22, 1))
    plt.figure()
    plt.subplot(121)
    plt.title("Magnitude")
    plt.imshow(reco[:, :, 0].T.abs(), origin="lower")
    plt.colorbar()
    plt.subplot(122)
    plt.title("Phase")
    plt.imshow(reco[:, :, 0].T.angle(), origin="lower", vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()

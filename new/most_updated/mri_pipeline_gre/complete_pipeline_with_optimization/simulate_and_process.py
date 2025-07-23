import MRzeroCore as mr0
import matplotlib.pyplot as plt
import pypulseq as pp
import torch
import eqdist_grappa_cuda
import numpy as np


def preprocess_raw_data(seq, signal, R, Nread, Nphase_in_practice, fourier_factor, time_steps, num_coils):
    kspace_frequencies = torch.Tensor(seq.calculate_kspace()[0])
    shots = []
    x_freq_per_shot = []
    y_freq_per_shot = []
    for index in range(R):
        single_shot = signal[index * Nread * Nphase_in_practice:(index + 1) * Nread * Nphase_in_practice]
        kspace_shot = kspace_frequencies[:, index * Nread * Nphase_in_practice:(index + 1) * Nread * Nphase_in_practice]

        x_freq_shot = kspace_shot[0].unsqueeze(1)
        y_freq_shot = kspace_shot[1].unsqueeze(1)
        x_freq_shot = torch.reshape(x_freq_shot, (Nphase_in_practice, Nread)).clone().T
        y_freq_shot = torch.reshape(y_freq_shot, (Nphase_in_practice, Nread)).clone().T
        x_freq_shot[:, 0::2] = torch.flip(x_freq_shot[:, 0::2], [0])[:, :]
        y_freq_shot[:, 0::2] = torch.flip(y_freq_shot[:, 0::2], [0])[:, :]
        expanded_x_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_x_freq_per_shot[:, index:int(Nread * fourier_factor):R] = x_freq_shot

        expanded_y_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_y_freq_per_shot[:, index:int(Nread * fourier_factor):R] = y_freq_shot
        x_freq_per_shot.append(expanded_x_freq_per_shot)
        y_freq_per_shot.append(expanded_y_freq_per_shot)

        # Initialize tensor with coils as last dimension
        expanded_kspace_per_shot = torch.zeros((Nread, Nread, num_coils), dtype=torch.complex64)
        # For each coil
        for coil in range(num_coils):
            single_shot_coil = single_shot[:, coil]  # Extract one coil
            single_shot_coil = torch.reshape(single_shot_coil, (Nphase_in_practice, Nread)).clone().T
            single_shot_coil[:, 0::2] = torch.flip(single_shot_coil[:, 0::2], [0])[:, :]
            expanded_kspace_per_shot[:, index: int(Nread * fourier_factor):R, coil] = single_shot_coil

        shots.append(expanded_kspace_per_shot)

    time_series_shots = []
    time_series_x_freq_per_shot = []
    time_series_y_freq_per_shot = []
    for step in range(time_steps):
        kspace_shot = kspace_frequencies[:,
                      (R + step) * Nread * Nphase_in_practice: (R + step + 1) * Nread * Nphase_in_practice]

        x_freq_shot = kspace_shot[0].unsqueeze(1)
        y_freq_shot = kspace_shot[1].unsqueeze(1)

        x_freq_shot = torch.reshape(x_freq_shot, (Nphase_in_practice, Nread)).clone().T
        y_freq_shot = torch.reshape(y_freq_shot, (Nphase_in_practice, Nread)).clone().T

        x_freq_shot[:, 0::2] = torch.flip(x_freq_shot[:, 0::2], [0])[:, :]
        y_freq_shot[:, 0::2] = torch.flip(y_freq_shot[:, 0::2], [0])[:, :]

        expanded_x_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_x_freq_per_shot[:, 0:int(Nread * fourier_factor):R] = x_freq_shot

        expanded_y_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_y_freq_per_shot[:, 0:int(Nread * fourier_factor):R] = y_freq_shot

        time_series_x_freq_per_shot.append(expanded_x_freq_per_shot)
        time_series_y_freq_per_shot.append(expanded_y_freq_per_shot)

        expanded_kspace_per_shot = torch.zeros((Nread, Nread, num_coils), dtype=torch.complex64)

        single_shot = signal[(R + step) * Nread * Nphase_in_practice: (R + step + 1) * Nread * Nphase_in_practice]
        for coil in range(num_coils):
            single_shot_coil = single_shot[:, coil]  # Extract one coil
            single_shot_coil = torch.reshape(single_shot_coil, (Nphase_in_practice, Nread)).clone().T
            single_shot_coil[:, 0::2] = torch.flip(single_shot_coil[:, 0::2], [0])[:, :]
            expanded_kspace_per_shot[:, 0: int(Nread * fourier_factor):R, coil] = single_shot_coil

        time_series_shots.append(expanded_kspace_per_shot)

    block_size = (4, 4)
    acc_factors_2d = (1, 3)
    regularization_factor = 0.00001
    device = "cuda"
    calibration_data = torch.sum(torch.stack(shots), dim=0)
    calibration_images_per_coil = []
    for coil in range(calibration_data.shape[-1]):
        img_coil = torch.abs(torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(calibration_data[:, :, coil]))))
        calibration_images_per_coil.append(img_coil)

    images_per_coil = torch.stack(calibration_images_per_coil, axis=-1)
    calibration_img_sos = torch.sqrt(torch.sum(images_per_coil ** 2, axis=-1))

    grappa_weights_torch = eqdist_grappa_cuda.GRAPPA_calibrate_weights_2d_torch(calibration_data,
                                                                                acc_factors_2d,
                                                                                device,
                                                                                block_size,
                                                                                regularization_factor)

    for time_step in range(time_steps):
        step = time_series_shots[time_step]
        kspace_recon_kykxc, image_coilcombined_sos, unmixing_map_coilWise = eqdist_grappa_cuda.GRAPPA_interpolate_imageSpace_2d_torch(
            step, acc_factors_2d, block_size, grappa_weights_torch, device)

        images_per_coil = []
        for coil in range(kspace_recon_kykxc.shape[-1]):
            img_coil = torch.abs(torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(kspace_recon_kykxc[:, :, coil]))))
            images_per_coil.append(img_coil)

        images_per_coil = torch.stack(images_per_coil, axis=-1)
        img_sos = torch.sqrt(torch.sum(images_per_coil ** 2, axis=-1))
        time_series_shots[time_step] = img_sos

    time_series_shots = torch.stack(time_series_shots, dim=0)

    return calibration_img_sos, time_series_shots


def simulate_and_process_mri(obj_p, seq_file_path, num_coils):
    # sequence parameter loading:
    seq_pulseq = pp.Sequence()
    seq_pulseq.read(seq_file_path)
    Nx = int(seq_pulseq.get_definition('Nx'))
    NySampled = int(seq_pulseq.get_definition('NySampled'))
    R = int(seq_pulseq.get_definition('AccelerationFactor'))
    fourier_factor = seq_pulseq.get_definition("PartialFourierFactor")
    time_steps = int(seq_pulseq.get_definition("TimeSteps"))

    seq_mr0 = mr0.Sequence.import_file(seq_file_path)

    # MR operations
    graph = mr0.compute_graph(seq_mr0.cuda(), obj_p.cuda(), 2048, 1e-3)
    signal = mr0.execute_graph(graph, seq_mr0.cuda(), obj_p.cuda(), print_progress=True)

    calibration_img_sos, time_series_shots = preprocess_raw_data(seq_pulseq, signal, R, Nx, NySampled, fourier_factor, time_steps,num_coils=num_coils)

    return calibration_img_sos, time_series_shots

import MRzeroCore as mr0
import torch
import pypulseq as pp


def load_mr0_data_torch(seq_file, phantom_path="numerical_brain_cropped.mat"):
    seq = pp.Sequence()
    seq.read(seq_file)
    ktraj_adc, _, _, _, t_adc = seq.calculate_kspace()
    ktraj_adc = torch.from_numpy(ktraj_adc).to(device='cuda',dtype=torch.float32)
    t_adc = torch.from_numpy(t_adc).to(device='cuda', dtype=torch.float32)
    Nx = 128
    Ny = 128
    NySampled = 128
    # freq_encoding_steps = 91
    freq_encoding_steps = 128
    R = 1
    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
    obj_p = obj_p.interpolate(int(Nx), int(Ny), 1)
    obj_p = obj_p.build()

    # Simulate the sequence
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    # Then reshape and permute
    signal = signal.reshape(NySampled, freq_encoding_steps)
    ktraj_adc = ktraj_adc.reshape(3, NySampled, freq_encoding_steps)
    t_adc = t_adc.reshape(NySampled, freq_encoding_steps)
    return signal, ktraj_adc, t_adc


def main():
    # seq_path = "epi_se_pypulseq.seq"
    seq_path = "epi_gre.seq"
    # seq_path = "epi_se_rs_pypulseq.seq"
    signal, ktraj_adc, t_adc = load_mr0_data_torch(seq_file=seq_path)
    from single_shot_analysis import fix_single_shot_epi
    fix_single_shot_epi(signal, ktraj_adc, t_adc)

    print("heree")


if __name__ == '__main__':
    main()

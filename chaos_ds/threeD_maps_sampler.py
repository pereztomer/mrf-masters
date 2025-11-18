import matplotlib.pyplot as plt
import numpy as np
import MRzeroCore as mr0
import cv2


class CorrelatedSampler3D:
    """3D binned sampler for T1, T2, PD correlation"""

    def __init__(self, t1, t2, pd, n_bins=15):
        self.n_bins = n_bins

        t1 = np.abs(t1)
        t2 = np.abs(t2)
        pd = np.abs(pd)

        # Define bin edges
        self.t1_min, self.t1_max = t1.min(), t1.max()
        self.t2_min, self.t2_max = t2.min(), t2.max()
        self.pd_min, self.pd_max = pd.min(), pd.max()

        self.t1_edges = np.linspace(self.t1_min, self.t1_max, n_bins + 1)
        self.t2_edges = np.linspace(self.t2_min, self.t2_max, n_bins + 1)
        self.pd_edges = np.linspace(self.pd_min, self.pd_max, n_bins + 1)

        # Create 3D bin structure
        self.bins = {}

        # Assign each voxel to a 3D bin
        t1_indices = np.digitize(t1, self.t1_edges) - 1
        t2_indices = np.digitize(t2, self.t2_edges) - 1
        pd_indices = np.digitize(pd, self.pd_edges) - 1

        # Clamp indices to valid range
        t1_indices = np.clip(t1_indices, 0, n_bins - 1)
        t2_indices = np.clip(t2_indices, 0, n_bins - 1)
        pd_indices = np.clip(pd_indices, 0, n_bins - 1)

        # Store voxel values in bins
        for i in range(len(t1)):
            bin_key = (t1_indices[i], t2_indices[i], pd_indices[i])
            if bin_key not in self.bins:
                self.bins[bin_key] = {'t1': [], 't2': [], 'pd': []}

            self.bins[bin_key]['t1'].append(t1[i])
            self.bins[bin_key]['t2'].append(t2[i])
            self.bins[bin_key]['pd'].append(pd[i])

        print(f"Sampler initialized: {len(self.bins)} non-empty bins out of {n_bins ** 3} total bins")

    def sample(self, t1, t2, pd):
        """
        Sample correlated T1, T2, PD values from bin corresponding to input values.
        """
        t1_idx = np.digitize(t1, self.t1_edges) - 1
        t2_idx = np.digitize(t2, self.t2_edges) - 1
        pd_idx = np.digitize(pd, self.pd_edges) - 1

        t1_idx = np.clip(t1_idx, 0, self.n_bins - 1)
        t2_idx = np.clip(t2_idx, 0, self.n_bins - 1)
        pd_idx = np.clip(pd_idx, 0, self.n_bins - 1)

        bin_key = (t1_idx, t2_idx, pd_idx)

        if bin_key not in self.bins or len(self.bins[bin_key]['t1']) == 0:
            return None

        idx = np.random.randint(0, len(self.bins[bin_key]['t1']))

        return {
            't1': self.bins[bin_key]['t1'][idx],
            't2': self.bins[bin_key]['t2'][idx],
            'pd': self.bins[bin_key]['pd'][idx]
        }

    def get_bin_stats(self):
        """Return statistics about bin occupancy"""
        occupancies = [len(self.bins[k]['t1']) for k in self.bins]
        return {
            'total_non_empty_bins': len(self.bins),
            'total_possible_bins': self.n_bins ** 3,
            'min_occupancy': min(occupancies),
            'max_occupancy': max(occupancies),
            'mean_occupancy': np.mean(occupancies)
        }

    def get_occupied_bins_info(self, top_n=20):
        """Get info about most occupied bins"""
        occupied_bins = []
        for bin_key, data in self.bins.items():
            occupancy = len(data['t1'])
            t1_idx, t2_idx, pd_idx = bin_key

            t1_center = (self.t1_edges[t1_idx] + self.t1_edges[t1_idx + 1]) / 2
            t2_center = (self.t2_edges[t2_idx] + self.t2_edges[t2_idx + 1]) / 2
            pd_center = (self.pd_edges[pd_idx] + self.pd_edges[pd_idx + 1]) / 2

            occupied_bins.append({
                'bin_key': bin_key,
                'occupancy': occupancy,
                't1_center': t1_center,
                't2_center': t2_center,
                'pd_center': pd_center
            })

        occupied_bins.sort(key=lambda x: x['occupancy'], reverse=True)
        return occupied_bins[:top_n]


def main():
    obj_p = mr0.VoxelGridPhantom.brainweb(r"C:\Users\perez\Desktop\phantom\subject05.npz")

    # Collect all brain voxels from all slices
    t1_all = []
    t2_all = []
    pd_all = []

    print("Collecting data from all slices...")
    for slice_number in range(128):
        images = {"T1": obj_p.T1[:, :, slice_number],
                  "T2": obj_p.T2[:, :, slice_number],
                  "PD": obj_p.PD[:, :, slice_number]}

        brain_mask = np.abs(images['PD']) > 0

        t1_slice = np.abs(images['T1'])[brain_mask]
        t2_slice = np.abs(images['T2'])[brain_mask]
        pd_slice = np.abs(images['PD'])[brain_mask]

        t1_all.append(t1_slice)
        t2_all.append(t2_slice)
        pd_all.append(pd_slice)

    # Concatenate all slices
    t1_all = np.concatenate(t1_all)
    t2_all = np.concatenate(t2_all)
    pd_all = np.concatenate(pd_all)

    print(f"Total brain voxels collected: {len(t1_all)}")

    # Create sampler
    print("\nBuilding 3D correlated sampler...")
    sampler = CorrelatedSampler3D(t1_all, t2_all, pd_all, n_bins=15)

    stats = sampler.get_bin_stats()
    print(f"Bin statistics: {stats}")

    # Show top occupied bins
    print("\nTop 20 occupied bins:")
    top_bins = sampler.get_occupied_bins_info(top_n=68)
    for i, b in enumerate(top_bins):
        print(
            f"{i + 1}. T1={b['t1_center']:.2f}, T2={b['t2_center']:.2f}, PD={b['pd_center']:.3f} -> {b['occupancy']} voxels")

    # # Test sampling from top bins
    # print("\n\nTest sampling from top 5 bins:")
    # for i, b in enumerate(top_bins[:5]):
    #     print(f"\nBin {i + 1} (T1={b['t1_center']:.2f}, T2={b['t2_center']:.2f}, PD={b['pd_center']:.3f}):")
    #     for _ in range(3):
    #         sample = sampler.sample(b['t1_center'], b['t2_center'], b['pd_center'])
    #         if sample:
    #             print(f"  -> T1={sample['t1']:.3f}, T2={sample['t2']:.3f}, PD={sample['pd']:.3f}")


if __name__ == "__main__":
    main()
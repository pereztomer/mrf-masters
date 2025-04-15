import numpy as np


def analyze_kspace_dat(filename):
    """
    Analyze the k-space data file to determine its structure
    """
    # Read raw data
    raw_data = np.fromfile(filename, dtype=np.complex64)
    total_points = len(raw_data)

    # Calculate absolute values
    magnitudes = np.abs(raw_data)

    # Count zeros
    zero_points = np.sum(magnitudes <= 1e-4)

    print(f"Total points: {len(raw_data)}")
    print(f"Points with zero magnitude: {zero_points}")
    print(f"Percentage of zeros: {(zero_points / len(raw_data)) * 100:.2f}%")

    # Calculate potential number of coils
    matrix_points = 128 * 128  # k-space matrix size
    potential_coils = total_points / matrix_points


    print(f"Total data points: {total_points}")
    print(f"K-space matrix points: {matrix_points}")
    print(f"Potential number of coils: {potential_coils}")

    return raw_data


def reshape_kspace(raw_data, matrix_size=(128, 128)):
    """
    Reshape the raw data into k-space based on detected dimensions
    """
    total_points = len(raw_data)
    matrix_points = matrix_size[0] * matrix_size[1]
    num_coils = total_points // matrix_points

    # Reshape data
    kspace = raw_data.reshape(num_coils, matrix_size[0], matrix_size[1])
    return kspace


def main():
    # Example usage
    filename = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1\meas_MID00045_FID14597_pulseq.dat"

    kspace_data = analyze_kspace_dat(filename)

    # Basic visualization of magnitude
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    for coil in range(min(4, kspace_data.shape[0])):  # Show first 4 coils
        plt.subplot(1, 4, coil + 1)
        plt.imshow(np.log(np.abs(kspace_data[coil])), cmap='gray')
        plt.title(f'Coil {coil + 1}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
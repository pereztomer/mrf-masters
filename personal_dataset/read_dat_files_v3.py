
from mapvbvd import mapVBVD
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gmean

# File paths
filename = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1\phantom_2.dat"
output_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1\phantom_2_output"
#(['Col', 'Cha', 'Lin', 'Par', 'Set'])

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read the Siemens raw data file
twixObj = mapVBVD(filename)
twixObj[0].image.squeeze = True
kspace_data = twixObj[0].image['']
print("K-space data shape:", kspace_data.shape)
# Expected shape (64, 52, 63, 63, 2) - [Col, Cha, Lin, Par, Set]

# Function to reconstruct a slice
def reconstruct_slice(kspace):
    # Apply 2D FFT
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return np.abs(image)

# Extract dimensions
num_cols = kspace_data.shape[0]
num_channels = kspace_data.shape[1]  # Coils
num_lines = kspace_data.shape[2]
num_partitions = kspace_data.shape[3]
num_sets = kspace_data.shape[4]

print(f"Data dimensions: Columns={num_cols}, Channels={num_channels}, Lines={num_lines}, Partitions={num_partitions}, Sets={num_sets}")
print("Averaging across channels (coils)...")

# Process each set
for set_idx in range(num_sets):
    # Create directory for this set
    set_dir = os.path.join(output_dir, f'set_{set_idx:02d}')
    os.makedirs(set_dir, exist_ok=True)

    print(f"Processing set {set_idx}/{num_sets-1}")

    # Process each partition
    for partition_idx in range(num_partitions):
        # Create directory for this partition
        partition_dir = os.path.join(set_dir, f'partition_{partition_idx:03d}')
        os.makedirs(partition_dir, exist_ok=True)

        # Extract k-space data for all channels for this partition and set
        # Shape: [Col, Cha, Lin, Par, Set] -> [Col, Cha, Lin]
        kspace_slice = kspace_data[:, :, :, partition_idx, set_idx]

        # Average across channels (coils)
        # Small epsilon to avoid zeros in geometric mean
        avg_kspace = gmean(np.abs(kspace_slice) + 1e-10, axis=1) - 1e-10

        # Save k-space image
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(np.log(avg_kspace + 1)), cmap='viridis')
        plt.colorbar()
        plt.title(f'Average K-space - Set {set_idx} - Partition {partition_idx}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(partition_dir, f'kspace.png'), dpi=150)
        plt.close()

        # Reconstruct and save image space
        avg_image = reconstruct_slice(avg_kspace)
        plt.figure(figsize=(10, 10))
        plt.imshow(avg_image, cmap='gray')
        plt.colorbar()
        plt.title(f'Average Image - Set {set_idx} - Partition {partition_idx}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(partition_dir, f'image.png'), dpi=150)
        plt.close()

# Create montages for each set
for set_idx in range(num_sets):
    print(f"Creating montage for set {set_idx}")

    plt.figure(figsize=(20, 20))
    rows = int(np.ceil(np.sqrt(num_partitions)))
    cols = int(np.ceil(num_partitions / rows))

    for partition_idx in range(num_partitions):
        # Extract data for all channels for this partition
        kspace_slice = kspace_data[:, :, :, partition_idx, set_idx]

        # Average across channels (coils)
        avg_kspace = gmean(np.abs(kspace_slice) + 1e-10, axis=1) - 1e-10

        # Reconstruct image
        avg_image = reconstruct_slice(avg_kspace)

        # Add to montage
        plt.subplot(rows, cols, partition_idx + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.title(f'Partition {partition_idx}')
        plt.axis('off')

    plt.suptitle(f'All Partitions - Set {set_idx}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'set_{set_idx}_montage.png'), dpi=200)
    plt.close()

print(f"All images saved to: {output_dir}")
print("Directory structure:")
print(f"{output_dir}/set_XX/partition_XXX/")
print("Montages saved to the main directory.")
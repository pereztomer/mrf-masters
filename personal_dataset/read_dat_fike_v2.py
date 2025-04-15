from mapvbvd import mapVBVD
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('TkAgg')
filename = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1\phantom_2.dat"
output_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1\phantom_2_output_images"
# Create a directory to save the images
os.makedirs(output_dir, exist_ok=True)

# Read the Siemens raw data file
twixObj = mapVBVD(filename)

mdh_flags = twixObj[0].MDH_flags()
for mdh in mdh_flags:
    print(f'MDH: {mdh}')
    print(twixObj[0][mdh])
    print('\n')

twixObj[0].image.squeeze = True

kspace_data = twixObj[0].image['']
print("K-space data shape:", kspace_data.shape)

image_data = twixObj[0].image

# Access squeezed dimensions
dimensions = image_data.sqzDims
print(dimensions)

# Access the size of each dimension
dimension_sizes = image_data.sqzSize
print(dimension_sizes)

exit()
# Function to reconstruct a slice
def reconstruct_slice(kspace, apply_epi_correction=True):
    # Apply EPI correction (flip alternate lines)
    corrected_kspace = kspace.copy()
    if apply_epi_correction:
        for i in range(1, corrected_kspace.shape[1], 2):
            corrected_kspace[:, i] = corrected_kspace[:, i][::-1]

    # Apply 2D FFT
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(corrected_kspace)))

    return np.abs(image)


# 1. Display images from individual coils
# Select a subset of coils to display (showing all 52 would be too many)
coils_to_display = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
middle_partition = kspace_data.shape[1] // 2
set_idx = 0

plt.figure(figsize=(15, 12))
for i, coil_idx in enumerate(coils_to_display):
    # Extract data for this coil, middle partition, first set
    coil_kspace = kspace_data[:, coil_idx, :, middle_partition,set_idx]
    coil_image = reconstruct_slice(coil_kspace, apply_epi_correction=False)

    plt.subplot(3, 4, i + 1)
    plt.imshow(coil_image, cmap='gray')
    plt.title(f'Coil {coil_idx}')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'coil_images.png'))
plt.show()


from scipy.stats import gmean

slice_kspace_data_multi_coil = kspace_data[:, :, :, middle_partition, set_idx]
kspace_slice_signle_coil = gmean(slice_kspace_data_multi_coil, axis=1)
# display kspace and image space
plt.imshow(np.abs(np.log(kspace_slice_signle_coil.T + 1)))
plt.title("K-space")
plt.show()
print("here")

image_slice_signle_coil = reconstruct_slice(kspace_slice_signle_coil, apply_epi_correction=False)
plt.imshow(np.abs(image_slice_signle_coil.T ))
plt.title("Image")
plt.show()
# # 2. Display images from different sets
# # Combine all coils using sum-of-squares
# set_images = []
# for set_idx in range(kspace_data.shape[4]):
#     # Combine all channels for this set
#     combined_kspace = np.sqrt(np.sum(np.abs(kspace_data[:, :, :, middle_partition, set_idx]) ** 2, axis=1))
#     set_image = reconstruct_slice(combined_kspace)
#     set_images.append(set_image)
#
#     plt.figure(figsize=(10, 8))
#     plt.imshow(set_image, cmap='gray')
#     plt.title(f'Set {set_idx}')
#     plt.colorbar()
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'set_{set_idx}.png'))
#     plt.show()
#
# # 3. Display a montage of partitions
# # Use the first set and combine all coils
# set_idx = 0
# combined_kspace = np.sqrt(np.sum(np.abs(kspace_data[:, :, :, :, set_idx]) ** 2, axis=1))
#
# # Display a subset of partitions
# plt.figure(figsize=(15, 12))
# partitions_to_display = min(16, combined_kspace.shape[2])
# rows = 4
# cols = 4
#
# for i in range(partitions_to_display):
#     # Select partitions evenly spaced across the volume
#     partition_idx = i * combined_kspace.shape[2] // partitions_to_display
#
#     partition_kspace = combined_kspace[:, :, partition_idx]
#     partition_image = reconstruct_slice(partition_kspace)
#
#     plt.subplot(rows, cols, i + 1)
#     plt.imshow(partition_image, cmap='gray')
#     plt.title(f'Partition {partition_idx}')
#     plt.axis('off')
#
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'partition_montage.png'))
# plt.show()
#
# # 4. Create a 3D visualization using slices in three orthogonal planes
# set_idx = 0
# combined_kspace = np.sqrt(np.sum(np.abs(kspace_data[:, :, :, :, set_idx]) ** 2, axis=1))
#
# # Reconstruct all partitions
# volume = np.zeros((combined_kspace.shape[0], combined_kspace.shape[1], combined_kspace.shape[2]))
# for i in range(combined_kspace.shape[2]):
#     volume[:, :, i] = reconstruct_slice(combined_kspace[:, :, i])
#
# # Display orthogonal views
# plt.figure(figsize=(15, 5))
#
# # Axial view (as we've been showing)
# plt.subplot(1, 3, 1)
# middle_slice_axial = volume[:, :, volume.shape[2] // 2]
# plt.imshow(middle_slice_axial, cmap='gray')
# plt.title('Axial (Middle Slice)')
# plt.axis('off')
#
# # Sagittal view
# plt.subplot(1, 3, 2)
# middle_slice_sagittal = volume[:, volume.shape[1] // 2, :]
# plt.imshow(middle_slice_sagittal, cmap='gray')
# plt.title('Sagittal (Middle Slice)')
# plt.axis('off')
#
# # Coronal view
# plt.subplot(1, 3, 3)
# middle_slice_coronal = volume[volume.shape[0] // 2, :, :]
# plt.imshow(middle_slice_coronal, cmap='gray')
# plt.title('Coronal (Middle Slice)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'orthogonal_views.png'))
# plt.show()
#
# print(f"All images saved to: {output_dir}")

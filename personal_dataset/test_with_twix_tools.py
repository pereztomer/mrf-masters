import twixtools
import numpy as np
import matplotlib.pyplot as plt
import os

# example_dir = r'C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1'
# Define output directory for saved images
# output_dir = r'C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\output_images_2'

# Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)


def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace


def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig) ** 2, axis))


# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\meas_MID00155_FID15914_pypulseq_tomer.dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\meas_MID00115_FID16031_Tomer_epi2.dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\epi_with_acceleration\meas_MID00071_FID16378_epi_with_acceleration.dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_3_epi_se_rs\meas_MID00070_FID16377_epi_se_rs.dat"
# path = r"C:\Users\perez\Downloads\epi (1).dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\epi_with_acceleration\meas_MID00071_FID16378_epi_with_acceleration.dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\meas_MID00115_FID16031_Tomer_epi2.dat"
# path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\meas_MID00155_FID15914_pypulseq_tomer.dat"
path = r"C:\Users\perez\Desktop\masters\mri_research\code\matlab\tutorials\11_from_GRE_to_EPI\data\04c_improvised_EPI.dat"
twix = twixtools.read_twix(path)

# twix is a list of measurements:
print('\nnumber of separate scans (multi-raid):', len(twix))

print('\nTR = %d ms\n' % (twix[-1]['hdr']['Phoenix']['alTR'][0] / 1000))

# list the mdh flags and line counters for every 8th measurement data block (mdb)
for mdb in twix[-1]['mdb'][::8]:
    print('line: %3d; flags:' % (mdb.cLin), mdb.get_active_flags())

# map the twix data to twix_array objects
mapped = twixtools.map_twix(twix)
im_data = mapped[-1]['image']

im_data.flags['remove_os'] = True
# make sure that we later squeeze the right dimensions:
print(im_data.non_singleton_dims)

# the twix_array object makes it easy to remove the 2x oversampling in read direction
# im_data.flags['remove_os'] = True

# read the data (array-slicing is also supported)
data = im_data[:].squeeze()
# data = data.reshape(150, 64, 52, 64)

# single_slice_image = ifftnd(single_slice_kspace, [0,1,2])
import torch
# coils_images = []
# kspace_images = []
# for coil_num in range(58):
#     spectrum = torch.fft.ifftshift(torch.Tensor(data[:,coil_num][:32]))
#     space = torch.fft.fft2(spectrum)
#     space = torch.fft.ifftshift(space)
#     coils_images.append(space)
#     kspace_images.append(spectrum)
#     # display each kspace and image
#     plt.imshow(np.log(np.abs(data[:,coil_num])+1))
#     plt.show()
#     plt.imshow(np.abs(space))
#     plt.show()
#     exit()
#
#
# stacked_coils = torch.stack(coils_images)
# single_slice_image = rms_comb(stacked_coils.numpy(), axis=0)
# stacked_kspace = torch.stack(kspace_images)
# single_slice_kspace = rms_comb(stacked_kspace.numpy(), axis=0)
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#
# axes[0].imshow(np.log(single_slice_kspace + 1), cmap='viridis')
# axes[0].set_title('kspace')
#
# axes[1].imshow(single_slice_image, cmap='gray')
# axes[1].set_title('Image phase')
#
# plt.tight_layout()
# plt.show()
#
# exit()

# Apply IFFT on spatial dimensions (row and column)
image = ifftnd(data, [0, 1])
# image = rms_comb(image, axis=1)

# Plot k-space
plt.subplot(121)
plt.title(f'k-space')
plt.imshow(np.log(np.abs(data)+1), cmap='gray', origin='lower')
plt.axis('off')

# Plot image
plt.subplot(122)
plt.title(f'Image')
plt.imshow(abs(image), cmap='gray', origin='lower')
plt.axis('off')
plt.show()
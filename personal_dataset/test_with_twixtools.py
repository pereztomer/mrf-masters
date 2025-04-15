import twixtools
import numpy as np
import matplotlib.pyplot as plt
import os

example_dir = r'C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\test_1'

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
    return np.sqrt(np.sum(abs(sig)**2, axis))


# parse the twix file
# twix = twixtools.read_twix(os.path.join(example_dir, 'phantom_2.dat'))
path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\mrf custom dataset\epi\test_2_default_scan_epi\meas_MID00155_FID15914_pypulseq_tomer.dat"
twix = twixtools.read_twix(path)

# twix is a list of measurements:
print('\nnumber of separate scans (multi-raid):', len(twix))

print('\nTR = %d ms\n'%(twix[-1]['hdr']['Phoenix']['alTR'][0]/1000))


# list the mdh flags and line counters for every 8th measurement data block (mdb)
for mdb in twix[-1]['mdb'][::8]:
    print('line: %3d; flags:'%(mdb.cLin), mdb.get_active_flags())


#
# # sort all 'imaging' mdbs into a k-space array
# image_mdbs = [mdb for mdb in twix[-1]['mdb'] if mdb.is_image_scan()]
#
# n_line = 1 + max([mdb.cLin for mdb in image_mdbs])
#
# # assume that all data were acquired with same number of channels & columns:
# n_channel, n_column = image_mdbs[0].data.shape
#
# kspace = np.zeros([n_line, n_channel, n_column], dtype=np.complex64)
# for mdb in image_mdbs:
#     kspace[mdb.cLin] = mdb.data
#
# print('\nk-space shape', kspace.shape)
#
# # reconstruct an image and show the result:
# plt.figure(figsize=[12,8])
# plt.subplot(121)
# plt.title('k-space')
# plt.imshow(abs(kspace[:,0])**0.2, cmap='gray', origin='lower')
# plt.axis('off')
#
# image = ifftnd(kspace, [0,-1])
# image = rms_comb(image)
# plt.subplot(122)
# plt.title('image')
# plt.imshow(abs(image), cmap='gray', origin='lower')
# plt.axis('off')
# plt.show()

# map the twix data to twix_array objects
mapped = twixtools.map_twix(twix)
im_data = mapped[-1]['image']

# make sure that we later squeeze the right dimensions:
print(im_data.non_singleton_dims)

# the twix_array object makes it easy to remove the 2x oversampling in read direction
# im_data.flags['remove_os'] = True

for i in range(0,150,5):
    # read the data (array-slicing is also supported)
    data = im_data[:].squeeze()
    data = data.reshape(150, 64, 52, 64)
    plt.figure(figsize=[12,8])
    plt.subplot(121)
    plt.title('k-space')
    plt.imshow(abs(data[i][:,0])**0.2, cmap='gray', origin='lower')
    plt.axis('off')

    image = ifftnd(data, [1,3])
    image = rms_comb(image, axis=2)

    plt.subplot(122)
    plt.title('image')
    plt.imshow(abs(image[i]), cmap='gray', origin='lower')
    plt.axis('off')
    plt.save()
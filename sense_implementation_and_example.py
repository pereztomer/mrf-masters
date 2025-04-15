# imports you will need
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib

matplotlib.use('TkAgg')


def show_grid(img: np.ndarray, cmap: str, title: str):
    """
    Plot all the coil channels images in a grid

    :param img: Set of images, shape (Nc,Ny,Nx)
    :param cmap: Which colormap to use
    :param title: The title of the figure (main title)

    """
    # Use plt.subplot2grid to create a grid of images
    # ====== YOUR CODE: ======
    Nc = img.shape[0]
    num = int(Nc / 4)
    plt.figure(figsize=(10, 10))
    i = 0
    for col in np.arange(num):
        for row in np.arange(num):
            temp_img = img[i, :, :]
            plt.subplot2grid(shape=(num, num), loc=(row, col))
            plt.imshow(temp_img, cmap=cmap)
            plt.axis('off')
            i += 1
    plt.suptitle(title, size=16)
    plt.show()
    # ========================


def ifft2(raw: np.ndarray):
    """
    Apply inverse FFT on the raw k-space.

    :param raw: Raw data (k-space)
    :return img: Image domain
    """
    # Use 1D np.fft functions, first apply 1D ifft on the rows and then on the columns.
    # Don't forget to shift the zero-frequency component to the center of the spectrum.

    # ====== YOUR CODE: ======
    #     img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(raw, -2), axis=-2), -2)
    #     img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(img, -1), axis=-1), -1)
    ## or fft2
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(raw), axes=(-2, -1)))
    # ========================
    return img


def sensitivity_map(img: np.ndarray, img_combined: np.ndarray):
    '''
    Create sensitivity map of the coils

    :params img: Multi-coil images (Nc,Ny,Nx)
    :params img_combined: The combined images of all the coils (Ny,Nx)
    :params thresh: Threshold
    :return S_2: The sensitivity map of the coils (Nc,Ny,Nx)

    '''
    # Use the function signal.convolve2d for applying a mean filter on the image
    # ====== YOUR CODE: ======
    Nc = img.shape[0]
    S_0 = img / (img_combined)

    # Smooth map
    kernel = np.ones((3, 3)) / 9
    S_1 = np.zeros(img.shape, dtype=complex)
    for coil in np.arange(Nc):
        S_1[coil, :, :] = signal.convolve2d(S_0[coil, :, :], kernel, mode='same')

    # Mask the sensitivities to retain only the well-defined regions
    thresh = 0.12 * np.max(np.abs(img_combined))
    mask = np.abs(img_combined) > thresh
    S_2 = S_1 * mask

    # ========================
    return S_2


def subsample(raw: np.ndarray, R: int):
    """
    Sample the k-space with a uniform cartesian pattern; each R row.

    :param raw: The k-space we want to sample, shape (Nc,Ny,Nx)
    :param R: Acceleration factor, sample each R row
    :return sampled_kspace: The sampled k-space
    """
    # ====== YOUR CODE: ======
    mask = np.zeros(raw.shape)
    mask[:, ::R, :] = 1
    sampled_kspace = raw * mask
    # ========================
    return sampled_kspace


def SENSE(undersampled_img: np.ndarray, S_mat: np.ndarray, R: int):
    '''
    Reconstructs the undersampled image.

    :params undersampled_img: The undersampled image to reconstruct (Nc,Ny,Nx)
    :params S_mat: The sensitivity matrix of the coils (Nc,Ny,Nx)
    :params R: Acceleration factor, sample each R row
    :return recon_SENSE: The reconstructed image
    '''
    # Iterate over the pixels of the image, and reconstruct the pixels' values in
    # the least-squares sense, according to equation (4).
    # Use the fucntion np.linalg.pinv for psaudo inverse
    # ====== YOUR CODE: ======
    _, Ny, Nx, = undersampled_img.shape
    recon_SENSE = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in np.arange(0, int(Ny / R)):
        y_samp = np.arange(y, Ny, Ny / R, dtype='int')
        for x in np.arange(0, Nx):
            S_R = S_mat[:, y_samp, x]
            recon_SENSE[y_samp, x] = np.matmul(np.linalg.pinv(S_R), undersampled_img[:, y, x])
    # ========================
    return recon_SENSE


# load data from fastmri dataset:
path = r"C:\Users\perez\Desktop\masters\mri_research\code\python\mrf-masters\raw_data_2_mri.npy"
raw = np.load(path)
[Nz, Nc, Ny, Nx] = np.shape(raw)
print("Ny =", Ny, "\nNx =", Nx, "\nNz =", Nz, "\nNc =", Nc)

img = ifft2(raw)
num_slice = 0
img_slice = img[num_slice]
print(img_slice.shape)
show_grid(np.abs(img_slice), cmap='gray', title='Coils map')

from scipy.stats import gmean

img_combined_avg = np.mean(img_slice, 0)
img_combined_geom = gmean(img_slice, 0)
# Plot images
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(np.abs(img_combined_geom), cmap='gray'), plt.title('Geom norm'), plt.axis('off')
plt.show()

S_2_geom = sensitivity_map(img_slice, img_combined_geom)
show_grid(np.abs(S_2_geom), cmap='jet', title='Sensitivity map-geom')

R = 2
raw_sampled_R2 = subsample(raw[num_slice, ...], R)
img_R2 = ifft2(raw_sampled_R2)
recon_SENSE_geom_2 = SENSE(img_R2, S_2_geom, R)

show_grid(np.abs(img_R2), cmap='gray', title='Coils map')
plt.figure(figsize=(10, 10))
plt.subplot(122), plt.imshow(np.abs(recon_SENSE_geom_2), cmap='gray'), plt.title(
    f'SENSE Recon geomm - R={R}'), plt.axis('off')
plt.show()

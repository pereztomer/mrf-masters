import numpy as np
from scipy import signal
def extract_sensitivity_map(img: np.ndarray, img_combined: np.ndarray):
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
    S_0 = img/(img_combined)
    # Smooth map
    kernel = np.ones((3,3))/9
    S_1 = np.zeros(img.shape, dtype=complex)
    for coil in np.arange(Nc):
        S_1[coil, :, :] = signal.convolve2d(S_0[coil, :, :], kernel, mode='same')
    # Mask the sensitivities to retain only the well-defined regions
    thresh = 0.12*np.max(np.abs(img_combined))
    mask = np.abs(img_combined) > thresh
    S_2 = S_1 * mask
    # ========================
    return S_2
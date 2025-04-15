import numpy as np

def SENSE(undersampled_img: np.ndarray, S_mat: np.ndarray, R: int):
    '''
    Reconstructs the undersampled image.
    :params undersampled_img: The undersampled image to reconstruct (Nc,Ny,Nx)
    :params S_mat: The sensitivity matrix of the coils (Nc,Ny,Nx)
    :params R: Acceleration factor, sample each R rows
    :return recon_SENSE: The reconstructed image
    '''
    # Iterate over the pixels of the image, and reconstruct the pixels' values in
    # the least-squares sense, according to equation (4).
    # Use the fucntion np.linalg.pinv for psaudo inverse
    # ====== YOUR CODE: ======
    _, Ny, Nx, = undersampled_img.shape
    recon_SENSE = np.zeros((Ny, Nx), dtype=np.complex128)
    for y in np.arange(0, int(Ny/R)):
        y_samp = np.arange(y, Ny, Ny/R, dtype='int')
        for x in np.arange(0, Nx):
            S_R = S_mat[:,y_samp, x]
            recon_SENSE[y_samp, x] = np.matmul(np.linalg.pinv(S_R), undersampled_img[:,y, x])
    # ========================
    return recon_SENSE
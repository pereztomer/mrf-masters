import pydicom
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Function to load DICOM images and extract necessary metadata
def load_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path)
    tr = dicom_data.RepetitionTime
    te = dicom_data.EchoTime
    flip_angle = dicom_data.FlipAngle
    pixel_array = dicom_data.pixel_array
    return tr, te, flip_angle, pixel_array

# Signal model for SPGR sequence
def spgr_signal(flip_angle, T1, M0, TR):
    flip_angle_rad = np.deg2rad(flip_angle)  # Convert flip angle to radians
    return M0 * np.sin(flip_angle_rad) * (1 - np.exp(-TR/T1)) / (1 - np.cos(flip_angle_rad) * np.exp(-TR/T1))

# Function to fit T1 for each voxel
def fit_t1_map(image1, image2, flip_angle1, flip_angle2, tr):
    t1_map = np.zeros_like(image1, dtype=np.float32)
    m0_map = np.zeros_like(image1, dtype=np.float32)

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            si = np.array([image1[i, j], image2[i, j]])
            # if si[0] > 0 and si[1] > 0:
            #     print("here")
            #     print(si)
            angles = np.array([flip_angle1, flip_angle2])
            try:
                popt, _ = curve_fit(lambda x, T1, M0: spgr_signal(x, T1, M0, tr), angles, si, p0=[1000, 1])
                t1_map[i, j] = popt[0]
                m0_map[i, j] = popt[1]
            except:
                t1_map[i, j] = 0
                m0_map[i, j] = 0

    return t1_map, m0_map

def main():
    # Replace 'path_to_dicom_file.dcm' with the actual path to your DICOM file
    dicom_path1 = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00054.dcm"
    dicom_path2 = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\OutPhase\IMG-0004-00053.dcm"

    tr1, te1, flip_angle1, image1 = load_dicom(dicom_path1)
    tr2, te2, flip_angle2, image2 = load_dicom(dicom_path2)

    # Check that TR is the same for both images, as required by the model
    if tr1 != tr2:
        raise ValueError("TR values are not the same for both images, cannot proceed.")

    # Fit T1 map
    t1_map, m0_map = fit_t1_map(image1, image2, flip_angle1, flip_angle2, tr1)

    # Plot the T1 map
    plt.imshow(t1_map, cmap='hot')
    plt.colorbar(label='T1 (ms)')
    plt.title('Estimated T1 Map')
    plt.show()

    plt.imshow(m0_map, cmap='hot')
    plt.colorbar(label='M0 (ms)')
    plt.title('Estimated M0 Map')
    plt.show()


if __name__ == "__main__":
    main()

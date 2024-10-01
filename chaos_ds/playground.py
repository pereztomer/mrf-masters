import pydicom
import matplotlib.pyplot as plt
import numpy as np


def read_and_display_dicoms(dicom_path1, dicom_path2):
    # Load the first DICOM image
    dicom_data1 = pydicom.dcmread(dicom_path1)
    dicom_image1 = dicom_data1.pixel_array
    # dicom_data1.EchoTime, dicom_data1.FlipAngle, dicom_data1.RepetitionTime
    # Load the second DICOM image
    dicom_data2 = pydicom.dcmread(dicom_path2)
    dicom_image2 = dicom_data2.pixel_array

    # Compute the difference image
    diff_image = np.abs(dicom_image1 - dicom_image2)

    # Display the images side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(dicom_image1, cmap='gray')
    ax[0].set_title('DICOM Image 1')
    ax[0].axis('off')

    ax[1].imshow(dicom_image2, cmap='gray')
    ax[1].set_title('DICOM Image 2')
    ax[1].axis('off')

    ax[2].imshow(diff_image, cmap='gray')
    ax[2].set_title('Difference Image')
    ax[2].axis('off')

    plt.show()

# Replace 'path_to_dicom_file.dcm' with the actual path to your DICOM file
dicom_in_phase = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00054.dcm"
docon_out_phase = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\OutPhase\IMG-0004-00053.dcm"
read_and_display_dicoms(dicom_in_phase, docon_out_phase)
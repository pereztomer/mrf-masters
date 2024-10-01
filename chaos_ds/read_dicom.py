import pydicom
import matplotlib.pyplot as plt


def open_dicom_file(file_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(file_path)

    # Display some metadata
    print("Patient's Name:", dicom_data.PatientName)
    print("Patient ID:", dicom_data.PatientID)
    print("Modality:", dicom_data.Modality)
    print("Study Date:", dicom_data.StudyDate)

    # Extract and display the pixel data
    pixel_array = dicom_data.pixel_array
    plt.imshow(pixel_array, cmap='gray')
    plt.title("DICOM Image")
    plt.show()


# Replace 'path_to_dicom_file.dcm' with the actual path to your DICOM file
dicom_file_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00002.dcm"
open_dicom_file(dicom_file_path)

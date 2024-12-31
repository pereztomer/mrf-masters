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

def plot_mri_image_and_segmentation_mask(dicom_path, seg_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Display some metadata
    print("Patient's Name:", dicom_data.PatientName)
    print("Patient ID:", dicom_data.PatientID)
    print("Modality:", dicom_data.Modality)
    print("Study Date:", dicom_data.StudyDate)

    pixel_array = dicom_data.pixel_array
    seg_mask = plt.imread(seg_path)

    # Create a figure with two subplots next to each other
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first image
    axes[0].imshow(pixel_array, cmap='gray')  # You can replace 'gray' with any desired colormap
    axes[0].set_title('DICOM Image')
    axes[0].axis('off')  # Hide the axes if desired

    # Plot the second image
    axes[1].imshow(seg_mask)  # You can replace 'gray' with any desired colormap
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')  # Hide the axes if desired

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()

    return dicom_data, seg_mask

# Replace 'path_to_dicom_file.dcm' with the actual path to your DICOM file
dicom_file_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\DICOM_anon\InPhase\IMG-0004-00054.dcm"
segmentation_mask_path = r"C:\Users\perez\Desktop\masters\mri_research\datasets\Chaos dataset\CHAOS_Train_Sets\Train_Sets\MR\1\T1DUAL\Ground\IMG-0004-00054.png"
plot_mri_image_and_segmentation_mask(dicom_file_path, segmentation_mask_path)

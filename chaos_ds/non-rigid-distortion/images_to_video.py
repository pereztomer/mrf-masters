import os
import cv2
output_dir = r"C:\Users\perez\Desktop\masters\mri_research\datasets\distortion_dataset"
video_filename = os.path.join(output_dir, 'distorted_m0_map_video.mp4')
video_fps = 24  # Frames per second

# Get list of image files
image_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]

# sort images via number in the last place in filenmae without .png
image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
# Read the first image to get the size
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))

for image_file in image_files:
    frame = cv2.imread(image_file)
    video.write(frame)

video.release()
print(f'Video saved as {video_filename}')

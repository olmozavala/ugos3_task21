# %%
import cv2
import os
from os.path import join
# Path to the directory containing the images
images_folder = '/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs'
# images_folder = '/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/results/DUACS'
output_folder = '/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs/videos'
starts_with = 'ex_'  # It can be 'Model' or 'Satellite' or DCHL_Comparison
video_title = 'default_sep_validation'

video_name = join(output_folder, f'{video_title}.mp4')
fps = .8  # Set the frame rate

# Get the list of all image files in the directory
images = sorted([f for f in os.listdir(images_folder) if f.startswith(starts_with) and f.endswith('.png')])
print(f'Images to process: {images}')

# Read the first image to get the width and height of the video
frame = cv2.imread(os.path.join(images_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use *'X264' if you have it available

video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Iterate through all images and add them to the video
max_images = 10
for i, image in enumerate(images):
    img = cv2.imread(os.path.join(images_folder, image))

    # Print the shape of the current image (to verify size)
    if img is not None:
        print(f"Processing image: {image}, Shape: {img.shape}")
        
        # Resize the image if it's not the same size as the first image
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
            print(f"Resized image: {image} to {img.shape}")
        
        # Write the resized image to the video
        video.write(img)
    else:
        print(f"Failed to read image: {image}")

    if i == max_images:
        break

# Release the video writer object
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")

# %%

import cv2
import os

# Path to the directory containing the images
image_folder = '/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs'
starts_with = 'ssh'  # It can be 'Model' or 'Satellite' or DCHL_Comparison

video_name = f'{starts_with}.mp4'
fps = .5  # Set the frame rate

# Get the list of all image files in the directory
images = sorted([f for f in os.listdir(image_folder) if f.startswith(starts_with) and f.endswith('.png')])
print(f'Images to process: {images}')

# Read the first image to get the width and height of the video
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use *'X264' if you have it available

video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Iterate through all images and add them to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))

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

# Release the video writer object
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")

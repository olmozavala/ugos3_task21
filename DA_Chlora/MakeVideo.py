import cv2
import os

# Path to the directory containing the images
image_folder = 'imgs'
starts_with = 'Satellite' # It can be 'Model' or 'Satellite' or DCHL_Comparison

video_name = f'{starts_with}.avi'

# Get the list of all image files in the directory
images = sorted([f for f in os.listdir(image_folder) if f.startswith(starts_with) and f.endswith('.png')])
print(f'Images to process: {images}')

# Read the first image to get the width and height of the video
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # or 'XVID'
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

# Iterate through all images and add them to the video
for image in images[0:20]:
    print("Processing image: ", image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release the video writer object
cv2.destroyAllWindows()
video.release()

print(f"Video saved as {video_name}")

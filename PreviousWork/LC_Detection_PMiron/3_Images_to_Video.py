import os
from os.path import  join
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

in_folder = "/data/GOFFISH/AVISO/imgs"
files = [x for x in os.listdir(in_folder) if x.find("lc_") != -1]
files.sort()

output_folder = "./output/videos"
file_name = "LC_1993-2021.mp4"

# while os.path.exists(join(output_folder, file_name)):
#     file_name = F"n_{file_name}"

fps = 20
video_size = (1115,700)

out_video = cv2.VideoWriter(join(output_folder, file_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size, True)
frames = []
th = 10
for i, file_name in enumerate(files):
    if i % 10 == 0:
        print(F"Adding file # {i}: {file_name}")
    c_file = join(in_folder, file_name)
    im = Image.open(c_file)
    np_im = asarray(im)[:,:,:3]
    out_video.write(np_im[:,:,::-1])

out_video.release()
cv2.destroyAllWindows()
print("Done! yeah babe!")
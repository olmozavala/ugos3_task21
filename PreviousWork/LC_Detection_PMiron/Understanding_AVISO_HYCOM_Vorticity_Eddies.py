import json
import os

from project_utils.lc_utils import read_aviso_data_adtuv
import numpy as np
from datetime import datetime, timedelta
from os.path import join
from PIL import Image
import cv2
import xarray as xr
from shapely.geometry import LineString
from viz_utils.eoa_viz import EOAImageVisualizer
import sys
sys.path.append("eoas_pyutils/")
sys.path.append("hycom_utils/python/")

from hycom.io import read_hycom_fields
from hycom.info import read_field_names
from viz_utils.eoa_viz import EOAImageVisualizer
from proc_utils.comp_fields import vorticity
from proc_utils.proj import haversineForGrid, get_ccrs_bbox
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

## Compare HYCOM with AVISO
## Reading AVISO
print("Reading data...")
years = np.arange(1993,1994)
tot_years = years[-1] - years[0] + 1
input_folder = "/data/GOFFISH/AVISO"
output_folder = "/data/GOFFISH/SSH_vs_ADT"
adt, u, v, lat, lon = read_aviso_data_adtuv(years, input_folder)
print("Done!")

## Reading HYCOM
read_n_days = 360
input_folder = "/nexsan/archive/GOMu0.04_501/data/netcdf/1993"
start_date = datetime.strptime("1993-01-01","%Y-%m-%d")
print("Reading Hycom....")
for iday in range(read_n_days):
    if iday % 10 == 0:
        print(F"Reading day: {iday}")
    c_date = start_date + timedelta(days=iday)
    c_date_str = c_date.strftime('%Y-%m-%d')
    file_name = join(input_folder, F"hycom_gomu_501_{c_date.strftime('%Y%m%d')}00_t000.nc")
    try:
        ds = xr.open_dataset(file_name, decode_times=False)
        if iday == 0 :
            lat_hycom = ds.lat
            lon_hycom = ds.lon
            hycom_ssh = np.empty((0, len(lat_hycom), len(lon_hycom)))

        # ---------- If you want to plot the daily anomaly
        hycom_ssh = np.ma.concatenate((hycom_ssh, ds['surf_el']), axis=0)  # [time, lat, lon]

    except Exception as e:
        print(F"Failed for {c_date_str}: {e}")
print("Done!")

##
cbar = .4
viz_obj_hycom = EOAImageVisualizer(disp_images=True, output_folder=output_folder, lats=lat_hycom, lons=lon_hycom)
viz_obj = EOAImageVisualizer(disp_images=True, output_folder=output_folder, lats=lat, lons=lon)
# anom_aviso = adt[iday,:,:] - np.nanmean(adt[iday,:,:])
# anom_hycom = ds['surf_el'] - np.nanmean(ds['surf_el'])
# viz_obj.plot_2d_data_np(adt[iday,:,:], ['adt'], title=F'LC', file_name_prefix=F'aviso')
# viz_obj_hycom.plot_3d_data_npdict(ds, ['surf_el'], title=F'Hycom', file_name_prefix=F'hycom')
# viz_obj.plot_2d_data_np(anom_aviso, ['adt'], title=F'Anom AVISO {c_date_str}', file_name_prefix=F"{c_date.strftime('%Y%m%d')}_aviso", mincbar=-1*cbar, maxcbar=cbar)
viz_obj_hycom.plot_2d_data_np(anom_hycom, ['surf_el'], title=F'Anom hycom {c_date_str}',
                              file_name_prefix=F"{c_date.strftime('%Y%m%d')}_hycom", mincbar=-1 * cbar, maxcbar=cbar)

## Making a video comparing the anomalies
# files = os.listdir(output_folder)
# hycom_files = [x for x in files if x.find("hycom") != -1 and x.find(".png") != -1]
# aviso_files = [x for x in files if x.find("aviso") != -1 and x.find(".png") != -1]
#
# hycom_files.sort()
# aviso_files.sort()
#
# output_file = join(output_folder, "SSH_vs_ADT_Anomalies.mp4")
# print(F"Generating video file: {output_file}")
# out_video = -1
# for i, file_name in enumerate(hycom_files):
#     if i % 10 == 0:
#         print(F"Adding file # {i}: {file_name}")
#     c_file_hycom = join(output_folder, file_name)
#     c_file_aviso = join(output_folder, aviso_files[i])
#     im = Image.open(c_file_hycom)
#     im_aviso = Image.open(c_file_aviso)
#     np_im = np.asarray(im)[:, :, :3]
#     np_im_aviso = np.asarray(im_aviso)[:, :, :3]
#
#     final_img = np.zeros((688, 1170 + 1132, 3), dtype=np.uint8)
#     final_img[0:688, :1170, :] = np_im
#     final_img[0:686, 1170:, :] = np_im_aviso
#     fps = 10
#     if i == 0:
#         video_size = (final_img.shape[1], final_img.shape[0])
#         out_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size, True)
#     out_video.write(final_img[:, :, ::-1])
#
# out_video.release()
# cv2.destroyAllWindows()
# print("Done! yeah babe!")
# exit()

## --------- Show the streamplot on top of vorticity
grid = np.meshgrid(lon, lat)
grid_dist = haversineForGrid(grid) / 1000
vort = vorticity(u, v, grid_dist)

base = datetime(years[0],1,1)
date_list = [base + timedelta(days=x) for x in range(365*tot_years)]
print("Done!")

##% Plotting vorticity with currents from AVISO
output_folder = "./output/imgs/eddies"
viz_obj = EOAImageVisualizer(disp_images=False, output_folder=output_folder,
                             lats=lat, lons=lon, show_var_names=True)
iday = 0

for c_day_str in [x.strftime("%Y-%m-%d") for x in date_list]:
    print(c_day_str)
    viz_obj.__setattr__('vector_field', {'u': u[iday,:,:], 'v': v[iday,:,:], 'x': grid[0], 'y': grid[1],
                                         'density': 3, 'linewidth':1, 'color':'black'})
    viz_obj.plot_2d_data_np(np.array([adt[iday,:,:], vort[iday,:,:]]), ['adt', 'vort'],
                            title=F'LC {c_day_str}', file_name_prefix=F'lc_{c_day_str}')
    iday += 1

print("Done!")
exit()

## OZ for detecting eddies
iday = 0
# def Laplacian(src, ddepth, dst=None, ksize=None, scale=None, delta=None, borderType=None): # real signature unknown; restored from __doc__
t = adt[iday,:,:].data
t[np.isnan(t)] = 0
# def Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None): # real signature unknown; restored from __doc__
test = cv2.Sobel(t, ddepth=cv2.CV_32F, dx=0, dy=1)

## 1) Initialize 1000 uniformly distributed particles
nlat = 100
nlon = 100
N = nlat*nlon
bbox = get_ccrs_bbox(lat, lon)  #(minlon, maxlon, minlat, maxlat)
lon0, lat0= np.meshgrid(np.linspace(bbox[0],bbox[1],nlon), np.linspace(bbox[2],bbox[3],nlat))
## Plot initialization
fig, ax = plt.subplots(1, 1, figsize=(13, 13), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent(bbox)
ax.stock_img()
ax.scatter(lon0, lat0, s=1, c='r')
plt.show()


## 2) Advect each particle until they are 'close' to their original position  or > t times
# OceanParcels

## 3) Plot identified closed contours
## 4) Merge intersected ones
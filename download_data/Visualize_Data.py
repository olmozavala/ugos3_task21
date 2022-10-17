import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import join
import os
import xarray as xr
import cmocean.cm as cm
from datetime import datetime, timedelta

bm_extent = (-180, 180, -90, 90)
blue_marble = '/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble_5400x2700.jpg'
bm_im = plt.imread(blue_marble)  # BATHYMETRY BLACK LAND



## ------ AVISO ADT ------
date_fmt = "%Y-%m-%d"
input_folder = "/Net/work/ozavala/GOFFISH/AVISO"

date_str = "1993-08"
viz_file = join(input_folder,F"{date_str}.nc")
ds = xr.open_dataset(viz_file)

lats = ds.latitude.data
lons = ds.longitude.data
extent = (lons.min(), lons.max(), lats.min(), lats.max())

fig, ax = plt.subplots(1,1, figsize = (8,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.imshow(bm_im, origin='upper', extent=bm_extent, transform=ccrs.PlateCarree())
im = ax.imshow(ds.adt[0,:,:], cmap=cm.deep, origin='lower', extent=extent)
plt.colorbar(im, location='right', shrink=.6, pad=.02)
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
plt.title(f"AVISO SSH Example {date_str}-01")
plt.tight_layout()
plt.show()

## ------- SST -------------------
input_folder = "/Net/work/ozavala/GOFFISH/SST/OISST/2002"
files = os.listdir(input_folder)

viz_file = join(input_folder, files[0]) # Plotting just the first file
ds = xr.open_dataset(viz_file)

lats = ds.lat.data
lons = ds.lon.data
extent = (lons.min(), lons.max(), lats.min(), lats.max())
print(extent)

fig, ax = plt.subplots(1, 1, figsize = (8,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.imshow(bm_im, origin='upper', extent=bm_extent, transform=ccrs.PlateCarree())
im = ax.imshow(ds.analysed_sst[0,:,:], cmap=cm.thermal, origin='lower', extent=extent)
plt.colorbar(im, location='right', shrink=.4, pad=.02)
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
plt.title(f"OISST SST {files[0][0:8]}")
plt.tight_layout()
plt.show()

## --------- SSS 2015 ---------------------
input_folder = "/Net/work/ozavala/GOFFISH/SSS/SMAP_Global/2015"
files = os.listdir(input_folder)

viz_file = join(input_folder, files[0])
ds_full = xr.open_dataset(viz_file)
ds = ds_full.sel(lat=slice(17, 31), lon=slice(-98.5 + 360, -74 + 360))  # Cropping by value
# ds = ds_full

lats = ds.lat.data
lons = ds.lon.data
extent = (lons.min()-360, lons.max()-360, lats.min(), lats.max())
print(extent)

fig, ax = plt.subplots(1,1, figsize = (8,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.imshow(bm_im, origin='upper', extent=bm_extent, transform=ccrs.PlateCarree())
im = ax.imshow(ds.sss_smap[:,:], cmap=cm.haline, origin='lower', extent=extent)
plt.colorbar(im, location='right', shrink=.4, pad=.02)
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
plt.title("SMAP Salinity")
plt.tight_layout()
plt.show()

##


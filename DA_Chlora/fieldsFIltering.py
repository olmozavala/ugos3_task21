# %% Load the necessary libraries and reference files
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import cmocean as cmo
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

indir = '/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM'
files = sorted(glob(join(indir, '*.nc')))

test_file = files[0]

#Function to plot the AVISO fields
def plot_aviso(data, title, gradient=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the data
    if gradient:
        gradient = np.gradient(data[0,:,:])
        norm_gradient = np.linalg.norm(gradient, axis=0)
    # Copy the adt data to a new variable
        gradient_copy = adt.copy()
        # Replace the adt data with the norm_gradient
        gradient_copy[0,:,:] = norm_gradient

        im = ax.pcolormesh(data.longitude, data.latitude, gradient_copy[0,:,:],
                       cmap=cmo.cm.balance)
    else:
        im = ax.pcolormesh(data.longitude, data.latitude, data[0,:,:],
                       cmap=cmo.cm.balance)

    # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, xlocs=ax.get_xticks(), ylocs=ax.get_yticks())
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.08, aspect=30)
    cbar.set_label(title)

    # Set title
    plt.title(f'{title} {pd.to_datetime(data.time.data[0]).strftime("%Y-%m-%d")}', fontsize=16)

    # plt.savefig("err_sla.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %%  Visualize the fields to have an idea of what we are working with
ds = xr.open_dataset(test_file)
adt = ds['adt']
err_sla = ds['err_sla']

plot_aviso(adt, 'ADT')
plot_aviso(adt, 'ADT gradient', gradient=True)
plot_aviso(err_sla, 'Error SLA')

display(ds)

# %%

# Get the mean of non nan values for each time step with numpy
mean_sla = np.nanmean(err_sla, axis=(1,2))
mean_sla_err = np.nanmean(mean_sla)
# Scatter plot the mean ssh values by date
plt.figure(figsize=(10, 6))
plt.scatter(range(mean_sla.shape[0]), mean_sla)
# Put the date as the label in the x axis
plt.xlabel("Time (days)")
plt.ylabel("Error (m)")
# Get the axis tics from the dates (only the month and day) only plot 10 ticks
plt.xticks(range(0, mean_sla.shape[0], mean_sla.shape[0]//10), 
           [t.strftime("%m-%d-%Y") for t in pd.to_datetime(ds.time.data)[::mean_sla.shape[0]//10]], 
           rotation=45)
# Set the y_limits
plt.ylim(.005, .05)
plt.title(f"Mean of mean error of the SLA field by date: {mean_sla_err:.4f} m")
plt.show()

# %% Simulate DUACs background field
def groundto2background(data, lat=(14.18613, 30.66479), lon=(-89.33899, -79.78333), x=715, y=652, resolution=0.25):
    downsampled_lats = np.arange(lat[0], lat[1], resolution)
    downsampled_lons = np.arange(lon[0], lon[1], resolution)
    upsampled_lats = np.linspace(lat[0], lat[1], y)
    upsampled_lons = np.linspace(lon[0], lon[1], x)

    ds = xr.Dataset({'ssh': (['latitude', 'longitude'], data)},
                    coords={'latitude': ('latitude', upsampled_lats),
                            'longitude': ('longitude', upsampled_lons)})
    ds = ds.interp(
        latitude=downsampled_lats, 
        longitude=downsampled_lons, 
        method='linear').interp(
            latitude=upsampled_lats, 
            longitude=upsampled_lons, 
            method='linear')
    ## Make 0 where the data is nan
    ds.ssh.data = np.where(np.isnan(ds.ssh.data), 0, ds.ssh.data)
    return ds.ssh.data
# %%
victim_file = '/Net/work/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/example_0001.nc'
ds = xr.open_mfdataset(victim_file, backend_kwargs={'format': 'netcdf4'})
display(ds)
lat_bounds = (ds.latitude.data[0], ds.latitude.data[-3])
lon_bounds = (ds.longitude.data[0], ds.longitude.data[-4])
y, x = ds.ssh.shape
print(y, x)
print(lat_bounds, lon_bounds)

background_field = groundto2background(ds.ssh.data, lat_bounds, lon_bounds, x, y)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pcolormesh(ds.ssh.data, cmap=cmo.cm.balance)
ax[1].pcolormesh(background_field, cmap=cmo.cm.balance)
plt.show()

# %%

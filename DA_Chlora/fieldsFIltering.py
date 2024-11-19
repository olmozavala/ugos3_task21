# %% Load the necessary libraries and reference files
import xarray as xr
import pandas as pd
import numpy as np
import scipy as sp
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import cmocean as cmo

indir = '/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM'
files = sorted(glob(join(indir, '*.nc')))

test_file = files[0]

# %% Function to plot the AVISO fields

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

def groundto2background(data, lat, lon, resolution=0.25):
    upsampled_lats = np.arange(lat.min(), lat.max(), resolution)
    upsampled_lons = np.arange(lon.min(), lon.max(), resolution)
    
    return upsampled_lats, upsampled_lons

# %%
import data_loader.data_loaders as module_data
import json

with open('config.json', 'r') as f:
    config = json.load(f)

logger = config.get_logger('test')

# setup data_loader instances
def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers'],
        previous_days=config['data_loader']['args']['previous_days'],
        dataset_type=config['data_loader']['args']['dataset_type'],
        demo=True
    )

    # Read the lats and lons from 
    lats = data_loader.dataset.lats
    lons = data_loader.dataset.lons
    print(lats.shape, lons.shape)

# %%
main(config)

# %%

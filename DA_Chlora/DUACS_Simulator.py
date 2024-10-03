# %% This script simulates DUACS interpolation 
import pickle
from os.path import join
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo
from data_loader.loader_utils import general_plot
# from pykrige.ok3d import OrdinaryKriging3D

# Error estimation from the AVISO files, averages per year. 
folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM"
# Read all the netcdf files in the folder and concatenate them
ds = xr.open_mfdataset(join(folder, "202*.nc"))
# ds = xr.open_dataset(join(folder, "2022-04.nc"))  # Test with a single file
ds = ds.sel(longitude=slice(-98.5, -79), latitude=slice(14, 32))
# I'm only interested in the err_sla variable
err_sla = ds["err_sla"]
adt = ds["adt"]
# Crop to the GoM region
# %% Plot one of the files to check
def plot_aviso(data, title, gradient=False):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Plot the data
    if gradient:
        gradient = np.gradient(data[0,:,:])
        norm_gradient = np.linalg.norm(gradient, axis=0)
    # Copy the adt data to a new variable
        gradient_copy = adt.copy()
        # Replace the adt data with the norm_gradient
        gradient_copy[0,:,:] = norm_gradient

        im = ax.pcolormesh(data.longitude, data.latitude, gradient_copy[0,:,:], 
                       transform=ccrs.PlateCarree(),
                       cmap=cmo.cm.balance)
    else:
        im = ax.pcolormesh(data.longitude, data.latitude, data[0,:,:], 
                       transform=ccrs.PlateCarree(),
                       cmap=cmo.cm.balance)

    # Add coastlines and borders
    ax.add_feature(cfeature.COASTLINE)

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

    # Adjust the extent to focus on the Gulf of Mexico
    ax.set_extent([-98.5, -79, 14, 32], crs=ccrs.PlateCarree())

    # plt.savefig("err_sla.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# plot_aviso(err_sla, "Error SLA")
plot_aviso(adt, "ADT")
plot_aviso(adt, "Gradient of ADT", gradient=True)
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
plt.ylim(.005, .030)
plt.title(f"Mean of mean error of the SLA field by date: {mean_sla_err:.4f} m")
plt.show()


# %% Function used to plot the interpolated DUACS data and the original HR data with the difference
def plot_duacs_interpolation(hr_ssh, duacs_ssh, lats, lons, output_file):
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), subplot_kw={'projection': proj})

    vmin, vmax = -.5, .5

    fraction = 0.038
    pad = 0.04

    for j, (data, title) in enumerate([(hr_ssh['ssh'], 'True HR SSH'), 
                                        (duacs_ssh['ssh'], 'Simulated DUACS'), 
                                        (duacs_ssh['ssh'] - hr_ssh['ssh'], 'Difference')]): 
        if j == 2:
            vmin = -.05
            vmax = .05
        else:
            vmin = -0.6
            vmax = 0.6
        im = general_plot(fig, axs[0, j], data.data[0,:,:], lats, lons, title, 
                          vmin=vmin, vmax=vmax, fraction=fraction, pad=pad)

    grad_true = np.gradient(hr_ssh['ssh'].data[0,:,:])
    grad_true = np.sqrt(grad_true[0]**2 + grad_true[1]**2)
    vmax = 0.05

    # Plot the gradient of the true SSH
    general_plot(fig, axs[1, 0], grad_true, lats, lons, "Gradient of True SSH", 
                 vmin=0, vmax=vmax, fraction=fraction, pad=pad)
    # Plot the gradient of the predicted SSH
    grad_pred = np.gradient(duacs_ssh['ssh'].data[0,:,:])
    grad_pred = np.sqrt(grad_pred[0]**2 + grad_pred[1]**2)
    general_plot(fig, axs[1, 1], grad_pred, lats, lons, "Gradient of Simulated DUACS", 
                 vmin=0, vmax=vmax, fraction=fraction, pad=pad)

    # Set the final figure to axis off
    axs[1, 2].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
# %% Function that simulates DUACS interpolation using simple linear interpolation
def simple_duacs_interpolation(Y, lats, lons, make_plots=True):

    duacs_lats = np.arange(lats.min(), lats.max(), 0.25)
    duacs_lons = np.arange(lons.min(), lons.max(), 0.25)

    rmse_list = []
    for i in range(Y.shape[0]):
        print(f"Simulating DUACS interpolation for example {i+1}/{Y.shape[0]}")
        # Extract the input variables for the current example
        hr_ssh = Y[i,:,:].reshape(1, Y.shape[1], Y.shape[2])
        # Generate an xr.Dataset with the hr_ssh using lats and lons
        # and a simulated time dimension
        simulated_time = [0]
        hr_ssh_ds = xr.Dataset({
            "ssh": (["time", "latitude", "longitude"], hr_ssh),
        }, coords={
            "time": simulated_time,
            "latitude": lats,
            "longitude": lons
        })

        hr_ssh_ds["time"] = simulated_time

        # Simulate DUACS interpolation
        ssh_duacs = hr_ssh_ds.interp(latitude=duacs_lats, longitude=duacs_lons, method='linear')
        # Interpolate again using xarray to the original HR grid
        ssh_hr_duacs = ssh_duacs.interp(latitude=lats, longitude=lons, method='linear')

        if make_plots and i % 10 == 0:
            output_file = join(output_folder, f"duacs_interpolation_{i:03d}.png")
            plot_duacs_interpolation(hr_ssh_ds, ssh_hr_duacs, lats, lons, output_file)

        # Calculate the RMSE
        rmse = np.sqrt(np.mean((hr_ssh_ds['ssh'] - ssh_hr_duacs['ssh'])**2))
        rmse_list.append(rmse.item())

    return rmse_list

# %% ========== Main code ==========
data_dir = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data"
output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/results/DUACS"
# pkl_file = "training_full.pkl"
pkl_file = "validation.pkl"
scalers_file = "scalers.pkl"

training_pkl_path = join(data_dir, pkl_file)
print(f"Reading {pkl_file} file and {scalers_file} file...")
with open(training_pkl_path, "rb") as f:
    X, Y, lats, lons = pickle.load(f)

# Read the scalers
with open(join(data_dir, scalers_file), "rb") as f:
    scalers = pickle.load(f)
print("Done!")

# Rescale the ssh data
mean_ssh = scalers["ssh"]["mean"]
std_ssh = scalers["ssh"]["std"]
Y = Y * std_ssh + mean_ssh

make_plots = True
rmse_list = simple_duacs_interpolation(Y, lats, lons, make_plots=True)

# %%
mean_rmse = np.mean(rmse_list)
title = f"Mean RMSE: {mean_rmse:.4f} m"
vmin = 0.005
vmax = 0.030

plt.figure(figsize=(10, 6))
plt.scatter(range(len(rmse_list)), rmse_list)
# Set the y_limits
plt.ylim(vmin, vmax)
plt.xlabel("Examples from validation set")
plt.ylabel("RMSE (m)")
plt.title(title)
plt.savefig(join(output_folder, "rmse_list.png"), dpi=300, bbox_inches='tight')
plt.close()
# Save the RMSE as a csv file
np.savetxt(join(output_folder, "rmse_list.csv"), rmse_list, delimiter=",")

# %% ==================== Kriging DUACS interpolation simulation with PyKrige ====================
# For each example in the validation set, simulate DUACS interpolation
# rmse_list = []
# make_plots = True
# prev_days = 7

# # Define the index of the ssh_track in the X array
# ssh_track_idx = 2
# swot_idx = 3
# target_time = prev_days + 1
# time_scale = 24

# # for i in range(prev_days, Y.shape[0]):
# i = prev_days
# print(f"Simulating DUACS interpolation for example {i+1}/{Y.shape[0]}")
# hr_tracks = np.zeros((prev_days, Y.shape[1], Y.shape[2]))
# hr_swot = np.zeros((prev_days, Y.shape[1], Y.shape[2]))

# # Extract the input variables for the current example
# hr_tracks[:,:,:] = X[i-prev_days:i,ssh_track_idx,:,:]
# hr_swot[:,:,:] = X[i-prev_days:i,swot_idx,:,:]

# # Transform non-nan values in hr_tracks to a time series of lats, lons, ssh_track values
# lats_1d = []
# lons_1d = []
# ssh_data_1d = []
# time_1d = []

# for t in range(prev_days):
#     # Tracks
#     mask = ~np.isnan(hr_tracks[t])
#     lats_mask, lons_mask = np.meshgrid(lats, lons, indexing='ij')
#     lats_1d.extend(lats_mask[mask])
#     lons_1d.extend(lons_mask[mask])
#     ssh_data_1d.extend(hr_tracks[t][mask])
#     time_1d.extend([t] * np.sum(mask))
#     # Swot
#     mask = ~np.isnan(hr_swot[t])
#     lats_mask, lons_mask = np.meshgrid(lats, lons, indexing='ij')
#     lats_1d.extend(lats_mask[mask])
#     lons_1d.extend(lons_mask[mask])
#     ssh_data_1d.extend(hr_swot[t][mask])
#     time_1d.extend([t] * np.sum(mask))

# lats_1d = np.array(lats_1d)
# lons_1d = np.array(lons_1d)
# ssh_data_1d = np.array(ssh_data_1d)
# time_1d = np.array(time_1d)

# # Print some information about the transformed data
# print(f"Number of non-nan values: {len(ssh_data_1d)}")
# print(f"Latitude range: {np.min(lats_1d):.2f} to {np.max(lats_1d):.2f}")
# print(f"Longitude range: {np.min(lons_1d):.2f} to {np.max(lons_1d):.2f}")
# print(f"Time range: {np.min(time_1d)} to {np.max(time_1d)}")

# %% Initialize Ordinary Kriging 3D
# print("Initializing Ordinary Kriging 3D...")
# OK3D = OrdinaryKriging3D(
#     lons_1d, lats_1d, time_1d, ssh_data_1d,
#     variogram_model='exponential',  # Choose appropriate model
#     verbose=False,
#     enable_plotting=False
# )
# print("Done!")

# # Create grid
# gridz = np.array([target_time * time_scale])  # Single time slice

# # Perform kriging
# print("Performing kriging...")
# k3d, ss3d = OK3D.execute('grid', lons, lats, gridz)
# print("Done!")

# # Interpolate the DUACS data to the HR grid
# duacs_ssh = xr.Dataset({
#     "ssh": (["latitude", "longitude"], k3d)
# }, coords={
#     "latitude": lats,
#     "longitude": lons
# })

# # Plot the DUACS data
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(duacs_ssh['ssh'], cmap=cmo.cm.balance)
# plt.colorbar()
# plt.show()
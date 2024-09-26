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
from pykrige.ok3d import OrdinaryKriging3D

# %% Function used to plot the interpolated DUACS data and the original HR data with the difference
def plot_duacs_interpolation(hr_ssh, duacs_ssh, lats, lons, output_file):
    num_days = hr_ssh['ssh'].shape[0]
    fig, axs = plt.subplots(num_days, 3, figsize=(18, 6*num_days), subplot_kw={'projection': ccrs.PlateCarree()})

    vmin, vmax = -.5, .5
    diff_vmin, diff_vmax = -.1, .1

    for i in range(num_days):
        for j, (data, title) in enumerate([(hr_ssh['ssh'], 'HR SSH'), 
                                           (duacs_ssh['ssh'], 'DUACS SSH'), 
                                           (duacs_ssh['ssh'] - hr_ssh['ssh'], 'Difference')]):

            ax = axs[i, j] if num_days > 1 else axs[j]
            
            im = ax.pcolormesh(lons, lats, data.isel(time=i), 
                               transform=ccrs.PlateCarree(),
                               cmap=cmo.cm.balance, 
                               vmin=vmin if j < 2 else diff_vmin, 
                               vmax=vmax if j < 2 else diff_vmax)
            
            ax.set_title(title)
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)
            ax.gridlines(draw_labels=True)
            
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
# %% Function that simulates DUACS interpolation using simple linear interpolation
def simple_duacs_interpolation(Y, lats, lons, make_plots=True):

    duacs_lats = np.arange(lats.min(), lats.max(), 0.25)
    duacs_lons = np.arange(lons.min(), lons.max(), 0.25)
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
        rmse_list.append(rmse)

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

# rmse_list = simple_duacs_interpolation(Y, lats, lons, make_plots=True)
rmse_list = []
make_plots = True

plt.figure(figsize=(10, 6))
plt.scatter(range(len(rmse_list)), rmse_list)
plt.xlabel("Example")
plt.ylabel("RMSE (m)")
plt.savefig(join(output_folder, "rmse_list.png"), dpi=300, bbox_inches='tight')
plt.close()
# Save the RMSE as a csv file
np.savetxt(join(output_folder, "rmse_list.csv"), rmse_list, delimiter=",")

# %% ==================== Kriging DUACS interpolation simulation with PyKrige ====================
# For each example in the validation set, simulate DUACS interpolation
rmse_list = []
make_plots = True
prev_days = 7

# Define the index of the ssh_track in the X array
ssh_track_idx = 2
swot_idx = 3
target_time = prev_days + 1
time_scale = 24

# for i in range(prev_days, Y.shape[0]):
i = prev_days
print(f"Simulating DUACS interpolation for example {i+1}/{Y.shape[0]}")
hr_tracks = np.zeros((prev_days, Y.shape[1], Y.shape[2]))
hr_swot = np.zeros((prev_days, Y.shape[1], Y.shape[2]))

# Extract the input variables for the current example
hr_tracks[:,:,:] = X[i-prev_days:i,ssh_track_idx,:,:]
hr_swot[:,:,:] = X[i-prev_days:i,swot_idx,:,:]

# Transform non-nan values in hr_tracks to a time series of lats, lons, ssh_track values
lats_1d = []
lons_1d = []
ssh_data_1d = []
time_1d = []

for t in range(prev_days):
    # Tracks
    mask = ~np.isnan(hr_tracks[t])
    lats_mask, lons_mask = np.meshgrid(lats, lons, indexing='ij')
    lats_1d.extend(lats_mask[mask])
    lons_1d.extend(lons_mask[mask])
    ssh_data_1d.extend(hr_tracks[t][mask])
    time_1d.extend([t] * np.sum(mask))
    # Swot
    mask = ~np.isnan(hr_swot[t])
    lats_mask, lons_mask = np.meshgrid(lats, lons, indexing='ij')
    lats_1d.extend(lats_mask[mask])
    lons_1d.extend(lons_mask[mask])
    ssh_data_1d.extend(hr_swot[t][mask])
    time_1d.extend([t] * np.sum(mask))

lats_1d = np.array(lats_1d)
lons_1d = np.array(lons_1d)
ssh_data_1d = np.array(ssh_data_1d)
time_1d = np.array(time_1d)

# Print some information about the transformed data
print(f"Number of non-nan values: {len(ssh_data_1d)}")
print(f"Latitude range: {np.min(lats_1d):.2f} to {np.max(lats_1d):.2f}")
print(f"Longitude range: {np.min(lons_1d):.2f} to {np.max(lons_1d):.2f}")
print(f"Time range: {np.min(time_1d)} to {np.max(time_1d)}")

# %% Initialize Ordinary Kriging 3D
print("Initializing Ordinary Kriging 3D...")
OK3D = OrdinaryKriging3D(
    lons_1d, lats_1d, time_1d, ssh_data_1d,
    variogram_model='exponential',  # Choose appropriate model
    verbose=False,
    enable_plotting=False
)
print("Done!")

# Create grid
gridz = np.array([target_time * time_scale])  # Single time slice

# Perform kriging
print("Performing kriging...")
k3d, ss3d = OK3D.execute('grid', lons, lats, gridz)
print("Done!")

# Interpolate the DUACS data to the HR grid
duacs_ssh = xr.Dataset({
    "ssh": (["latitude", "longitude"], k3d)
}, coords={
    "latitude": lats,
    "longitude": lons
})

# Plot the DUACS data
plt.figure(figsize=(10, 6))
plt.pcolormesh(duacs_ssh['ssh'], cmap=cmo.cm.balance)
plt.colorbar()
plt.show()
# %% In this code we generate examples that simulate altimeter tracks, SWOT, SST and Chl-a with clouds data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
from os.path import join
import os
import sys
from preproc.preproc_tools import get_mask
from datetime import date
from scipy.spatial import cKDTree
import pandas as pd
import cmocean
from scipy.ndimage import gaussian_filter, convolve
import time

sys.path.append("eoas_pyutils")
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import BackgroundType

from io_utils.coaps_io_data import get_biorun_cicese_nemo_by_date_range
import re

# %% Paths
data_folder = "/unity/f1/ozavala/DATA/GOFFISH/"
sst_path = join(data_folder, "SST/ODYSSEA_SST_L3_P1D_1982/")
chl_path = join(data_folder, "CHLORA/COPERNICUS_GOM_L3_2016_OLCI_4km/")
ssh_path = "/unity/g1/abozec_NEW/TSIS/data/aviso/GOMb0.01/"
swot_path = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/Masks/"
# output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/"
output_folder = "/tmp/OZ/"

num_training_examples = 1758*2# How many random examples to generate
generate_sample_images = False
display_images = True
ex_to_plot = 1 # How many examples to plot

# Define lat and lon coords
# TODO replace this file

imgs_output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs/"

# %% ================= Reading model data ==========================
print("Reading model data...")
start_date = date(2016, 8, 1)
# end_date = date(2021, 7, 1)
end_date = date(2016, 9, 10)
model_data, lats, lons = get_biorun_cicese_nemo_by_date_range(start_date, end_date)
tot_ex_model = model_data.time.shape[0]
print("Total number of model examples: ", tot_ex_model)
mask = model_data['dchl'].values == 0
model_data['temperature'].values[mask] = np.nan
model_data['ssh'].values[mask] = np.nan
model_data['dchl'].values[mask] = np.nan
times = model_data.time.values
# Times to nice string yyyy-mm-dd
times_str = [str(time)[:10] for time in times]
# Print the BBOX
print(f"BBOX: {lats.min():.2f}, {lats.max():.2f}, {lons.min():.2f}, {lons.max():.2f}")

# %% Initialize visualizations with model lats and lons
viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=3, 
                              fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                              show_var_names=True, land=True, background=BackgroundType.WHITE)

if generate_sample_images:
    for i in range(ex_to_plot):
    # for i in range(1):
        print(f"Generating sample image {i}...")
        viz_obj.plot_3d_data_npdict(model_data, var_names=["temperature", "ssh","dchl"], 
                                    z_levels=[i], title=times_str[i], file_name_prefix=f"model_data_{i}", show_color_bar=True,
                                    mincbar=[28, -.4, None], maxcbar=[32, .8, None],
                                    norm=[ None, None,  LogNorm(.001, 5)])

print("Done!")

# %% ================= Getting SWOT masks ==========================
# Read all the files
print("Generating SWOT masks...")
data = xr.open_mfdataset(swot_path + "*.nc", combine="by_coords")
swot_all_masks = data.mask
tot_ex_swot = swot_all_masks.date.shape[0]
print("Total number of SWOT examples: ", tot_ex_swot)

# Crop to lats and lons
swot_all_masks = swot_all_masks.interp(lat=lats, lon=lons, method="linear")
# Rename lat and lon to latitude and longitude
swot_all_masks = swot_all_masks.rename({"lat": "latitude", "lon": "longitude"})
# Rename date to time
swot_all_masks = swot_all_masks.rename({"date": "time"})

if generate_sample_images:
    for i in range(ex_to_plot):
        idx = np.random.randint(0, tot_ex_swot)
        time = swot_all_masks.date[idx].values
        time_str = str(time)[:10]
        viz_obj.plot_2d_data_np([swot_all_masks[idx, :, :]], var_names=[f"SWOT Data {time_str}"], 
                                title="", file_name_prefix=f"swot_mask_data_{i:02}", show_color_bar=True,
                                cmap=["Reds"])

# %% ================= Getting SST masks ==========================
# Read all the files
print("Generating SST masks...")
data = xr.open_mfdataset(sst_path + "*.nc", combine="by_coords")
sst_data = data.sea_surface_temperature
tot_ex_sst = sst_data.time.shape[0]
print("Total number of SST examples: ", tot_ex_sst)

# Crop to lats and lons
sst_data = sst_data.interp(latitude=lats, longitude=lons, method="linear")

sst_masks  = get_mask(sst_data)

# Change long_name to sst_mask
sst_masks.attrs["long_name"] = "sst_mask"

# Plot a single example to see it looks correct
if generate_sample_images:
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=2, 
                              fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                              show_var_names=True, land=True, background=BackgroundType.WHITE)

    for i in range(ex_to_plot):
    # for i in range(2):
        idx = np.random.randint(0, tot_ex_sst)
        time = sst_data.time[idx].values
        time_str = str(time)[:10]
        viz_obj.plot_2d_data_np([sst_data[idx,:,:], sst_masks[idx,:,:]], var_names=[f"SST Data {time_str}", f"SST Mask {time_str}"], 
                                title=f"", file_name_prefix=f"sst_mask_data_{i:02}", show_color_bar=True,
                                mincbar= [None, 0], maxcbar=[None, 1],
                                cmap=[cmocean.cm.thermal, "Reds"],)

print("Done!")

# %% ================= Getting Chlora masks ==========================
# Read all the files
print("Generating Chlora masks...")
data = xr.open_mfdataset(chl_path + "*.nc", combine="by_coords")
chlora_data = data.CHL
tot_ex_chlora = chlora_data.time.shape[0]
print("Total number of chlora examples: ", tot_ex_chlora)

# Crop to lats and lons
chlora_data = chlora_data.interp(latitude=lats, longitude=lons, method="linear")

# Obtain binary mask from data
chlora__masks = get_mask(chlora_data)

# Change long_name to chlora__mask
chlora__masks.attrs["long_name"] = "chlora_mask"

if generate_sample_images:
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=2, 
                              fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                              show_var_names=True, land=True, background=BackgroundType.WHITE)
    # Plot a single example to see it looks correct
    for i in range(ex_to_plot):
        idx = np.random.randint(0, tot_ex_chlora)
        time = chlora_data.time[idx].values
        time_str = str(time)[:10]
        viz_obj.plot_2d_data_np([chlora_data[idx,:,:], chlora__masks[idx,:,:]], var_names=[f"Chlora Data {time_str}", f"Chlora Mask {time_str}"], 
                                title="", file_name_prefix=f"chlora_mask_data_{i:02}", show_color_bar=True,
                                cmap=[cmocean.cm.algae, "Reds"],
                            mincbar= [.00001, 0], maxcbar=[3, 1], norm=[ LogNorm(0.005,20), None])

print("Done!")

# %% ================= Getting Altimeter data ==========================
# Read all the folders
output_ssh_tracks_file = "/unity/f1/ozavala/DATA/GOFFISH/CHLORA/SSH_Tracks/ssh_tracks.nc"
if os.path.exists(output_ssh_tracks_file):
    print("Reading altimeter data from file...")
    final_ssh_data_ds = xr.open_dataset(output_ssh_tracks_file)
else:
    print("Reading altimeter data...")
    all_satellites = [folder for folder in os.listdir(ssh_path) if os.path.isdir(os.path.join(ssh_path, folder))]
    all_satellites = sorted(all_satellites)
    print("Found satellites: ", all_satellites)
    # Read the following years
    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)
    years = [str(year) for year in range(start_date.year, end_date.year + 1)]
    ssh_files = []

    for satellite in all_satellites:
        for year in years:
            file_pattern = f"nrt_.*_{year}.*\.nc"
            # Find all the files that match the pattern
            files = [f for f in os.listdir(join(ssh_path, satellite)) if re.match(file_pattern, f)]
            # print(f"Found {len(files)} files for {satellite} in {year}")
            # Append the files to the list
            ssh_files += [join(ssh_path, satellite, f) for f in files]

    print("Total number of files found: ", len(ssh_files))

    grid_lats = lats
    grid_lons = lons
    grid_lon, grid_lat = np.meshgrid(lats, lons)
    time_grid = pd.date_range(start=start_date, end=end_date, freq='D')

    final_ssh_data = np.zeros((len(time_grid), len(grid_lats), len(grid_lons)))

    #  Iterate over the files and put the data in the same grid as the model data
    # TODO we need to speed this part up
    for i, file in enumerate(ssh_files):
        if i % 100 == 0:
            print(f"Processing file {file} ({i+1}/{len(ssh_files)})")

        ssh_data = xr.open_dataset(file)

        #  Get the date from the file name
        date_str = re.search(r'\d{8}', file).group()
        cur_date = pd.to_datetime(date_str, format="%Y%m%d")
        # Get number of days since start_date
        if cur_date.date() < start_date or cur_date.date() > end_date:
            continue

        days_idx = (cur_date.date() - start_date).days

        satellite_values = ssh_data['mdt'].values
        satellite_lats = ssh_data['latitude'].values
        satellite_lons = ssh_data['longitude'].values

        valid_points_mask = (satellite_lons > grid_lons.min()) & (satellite_lons < grid_lons.max()) & \
                            (satellite_lats > grid_lats.min()) & (satellite_lats < grid_lats.max())
            
        satellite_lons_valid = satellite_lons[valid_points_mask]
        satellite_lats_valid = satellite_lats[valid_points_mask]
        satellite_values_valid = satellite_values[valid_points_mask]
        # print(f"satellite_lons_valid.shape: {satellite_lons_valid.shape}")

        grid_points = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

        #  Create a KDTree for quick nearest-neighbor lookup
        tree = cKDTree(grid_points)

        # Assume satellite_lons and satellite_lats are your satellite track coordinates
        satellite_points = np.column_stack((satellite_lats, satellite_lons))
        # print(f"satellite_points.shape: {satellite_points.shape}")

        # Find the nearest spatial grid point for each satellite track point
        distances, spatial_indices = tree.query(satellite_points)
        # print(f"spatial_indices[:10] {spatial_indices[:10]}")
        # print(f"distances[:10] {distances[:10]}")

        obtained_indices = np.unravel_index(spatial_indices, grid_lon.shape)
        obtained_indices = np.array(obtained_indices)
        # obtained_indices = len(grid_lats) - obtained_indices - 1

        final_ssh_data[days_idx, obtained_indices[1], obtained_indices[0]] = 1

    # Clear the borders
    final_ssh_data[:,:,-1] = 0
    final_ssh_data[:,-1,:] = 0
    # Create a dataset with the final ssh data
    final_ssh_data_da = xr.DataArray(final_ssh_data, coords=[time_grid, grid_lats, grid_lons], dims=['time', 'latitude', 'longitude'])
    final_ssh_data_ds = final_ssh_data.to_dataset(name="ssh")
    # Save the data
    final_ssh_data_ds.to_netcdf()

tot_ex_ssh = final_ssh_data_ds['ssh'].shape[0]
print("Total number of SSH examples: ", tot_ex_ssh)
print("Done!")

# %% Plot some examples
if generate_sample_images:
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=2, 
                              fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                              show_var_names=True, land=True, background=BackgroundType.WHITE)

    sigma = 1
    for i in range(ex_to_plot):
        idx = np.random.randint(0, tot_ex_ssh)
        ssh_masked_data = final_ssh_data_ds['ssh'][idx, :, :]
        # Replace NaNs with zeros in ssh_masked_data
        ssh_masked_data = ssh_masked_data.fillna(0)
        # Smooth to be able to see the tracks
        smooth_ssh_masked_data = gaussian_filter(ssh_masked_data, sigma=sigma)
        # Make zero values NaN again for the smooth_ssh_masked_data
        smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)

        time = final_ssh_data_ds['time'][idx].values
        time_str = str(time)[:10]
        viz_obj.plot_2d_data_np([smooth_ssh_masked_data], var_names=[f"Altimeter Data {time_str}"], 
                                title="", file_name_prefix=f"ssh_mask_data_{i:02}", show_color_bar=True,
                                # cmap=[cmocean.cm.curl, "Reds"],
                                cmap=[cmocean.cm.curl])

# %% ================= Generating synthetic examples ==========================
viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=2, 
                            fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                            show_var_names=True, land=True, background=BackgroundType.WHITE)

sigma = 2# Adjust the sigma value for more or less smoothing

print("Generating synthetic examples...")
from joblib import Parallel, delayed

def generate_example(i, tot_ex_sst, tot_ex_chlora, tot_ex_model, tot_ex_ssh, tot_ex_swot):
    print(f"Generating example {i}...")
    idx_sst = np.random.randint(0, tot_ex_sst)
    idx_chlora = np.random.randint(0, tot_ex_chlora)
    # idx_model = np.random.randint(0, tot_ex_model)
    idx_model = i % tot_ex_model
    idx_ssh = np.random.randint(0, tot_ex_ssh)
    idx_swot = np.random.randint(0, tot_ex_swot)

    #  ---------------   Simulate the SST data
    sst_mask = sst_masks[idx_sst, :, :]
    sst_data = model_data['temperature'][idx_model, :, :]
    sst_data = sst_data.where(sst_mask == 1, other=np.nan)

    #  ---------------   Simulate the SSH track data
    ssh_mask = final_ssh_data_ds['ssh'][idx_ssh, :, :]
    ssh_data = model_data['ssh'][idx_model, :, :]
    ssh_masked_data = ssh_data.where(ssh_mask > 0, other=0)
    smooth_ssh_masked_data = gaussian_filter(ssh_masked_data, sigma=sigma)
    smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)

    #  ---------------   Simulate the SWOT data
    swot_mask = swot_all_masks[idx_swot, :, :]
    swot_mask = swot_mask.where(swot_mask == 1, other=np.nan)
    swot_masked_data = ssh_data.where(swot_mask == 1, other=np.nan)

    #  ---------------   Simulate the Chlora data
    chlora_mask = chlora__masks[idx_chlora, :, :]
    chlora_data = model_data['dchl'][idx_model, :, :]
    chlora_data = chlora_data.where(chlora_mask == 1, other=np.nan)

    # if generate_sample_images and i % 1 == 0:
    #     print(f"Plotting example {i}...")
    #     viz_obj.plot_2d_data_np([sst_data, chlora_data, smooth_ssh_masked_data, ssh_data, swot_masked_data], 
    #                         var_names=["SST", "Chlor-a", "Tracks", "SSH", "SWOT"], 
    #                         title=f"Example {i}", file_name_prefix=f"INPUT_data_masked_{i}", show_color_bar=True,
    #                         cmap=[cmocean.cm.thermal, cmocean.cm.algae, cmocean.cm.curl, cmocean.cm.curl, cmocean.cm.curl],
    #                         mincbar=[28, 0, None, -.4, None], maxcbar=[32, 3, None, .8, None],
    #                         norm=[None, LogNorm(vmin=1e-3, vmax=1e-1), None, None, None])


    print("Making dataset...")
    start_time = time.time()
    cur_example_ds = xr.Dataset({
        'sst': (['latitude', 'longitude'], sst_data.data),
        'chlora': (['latitude', 'longitude'], chlora_data.data),
        'ssh_track': (['latitude', 'longitude'], ssh_masked_data.data),
        'ssh': (['latitude', 'longitude'], ssh_data.data),
        'swot': (['latitude', 'longitude'], swot_masked_data.data)
    }, coords={
        'latitude': lats,
        'longitude': lons
    })
    dataset_time = time.time() - start_time
    print(f"Time to make dataset: {dataset_time:.4f} seconds")
    
    print(f"Saving netcdf for example {i}  ...")
    start_time = time.time()
    cur_example_ds.to_netcdf(join(output_folder, f"example_{i:03}.nc"))
    save_time = time.time() - start_time
    print(f"Time to save netcdf: {save_time:.4f} seconds")

print("Generating synthetic examples...")

# Use joblib to parallelize the loop
Parallel(n_jobs=10)(delayed(generate_example)(i, tot_ex_sst, tot_ex_chlora, tot_ex_model, tot_ex_ssh, tot_ex_swot) 
                for i in range(0, num_training_examples))

# %% Move everything in /tmp/OZ/ to /unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/
os.system(f"mv /tmp/OZ/*.nc /unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/")

print("Done saving all examples!")
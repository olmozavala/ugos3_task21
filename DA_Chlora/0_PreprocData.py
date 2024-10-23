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
import dask.array as da
import dask.dataframe as dd
from multiprocessing import Pool
import shutil

sys.path.append("eoas_pyutils")
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import BackgroundType

from io_utils.coaps_io_data import get_biorun_cicese_nemo_by_date_range
import re

# Set up Dask to use multiple threads
import dask
dask.config.set(scheduler='threads', num_workers=4)  # Adjust the number of workers as needed

# Rest of the code remains the same...sys.path.append("eoas_pyutils")
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
output_folder = "/tmp/OZ/"

num_training_examples = 1758*2  # How many random examples to generate
generate_sample_images = False
display_images = True
ex_to_plot = 1  # How many examples to plot

imgs_output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs/"

# %% ================= Reading model data ==========================
print("Reading model data...")
start_date = date(2016, 8, 1)
end_date = date(2016, 10, 1)
# end_date = date(2021, 7, 1)
model_data, lats, lons = get_biorun_cicese_nemo_by_date_range(start_date, end_date)
model_data = model_data.chunk({'time': 100})  # Adjust chunk size as needed

tot_ex_model = model_data.time.shape[0]
print("Total number of model examples: ", tot_ex_model)

mask = da.from_array(model_data['dchl'].values == 0, chunks=model_data['dchl'].chunks)
model_data['temperature'] = model_data['temperature'].where(~mask)
model_data['ssh'] = model_data['ssh'].where(~mask)
model_data['dchl'] = model_data['dchl'].where(~mask)

times = model_data.time.values
times_str = [str(time)[:10] for time in times]
print(f"BBOX: {lats.min():.2f}, {lats.max():.2f}, {lons.min():.2f}, {lons.max():.2f}")

# %% Initialize visualizations with model lats and lons
viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=display_images, max_imgs_per_row=3, 
                             fig_size=10, coastline=True, contourf=False, output_folder=imgs_output_folder, 
                             show_var_names=True, land=True, background=BackgroundType.WHITE)

if generate_sample_images:
    for i in range(ex_to_plot):
        print(f"Generating sample image {i}...")
        viz_obj.plot_3d_data_npdict(model_data, var_names=["temperature", "ssh","dchl"], 
                                    z_levels=[i], title=times_str[i], file_name_prefix=f"model_data_{i}", show_color_bar=True,
                                    mincbar=[28, -.4, None], maxcbar=[32, .8, None],
                                    norm=[None, None, LogNorm(.001, 5)])

print("Done!")

# %% ================= Getting SWOT masks ==========================
print("Generating SWOT masks...")
swot_file = join(swot_path, "swot_masks_2016_2024.nc")
print(f"Reading SWOT masks from {swot_file}...")
data = xr.open_mfdataset(swot_file, combine="by_coords", chunks={'date': 100})
swot_all_masks = data.mask
tot_ex_swot = swot_all_masks.date.shape[0]
print("Total number of SWOT examples: ", tot_ex_swot)

swot_all_masks = swot_all_masks.interp(lat=lats, lon=lons, method="linear")
swot_all_masks = swot_all_masks.rename({"lat": "latitude", "lon": "longitude", "date": "time"})

# %% ================= Getting SST masks ==========================
print("Generating SST masks...")
data = xr.open_mfdataset(sst_path + "*.nc", combine="by_coords", chunks={'time': 100})
sst_data = data.sea_surface_temperature
tot_ex_sst = sst_data.time.shape[0]
print("Total number of SST examples: ", tot_ex_sst)

sst_data = sst_data.interp(latitude=lats, longitude=lons, method="linear")
sst_masks = get_mask(sst_data)
sst_masks.attrs["long_name"] = "sst_mask"

# %% ================= Getting Chlora masks ==========================
print("Generating Chlora masks...")
data = xr.open_mfdataset(chl_path + "*.nc", combine="by_coords", chunks={'time': 100})
chlora_data = data.CHL
tot_ex_chlora = chlora_data.time.shape[0]
print("Total number of chlora examples: ", tot_ex_chlora)

chlora_data = chlora_data.interp(latitude=lats, longitude=lons, method="linear")
chlora__masks = get_mask(chlora_data)
chlora__masks.attrs["long_name"] = "chlora_mask"

# %% ================= Getting Altimeter data ==========================
output_ssh_tracks_file = "/unity/f1/ozavala/DATA/GOFFISH/CHLORA/SSH_Tracks/ssh_tracks.nc"
if os.path.exists(output_ssh_tracks_file):
    print("Reading altimeter data from file...")
    final_ssh_data_ds = xr.open_dataset(output_ssh_tracks_file, chunks={'time': 100})
else:
    print("Reading altimeter data...")
    all_satellites = sorted([folder for folder in os.listdir(ssh_path) if os.path.isdir(os.path.join(ssh_path, folder))])
    print("Found satellites: ", all_satellites)
    
    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)
    years = [str(year) for year in range(start_date.year, end_date.year + 1)]
    
    def process_ssh_file(file):
        ssh_data = xr.open_dataset(file)
        date_str = re.search(r'\d{8}', file).group()
        cur_date = pd.to_datetime(date_str, format="%Y%m%d")
        if cur_date.date() < start_date or cur_date.date() > end_date:
            return None
        
        days_idx = (cur_date.date() - start_date).days
        satellite_values = ssh_data['mdt'].values
        satellite_lats = ssh_data['latitude'].values
        satellite_lons = ssh_data['longitude'].values
        
        valid_points_mask = (satellite_lons > lons.min()) & (satellite_lons < lons.max()) & \
                            (satellite_lats > lats.min()) & (satellite_lats < lats.max())
        
        satellite_lons_valid = satellite_lons[valid_points_mask]
        satellite_lats_valid = satellite_lats[valid_points_mask]
        satellite_values_valid = satellite_values[valid_points_mask]
        
        grid_points = np.column_stack((lons.ravel(), lats.ravel()))
        tree = cKDTree(grid_points)
        satellite_points = np.column_stack((satellite_lats_valid, satellite_lons_valid))
        distances, spatial_indices = tree.query(satellite_points)
        
        obtained_indices = np.unravel_index(spatial_indices, lons.shape)
        final_ssh_data = np.zeros((len(lats), len(lons)))
        final_ssh_data[obtained_indices[1], obtained_indices[0]] = 1
        
        return xr.DataArray(final_ssh_data, coords=[('latitude', lats), ('longitude', lons)], 
                            dims=['latitude', 'longitude'], name='ssh')
    
    ssh_files = []
    for satellite in all_satellites:
        for year in years:
            file_pattern = f"nrt_.*_{year}.*\.nc"
            files = [join(ssh_path, satellite, f) for f in os.listdir(join(ssh_path, satellite)) if re.match(file_pattern, f)]
            ssh_files.extend(files)
    
    print("Total number of files found: ", len(ssh_files))
    
    ssh_data_list = dd.from_pandas(pd.DataFrame({'file': ssh_files}), npartitions=10)
    processed_ssh_data = ssh_data_list.apply(process_ssh_file, axis=1).compute()
    
    final_ssh_data_ds = xr.concat(processed_ssh_data.dropna(), dim='time')
    final_ssh_data_ds = final_ssh_data_ds.to_dataset(name="ssh")
    final_ssh_data_ds.to_netcdf(output_ssh_tracks_file)

tot_ex_ssh = final_ssh_data_ds['ssh'].shape[0]
print("Total number of SSH examples: ", tot_ex_ssh)
print("Done!")

# %% ================= Generating synthetic examples ==========================
# ... (previous code remains the same)

def generate_example(i, model_data, sst_masks, chlora__masks, final_ssh_data_ds, swot_all_masks):
    idx_sst = np.random.randint(0, tot_ex_sst)
    idx_chlora = np.random.randint(0, tot_ex_chlora)
    idx_model = i % tot_ex_model
    idx_ssh = np.random.randint(0, tot_ex_ssh)
    idx_swot = np.random.randint(0, tot_ex_swot)

    sst_mask = sst_masks[idx_sst, :, :]
    sst_data = model_data['temperature'][idx_model, :, :]
    sst_data = sst_data.where(sst_mask == 1, other=np.nan)

    ssh_mask = final_ssh_data_ds['ssh'][idx_ssh, :, :]
    ssh_data = model_data['ssh'][idx_model, :, :]
    ssh_masked_data = ssh_data.where(ssh_mask > 0, other=0)
    smooth_ssh_masked_data = gaussian_filter(ssh_masked_data, sigma=2)
    smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)

    swot_mask = swot_all_masks[idx_swot, :, :]
    swot_mask = swot_mask.where(swot_mask == 1, other=np.nan)
    swot_masked_data = ssh_data.where(swot_mask == 1, other=np.nan)

    chlora_mask = chlora__masks[idx_chlora, :, :]
    chlora_data = model_data['dchl'][idx_model, :, :]
    chlora_data = chlora_data.where(chlora_mask == 1, other=np.nan)

    return {
        'sst': sst_data.compute().data,
        'chlora': chlora_data.compute().data,
        'ssh_track': ssh_masked_data.compute().data,
        'ssh': ssh_data.compute().data,
        'swot': swot_masked_data.compute().data
    }

def save_example(data, i):
    cur_example_ds = xr.Dataset({
        'sst': (['latitude', 'longitude'], data['sst']),
        'chlora': (['latitude', 'longitude'], data['chlora']),
        'ssh_track': (['latitude', 'longitude'], data['ssh_track']),
        'ssh': (['latitude', 'longitude'], data['ssh']),
        'swot': (['latitude', 'longitude'], data['swot'])
    }, coords={
        'latitude': lats,
        'longitude': lons
    })
    
    cur_example_ds.to_netcdf(join(output_folder, f"example_{i:04d}.nc"))
    print(f"Example {i} generated and saved.")

print("Generating synthetic examples...")

# Pre-load data into memory if it fits
start_time = time.time()
model_data = model_data.compute()
sst_masks = sst_masks.compute()
chlora__masks = chlora__masks.compute()
final_ssh_data_ds = final_ssh_data_ds.compute()
swot_all_masks = swot_all_masks.compute()
end_time = time.time()
print(f"Data pre-loading took {end_time - start_time:.2f} seconds.")

# Generate examples
start_time = time.time()
examples = range(num_training_examples)
results = [generate_example(i, model_data, sst_masks, chlora__masks, final_ssh_data_ds, swot_all_masks) for i in examples]

# Save examples
for i, result in enumerate(results):
    save_example(result, i)

end_time = time.time()
print("Done generating examples!")
print(f"Saving examples took {end_time - start_time:.2f} seconds.")

# %% Move everything in /tmp/OZ/ to /unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/
def move_file(file):
    shutil.move(join("/tmp/OZ/", file), "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/")

files_to_move = [f for f in os.listdir("/tmp/OZ/") if f.endswith('.nc')]
with Pool(processes=10) as pool:
    pool.map(move_file, files_to_move)

print("Done moving all examples!")
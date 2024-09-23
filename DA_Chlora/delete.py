import xarray as xr
import numpy as np
import time
from os.path import join

# Synthetic data dimensions
lat_dim = 652
lon_dim = 715

# Create synthetic data (all float32 type)
sst_data = np.random.rand(lat_dim, lon_dim).astype('float32')
chlora_data = np.random.rand(lat_dim, lon_dim).astype('float32')
ssh_masked_data = np.random.rand(lat_dim, lon_dim).astype('float32')
ssh_data = np.random.rand(lat_dim, lon_dim).astype('float32')
swot_masked_data = np.random.rand(lat_dim, lon_dim).astype('float32')

# Synthetic latitude and longitude coordinates
lats = np.linspace(-90, 90, lat_dim).astype('float32')
lons = np.linspace(-180, 180, lon_dim).astype('float32')

# Create xarray Dataset
cur_example_ds = xr.Dataset(
    {
        'sst': (['latitude', 'longitude'], sst_data),
        'chlora': (['latitude', 'longitude'], chlora_data),
        'ssh_track': (['latitude', 'longitude'], ssh_masked_data),
        'ssh': (['latitude', 'longitude'], ssh_data),
        'swot': (['latitude', 'longitude'], swot_masked_data)
    },
    coords={
        'latitude': lats,
        'longitude': lons
    }
)

# Define output folder (change to your desired folder)
output_folder = "/tmp/OZ/"
# output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data"

# Measure the time taken to save the NetCDF file
i = 1
print(f"Saving NetCDF for example {i}...")
start_time = time.time()
cur_example_ds.to_netcdf(join(output_folder, f"example_{i:03}.nc"))
save_time = time.time() - start_time
print(f"Time to save NetCDF: {save_time:.4f} seconds")

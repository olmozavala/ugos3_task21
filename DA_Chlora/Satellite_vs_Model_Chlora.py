# %% 
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("eoas_pyutils")

from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import BackgroundType
from proc_utils.gom import lc_from_date, lc_from_ssh
from io_utils.coaps_io_data import get_chlora_noaa_by_date_range, get_biorun_cicese_nemo_by_date_range, get_chlora_copernicus_by_date_range

from shapely.geometry import LineString
from matplotlib.colors import LogNorm
import xarray as xr
import cmocean.cm as ccm
import os
from os.path import join
from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# What this code does is obtain a climatology for each season (Spring, Summer, Fall, Winter) and then plot the climatology for each season
# Comparing the model and the satellite data
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# %% 
year = 2019
disp_images = False

noaa_chlora_folder = "/unity/f1/ozavala/DATA/GOFFISH/CHLORA/NOAA"
cop_chlora_folder = "/unity/f1/ozavala/DATA/GOFFISH/CHLORA/COPERNICUS"
# %%
lon = [-98, -76]  
lat = [17.5, 32.5]
bbox = [lat[0], lat[1], lon[0], lon[1]]

climatology = {}

model_data, lats, lons = get_biorun_cicese_nemo_by_date_range(date(2017,1,1), date(2017,1,3), bbox=bbox)

# %%
for c_season in seasons:
    try: 
        if c_season == 'Winter':
            start_date = date(year, 1, 1)
            end_date = date(year, 3, 20)
        elif c_season == 'Spring':
            start_date = date(year, 3, 21)
            end_date = date(year, 6, 20)
        elif c_season == 'Summer':
            start_date = date(year, 6, 21)
            end_date = date(year, 9, 22)
        elif c_season == 'Fall':
            start_date = date(year, 9, 23)
            end_date = date(year, 12, 20)

        # %% -------- Read Chlor-a Copernicus data
        cop_sat, _,  _= get_chlora_copernicus_by_date_range(cop_chlora_folder, start_date, end_date, bbox=bbox)

        # %% -------- Read Chlor-a NOAA data
        noaa_sat, _,  _= get_chlora_noaa_by_date_range(noaa_chlora_folder, start_date, end_date, bbox=bbox)

        # %% -------- model_data from cicese
        model_data, lats, lons = get_biorun_cicese_nemo_by_date_range(start_date, end_date, bbox=bbox)

        # %%--------- Interpolate all the data to the lat and lon from the model
        noaa_sat =  noaa_sat.interp(latitude=lats, longitude=lons)
        cop_sat =  cop_sat.interp(latitude=lats, longitude=lons)

        # %% -------- Average the data for the season
        noaa_sat_mean = noaa_sat.mean(dim='time')
        cop_sat_mean = cop_sat.mean(dim='time')
        model_mean = model_data.mean(dim='time')
        
        # %% -------- Save the data   
        climatology[c_season] = {'noaa_sat': noaa_sat_mean, 'cop_sat':cop_sat_mean, 'model': model_mean}

        # %% -------- Plot the data for dhcl
        viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=disp_images, 
                                    max_imgs_per_row=3, fig_size=15, coastline=True, contourf=False,
                                    output_folder="imgs", show_var_names=True, land=True, background=BackgroundType.WHITE)

        viz_obj.plot_2d_data_np(np.array([model_mean.dchl.squeeze(), noaa_sat_mean.chlor_a.squeeze(), cop_sat_mean,
                                          model_mean.dchl.squeeze(), noaa_sat_mean.chlor_a.squeeze(),  cop_sat_mean ]),
                                ['NEMO DCHL', 'NOAA CHLORA', 'COP_CHLORA', 'Log NEMO DCHL', 'Log NOAA CHLORA', 'Log COP_CHLORA'],
                                norm = np.array([None, None, None, LogNorm(0.01,1), LogNorm(0.04,1), LogNorm(0.04,1)]), 
                                mincbar= [0.01, 0.01, 0.01, .00001, .00001, .00001], maxcbar=[10, 50, 50, 2, 2, 2],
                                file_name_prefix=f"DCHL_Comparison_{c_season}", title=f"Mean for {c_season} year {year}")

        # %% -------- Plot the data for 
    except Exception as e:
        print(f"Error processing season {c_season}: {e}")


# %% Generate images of swot masks 
print("Reading SWOT masks")
swot_masks_file = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/Masks/swot_masks_2016_2024.nc"
output_folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/preproc_imgs"
swot_masks = xr.open_dataset(swot_masks_file)
imgs_to_plot = 15
print(f"Total number of SWOT masks: {len(swot_masks.date)}")
# Crop to the first 10 masks
swot_masks = swot_masks.isel(date=slice(0, imgs_to_plot))
# Interpolate to the model grid
swot_masks = swot_masks.interp(lat=lats, lon=lons)
print("Interpolated SWOT masks done!")

# %%
for i in range(imgs_to_plot):
    print(f"Plotting mask {i}")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the SWOT mask
    im = ax.imshow(np.flipud(swot_masks['mask'].values[i,:,:]), 
                   extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                   transform=ccrs.PlateCarree(),
                   cmap='viridis',  # You can change the colormap as needed
                   alpha=0.7)  # Adjust alpha to control the transparency of the mask
    
    # Add coastlines
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    
    # Add land on top of the image
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='black', 
                                        facecolor='lightgrey')
    ax.add_feature(land, zorder=1)
    
    ax.set_title(f"SWOT mask {i}")
    
    # Set lat/lon ticks
    ax.set_xticks(np.arange(np.floor(lons[0]), np.ceil(lons[-1]), 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(lats[0]), np.ceil(lats[-1]), 1), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:.0f}°W' if v < 0 else f'{v:.0f}°E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:.0f}°S' if v < 0 else f'{v:.0f}°N'))
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(output_folder, f"swot_mask_{i:02d}.png"))
    plt.close()
# %%
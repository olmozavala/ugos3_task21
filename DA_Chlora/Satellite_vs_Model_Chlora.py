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
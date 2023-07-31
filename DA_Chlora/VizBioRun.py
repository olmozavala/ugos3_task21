# %% 
import sys
sys.path.append("eoas_pyutils")

from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import PlotMode, BackgroundType
from proc_utils.gom import lc_from_date, lc_from_ssh
from io_utils.coaps_io_data import get_aviso_by_month, get_aviso_by_date, get_biorun_cicese_nemo_by_date, get_chlora_noaa_by_date, get_hycom_gom_raw_by_date, get_sst_by_date

from shapely.geometry import LineString
from matplotlib.colors import LogNorm
import xarray as xr
import cmocean.cm as ccm
import os
from os.path import join
from datetime import datetime, date, timedelta
import numpy as np

# %% 
# Generate an array of datetimes from 2010 to 2020
start_date = date(2019, 3, 5)
end_date = date(2019, 3, 7)
dates = [start_date + timedelta(days=x) for x in range(0, (end_date - start_date).days)]

aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
satellite_sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
sss_folder = "/unity/f1/ozavala/DATA/GOFFISH/SSS/SMAP_Global"
chlora_folder = "/unity/f1/ozavala/DATA/GOFFISH/CHLORA/NOAA"
# %%
lon = [-98, -76]  
lat = [17.5, 32.5]
bbox = [lat[0], lat[1], lon[0], lon[1]]

# For each day we are going to load the AVISO data, Chlora data, SST data, LCS data and plot them
prev_month = -1
# for c_date in dates:
# Current date as string
c_date = dates[0]
c_date_str = c_date.strftime("%Y-%m-%d")

# %%  -------- Read AVISO data
c_month = c_date.month
if prev_month != c_month:
    aviso_data, lats, lons = get_aviso_by_month(aviso_folder, c_date, bbox=bbox)
aviso_today = aviso_data.adt[c_date.day-1, :, :]

# %% -------- Read Chlor-a NOAA data
chlora_data, _,  _= get_chlora_noaa_by_date(chlora_folder, c_date, bbox=bbox)


#  %%-------- Read HYCOM data
hycom_data, _,  _=  get_hycom_gom_raw_by_date(c_date, bbox)

# # Select current day
hycom_today_ssh = hycom_data.ssh[0, :, :]
hycom_today_sst = hycom_data.temperature[0, 0, :, :]

# %% -------- Read Stellite SST data
sst_data, lats_sst, lons_sst= get_sst_by_date(satellite_sst_folder, c_date, bbox=bbox)

# %% -------- Biorun from cicese
biorun, lats, lons = get_biorun_cicese_nemo_by_date(c_date, bbox)

# %%--------------- Interpolate all the data to the lat and lon from sst satellite
biorun = biorun.interp(latitude=lats_sst, longitude=lons_sst)
aviso_today = aviso_today.interp(latitude=lats_sst, longitude=lons_sst)
chlora_data =  chlora_data.interp(latitude=lats_sst, longitude=lons_sst)
hycom_today_ssh = hycom_today_ssh.interp(latitude=lats_sst, longitude=lons_sst)
hycom_today_sst = hycom_today_sst.interp(latitude=lats_sst, longitude=lons_sst)


# %% --------------- Plot all the obtained datasets -----------------------
viz_obj = EOAImageVisualizer(lats=lats_sst, lons=lons_sst, disp_images=True, 
                            max_imgs_per_row=2, fig_size=15,
                             output_folder="imgs", show_var_names=True)

lc = lc_from_date(c_date)
mylinestring = LineString(list(lc))
viz_obj.__setattr__('additional_polygons', [mylinestring])
viz_obj.plot_2d_data_np(np.array([aviso_today, hycom_today_ssh, hycom_today_sst, 
                                  sst_data['analysed_sst'][0, :, :], chlora_data['chlor_a'][0, 0, :, :]]), 
                        ['AVISO ADT', 'HYCOM SSH', 'HYCOM SST', 'Satellite SST', 'Chlor-a NOAA'],
                        norm = np.array([None, None, None, None, LogNorm(0.07,1)]), 
                        file_name_prefix=c_date_str, title=c_date_str)


# %%
viz_obj = EOAImageVisualizer(lats=lats_sst, lons=lons_sst, disp_images=True, 
                            max_imgs_per_row=2, fig_size=15,
                             output_folder="imgs", show_var_names=True)

lc_bio = lc_from_ssh(biorun.ssh[0, :, :].values, lons_sst, lats_sst, np.nanmean(biorun.ssh[0, :, :].values))
mylinestring = LineString(list(lc_bio))
viz_obj.__setattr__('additional_polygons', [mylinestring])
viz_obj.plot_2d_data_np(np.array([biorun.nchl[0, :, :], biorun.dchl[0, :, :], biorun.ssh[0, :, :], biorun.temperature[0, :, :]]), 
                        ['Biorun NCHL', 'Biorun DCHL', 'Biorun SSH', 'Biorun SST'],
                        norm = np.array([LogNorm(.03,.30), LogNorm(0.005,.3), None, None]), 
                        file_name_prefix=c_date_str, title=c_date_str)
# %%

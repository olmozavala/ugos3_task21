import xarray as xr
import numpy as np
import h5py
import sys
import pandas as pd
from proc_utils.geometries import histogram_from_locations
from datetime import timedelta, datetime
from enum import Enum
from os.path import join

# Vit Contour where circulation is maximum
class EddyProperty(Enum):
    lon_center = 0  #  '0.Value of Xcenter                      '; ...
    lat_center = 1  #  '1.Value of Ycenter                      '; ...
    xcentroid = 2  #  '2.Value of Xcentroid Vit                '; ...
    ycentroid = 3  #  '3.Value of Ycentroid Vit                '; ...
    table_xcontour_out = 4 #  '4.Table of Xcontour Out                 '; ...
    table_ycontour_out = 5  #  '5.Table of Ycontour Out                 '; ...
    radius_out = 6  #  '6.Equivalent Radius Out [km]            '; ...
    amplitude_out = 7  #  '7.Amplitude Out [m]                     '; ...
    table_xcountour = 8  #  '8.Table of Xcontour Vit                 '; ...
    table_ycountour = 9  #  '9.Table of Ycontour Vit                '; ...
    radius = 10  #  '10.Equivalent Radius Vit [km]           '; ...
    amplitude = 11  #  '11.Amplitude Vit [m]                    '; ...
    azimuthal = 12  #  '12.Azimuthal Vit [m/s]                  '; ...
    center_vorticity = 13  #  '13.Vorticity at center  [1/s]           '; ...
    mean_eke = 14  #  '14.Mean EKE [(m/s)^2]                   '; ...
    mean_speed = 15  #  '15.Mean Speed [m/s]                     '; ...
    mean_vorticity = 16  #  '16.Mean Vorticity [1/s]                 '; ...
    mean_straining = 17  #  '17.Mean Straining deformation rate [1/s]'; ...
    mean_shearing = 18  #  '18.Mean shearing deformation rate [1/s] '; ...
    mean_okubo_weiss = 19  #  '19.Mean Okubo_Weiss parameter  [1/s]    ';...
    array_azim = 20  #  '20.2D Array azim Vit/km [2]             '];

CYC_EDDY = 'Cyclonic_Cell'
ANTICYC_EDDY = 'Anticyclonic_Cell'

file_name = "/data/GOFFISH/TOEddies/Eddies/adt_1993-01-01.mat"
output_folder = "output"
df = h5py.File(file_name, 'r')

lats = np.squeeze(df['Y'][:,:])
lons = np.squeeze(df['X'][:,:]) - 360

def get_num_eddies(df):
    '''
    It obtains the number of cyclonic and anticyclonic eddies from the dataset
    :param df:
    :return:
    '''
    num_cyc_eddies = df[CYC_EDDY].shape[1]
    num_anti_eddies = df[ANTICYC_EDDY].shape[1]
    return num_cyc_eddies, num_anti_eddies

def get_eddy_var(df, eddy_type, eddie_var):
    '''
    Obtains a specific eddy variable from the input dataset. The desired
    variable is selected from the EddyProperty enum list.
    :param df:
    :param eddy_type:
    :param eddie_var:
    :return:
    '''
    df_e = df[eddy_type]
    n_eddies = df_e.shape[1]
    eddies_var = np.zeros(n_eddies)
    for i in range(n_eddies):
        value_ref = df[eddy_type][eddie_var.value, i]
        eddies_var[i] = df[value_ref][0][0]
    return eddies_var

## # Compute metrics looping through all the eddies
date_format = "%Y-%m-%d"
start_date_str = "2019-01-01"
# end_date_str = "2019-01-05"
end_date_str = "2021-12-31"
start_date = datetime.strptime(start_date_str,date_format)
cur_date = start_date
end_date = datetime.strptime(end_date_str,date_format)

# Statistical metrics
number_eddies = []
eddy_radius = []
eddy_eke = []
eddy_amplitude = []
dates_str = []

# Spatial histograms metrics
gridres = 1
minlat, maxlat, rangelat = (16, 31, 31 - 16)
minlon, maxlon, rangelon = (-98, -78, 98 - 78)
lats_coarse = np.linspace(minlat, maxlat, int(rangelat / gridres) + 1)
lons_coarse = np.linspace(minlon, maxlon, int(rangelon / gridres) + 1)
grid_coarse = np.zeros((len(lats_coarse), len(lons_coarse)))

geo_num_eddies_cyc = grid_coarse.copy()
geo_num_eddies_anti = grid_coarse.copy()
geo_radius_cyc = grid_coarse.copy()
geo_radius_anti = grid_coarse.copy()
geo_amp_cyc = grid_coarse.copy()
geo_amp_anti = grid_coarse.copy()
geo_eke_cyc = grid_coarse.copy()
geo_eke_anti = grid_coarse.copy()

while cur_date <= end_date:
    cur_date_str = cur_date.strftime(date_format)
    print(f"Working with {cur_date_str}")
    file_name = f"/data/GOFFISH/TOEddies/Eddies/adt_{cur_date_str}.mat"
    df = h5py.File(file_name, 'r')
    # ---------- Compute Metrics
    # Num Eddy
    num_cyc_eddies, num_anti_eddies = get_num_eddies(df)
    # Center of eddies
    lon_center_cyc = get_eddy_var(df, CYC_EDDY, EddyProperty.lon_center) - 360
    lat_center_cyc = get_eddy_var(df, CYC_EDDY, EddyProperty.lat_center)
    lon_center_anti = get_eddy_var(df, ANTICYC_EDDY, EddyProperty.lon_center) - 360
    lat_center_anti = get_eddy_var(df, ANTICYC_EDDY, EddyProperty.lat_center)
    # Eddy Amplitude
    amp_cyc_eddies = get_eddy_var(df, CYC_EDDY, EddyProperty.amplitude_out)
    amp_anti_eddies = get_eddy_var(df, ANTICYC_EDDY, EddyProperty.amplitude_out)
    # Eddy Radius
    radius_cyc_eddies = get_eddy_var(df, CYC_EDDY, EddyProperty.radius_out)
    radius_anti_eddies = get_eddy_var(df, ANTICYC_EDDY, EddyProperty.radius_out)
    # EKE
    eke_cyc_eddies = get_eddy_var(df, CYC_EDDY, EddyProperty.mean_eke)
    eke_anti_eddies = get_eddy_var(df, ANTICYC_EDDY, EddyProperty.mean_eke)

    # ------------ Compute histograms
    # Number of eddies
    geo_num_eddies_cyc = histogram_from_locations(geo_num_eddies_cyc, lats_coarse, lons_coarse,
                                                   zip(lat_center_cyc, lon_center_cyc))
    geo_num_eddies_anti = histogram_from_locations(geo_num_eddies_anti, lats_coarse, lons_coarse,
                                                    zip(lat_center_anti, lon_center_anti))
    # Eddy Radius
    geo_radius_cyc = histogram_from_locations(geo_radius_cyc, lats_coarse, lons_coarse,
                                               zip(lat_center_cyc, lon_center_cyc), values=radius_cyc_eddies)
    geo_radius_anti = histogram_from_locations(geo_radius_anti, lats_coarse, lons_coarse,
                                                zip(lat_center_anti, lon_center_anti), values=radius_anti_eddies)
    # Eddy amp
    geo_amp_cyc = histogram_from_locations(geo_radius_cyc, lats_coarse, lons_coarse,
                                        zip(lat_center_cyc, lon_center_cyc), values=amp_cyc_eddies)
    geo_amp_anti = histogram_from_locations(geo_radius_anti, lats_coarse, lons_coarse,
                                                zip(lat_center_anti, lon_center_anti), values=amp_anti_eddies)
    # Eddy eke
    geo_eke_cyc = histogram_from_locations(geo_eke_cyc, lats_coarse, lons_coarse,
                                                zip(lat_center_cyc, lon_center_cyc), values=eke_cyc_eddies)
    geo_eke_anti = histogram_from_locations(geo_eke_anti, lats_coarse, lons_coarse,
                                                zip(lat_center_anti, lon_center_anti), values=eke_anti_eddies)

    # ---------- Append current time to whole time series
    dates_str.append(cur_date_str)
    number_eddies.append([num_cyc_eddies, num_anti_eddies])
    eddy_amplitude.append([amp_cyc_eddies.mean(), amp_anti_eddies.mean()])
    eddy_radius.append([radius_cyc_eddies.mean(), radius_anti_eddies.mean()])
    eddy_eke.append([eke_cyc_eddies.mean(), eke_anti_eddies.mean()])

    cur_date += timedelta(days=1)

# ---------- Average spatial metrics
nonan = ~np.isnan(geo_num_eddies_cyc)
geo_radius_cyc[nonan] = geo_radius_cyc[nonan] / geo_num_eddies_cyc[nonan]
geo_eke_cyc[nonan] = geo_eke_cyc[nonan] / geo_num_eddies_cyc[nonan]
geo_amp_cyc[nonan] = geo_amp_cyc[nonan] / geo_num_eddies_cyc[nonan]

geo_radius_anti[nonan] = geo_radius_anti[nonan] / geo_num_eddies_anti[nonan]
geo_eke_anti[nonan] = geo_eke_anti[nonan] / geo_num_eddies_anti[nonan]
geo_amp_anti[nonan] = geo_amp_anti[nonan] / geo_num_eddies_anti[nonan]

for geo_histo in [geo_num_eddies_cyc, geo_num_eddies_anti, geo_radius_cyc, geo_radius_anti, geo_eke_cyc, geo_eke_anti,
                  geo_amp_cyc, geo_amp_anti, geo_radius_cyc, geo_radius_anti]:
    zerovals = geo_histo == 0
    geo_histo[zerovals] = np.nan

number_eddies = np.array(number_eddies)
eddy_amplitude = np.array(eddy_amplitude)
eddy_radius = np.array(eddy_radius)
eddy_eke = np.array(eddy_eke)

## Saving data
times = pd.date_range(start_date_str, end_date_str)
ds = xr.Dataset(
    {
        # 2D histos
        "histo_number_eddies_cyc":(("latitude", "longitude"), geo_num_eddies_cyc),
        "histo_number_eddies_anti":(("latitude", "longitude"), geo_num_eddies_anti),
        "histo_eddy_radius_cyc":(("latitude", "longitude"), geo_radius_cyc),
        "histo_eddy_radius_anti":(("latitude", "longitude"), geo_radius_anti),
        "histo_eddy_amp_cyc":(("latitude", "longitude"),  geo_amp_cyc),
        "histo_eddy_amp_anti":(("latitude", "longitude"), geo_amp_anti),
        "histo_eddy_eke_cyc":(("latitude", "longitude"),  geo_eke_cyc),
        "histo_eddy_eke_anti":(("latitude", "longitude"), geo_eke_anti),
        # 1D vars
        "num_eddies_cyc":(("time"), number_eddies[:,0]),
        "num_eddies_anti": (("time"), number_eddies[:, 1]),
        "eke_cyc": (("time"), eddy_eke[:, 0], {"units":"m2 s-2"}),
        "eke_anti": (("time"), eddy_eke[:, 1], {"units":"m2 s-2"}),
        "amp_cyc": (("time"), eddy_amplitude[:, 0]),
        "amp_anti": (("time"), eddy_amplitude[:, 1]),
        "radius_cyc": (("time"), eddy_radius[:, 0]),
        "radius_anti": (("time"), eddy_radius[:, 1]),
    },
    {"time": times, "latitude": lats_coarse, "longitude": lons_coarse}
)

ds.to_netcdf(join(output_folder,f"{start_date_str}_{end_date_str}.nc"))
print("Done!")
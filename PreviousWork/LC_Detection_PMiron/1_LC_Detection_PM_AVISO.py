import os
import json
from os.path import join
import xarray as xr
import numpy as np
import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.append("eoas_pyutils/")
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import PlotMode, BackgroundType
from proc_utils.geometries import intersect_polygon_grid
from shapely.geometry import Polygon, LineString
from skimage import measure
from viz_utils.eoa_viz import EOAImageVisualizer
from project_utils.lc_utils import read_aviso_data_adtuv, read_aviso_data, bathymetry_etopo

#%% --------- Reading data --------------------
print("Reading data...")
years = range(1993,2022)
input_folder = "/data/GOFFISH/AVISO"
adt, lat, lon = read_aviso_data(years, input_folder)
output_folder = "./output"

viz_obj = EOAImageVisualizer(disp_images=True, output_folder=join(output_folder,'imgs'), lats=lat, lons=lon)
viz_obj.plot_2d_data_np(adt, ['adt'], title=F'ADT Example', file_name_prefix='adt')
print("Done!")

#%%
### Remove daily average ADT over the GoM deepwater (>200m depth) from ADT to remove thermal expansion effects
base = datetime.datetime(years[0],1,1)
numdays = len(adt[:,0,0])
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]

# box for Caribbean and Atlantic ocean
gom_bnd = np.array([[-87.5, 21.15], [-84.15, 22.35], [-82.9, 22.9], [-81, 22.9], [-81, 27], [-82.5, 32.5], [-76.5, 32.5], [-76.5, 16.5], [-90, 16.5], [-87.5, 21.15]])
gom_path = path.Path(gom_bnd)

print("Making intersection...")
gom = np.ma.copy(adt[0,:,:])
gom = intersect_polygon_grid(gom, lat, lon, gom_bnd)
nan_idxs = np.isnan(gom)
gom = 1 - gom.astype(bool)
gom[nan_idxs] = np.nan
viz_obj.__setattr__('additional_polygons', [Polygon(gom_bnd)])
viz_obj.plot_2d_data_np(gom, ['binary_grid'], flip_data=False, rot_90=False, title=F'ADT ', file_name_prefix='intersect')
print("Done!")

#%%
## bathymetry data for the domain
print("Mask for deeper areas")
plot_lon = [-98, -78]
plot_lat = [18.125, 31]
ex, ey, ez = bathymetry_etopo(plot_lon, plot_lat)

skip = 1
elon = ex[::skip]
elat = ey[::skip]
depth = RegularGridInterpolator((elon, elat), np.moveaxis(ez, [0, 1], [1, 0]), method='linear', bounds_error=False,
                                fill_value=np.nan)

# get a mask for deepwater of the GoM which is defined by Leben et al. by 200m+
gom_deep = np.copy(gom.T)
for i in range(0, len(lon)):
    for j in range(0, len(lat)):
        if depth([lon[i], lat[j]]) > -200:
            gom_deep[i, j] = False
print("Done!")

viz_obj.__setattr__('additional_polygons', [])
viz_obj.plot_2d_data_np(gom_deep, ['Deep GoM'], flip_data=True, rot_90=True, title=F'GoM Deep', file_name_prefix='deeper_200')

#%%
## calculate the mean value of adt in the deep GoM (deeper than 200m)
mean_adt = np.mean(adt[:, gom_deep.T == 1])
adt -= mean_adt
print(F"Mean adt: {mean_adt}")

day = 0
viz_obj.plot_2d_data_np(adt[day,:,:], ['adt'], title=F'LC contours',  file_name_prefix='17cm_lc_example', plot_mode=PlotMode.MERGED)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.contourf(lon, lat, adt[day, :, :])
cc = ax.contour(lon, lat, adt[day, :, :], [0.17])

ax.tick_params(labelsize=16)

ax.set_title('%s' % date_list[day].strftime("%m/%d/%Y"))
plt.show()
# fig.savefig('lc_ex.png', dpi=600)

# Find contours of 17 cm
ext_lon = np.zeros(numdays)
ext_lat = np.zeros(numdays)
cc = None

#%%
print("Getting contours...")
viz_obj = EOAImageVisualizer(disp_images=False, output_folder=join(output_folder,'imgs'), lats=lat, lons=lon)
LC_contours = {}
for day in range(numdays):
    if day % 100 == 0:
        print(day)
    c_day_str = date_list[day].strftime("%Y-%m-%d")
    try:
        # It finds it rows by columns (lat, lon)
        # cco = measure.find_contours(adt[day, 10:, :], 0.17, fully_connected='low', positive_orientation='low')

        # We find the contours on a subregion (min lat of 20)
        cco = measure.find_contours(adt[day, 10:, :], 0.17, fully_connected='low', positive_orientation='low')
        cc = change_units(cco, lon, lat[10:])
        cc = filter_contour2(cc, gom_path)
        # for c_c in cc:
        #     pos = zip(c_c[:,1], c_c[:,0])
        #     viz_obj.__setattr__('additional_polygons', [LineString(pos)])
        #     viz_obj.plot_2d_data_np(adt[day, :, :], ['adt'], title=F'{c_day_str}', file_name_prefix=F'{c_day_str}')

        # Concatenating all the contours into a single one
        # Sorts the contours for making the linestring properly it assumes we have at most two contours for the LC
        indexes = [0]
        if len(cc) >= 2:
            indexes = [0, 1]
            # for a in cc:
            #     print(F"{c_day_str} contour: ({a[0, 1]:0.2f}, {a[0, 0]:0.2f}),"
            #                                F"({a[-1,1]:0.2f}, {a[-1, 0]:0.2f})")
            # if cc[0][0,1] > cc[1][0, 1]:
                # print(F"Flipping")
                # print(F"{c_day_str} First contour: {cc[0][0,1]:0.3f} - {cc[0][-1,1]:0.3f}")
                # print(F"{c_day_str} Second contour: {cc[1][0,1]:0.3f} - {cc[1][-1,1]:0.3f}")
                # indexes = [1, 0]  # In this case we flip the order

        lc_lats = np.concatenate([np.flip(cc[i][:,0]) for i in indexes])
        lc_lons = np.concatenate([np.flip(cc[i][:,1]) for i in indexes])
        pos = zip(lc_lons, lc_lats)
        LC_contours[c_day_str] = pos
        # ext_lon[day], ext_lat[day] = loop_current_extent3(cc, gom_path)
    except Exception as e:
        print(F"Failed for day {day}")

#%%
json_data = json.dumps({key:";".join([F"{x[0]:0.5f},{x[1]:0.5f}" for x in value]) for key,value in LC_contours.items()})
f = open(join(output_folder,"lc_contours","LC_data.json"),"w")
f.write(json_data)
f.close
print("Super done!")
import json
from project_utils.lc_utils import read_aviso_data
import numpy as np
import datetime
from shapely.geometry import LineString
import rasterio.features
import sys
sys.path.append("eoas_pyutils/")
from viz_utils.eoa_viz import EOAImageVisualizer

#%%
print("Reading lc_contour..")
f = open('./output/lc_contours/LC_data.json')
lc_contour = json.load(f)
lc_contour = {key: np.array([np.fromstring(x, sep=',') for x in value.split(';')]) for key,value in lc_contour.items()}
print("Done!")

##%
print("Reading data...")
years = np.arange(1993,2013)
tot_years = years[-1] - years[0] + 1
input_folder = "/data/GOFFISH/AVISO"
adt, lat, lon = read_aviso_data(years, input_folder)

base = datetime.datetime(years[0],1,1)
date_list = [base + datetime.timedelta(days=x) for x in range(365*tot_years)]
print("Done!")

## ----------- If we want to save a mask -----------------
import matplotlib.pyplot as plt

minlat = lat.data.min()
maxlat = lat.data.max()
latlen = lat.shape[0]
rangelat = maxlat - minlat
minlon = lon.data.min()
maxlon = lon.data.max()
rangelon = maxlon - minlon
lonlen = lon.shape[0]

tot_ex = adt.shape[0]
mask = np.zeros(adt.shape)
temp = adt.data
nans = np.isnan(temp)
for iday, c_day_str in enumerate([x.strftime("%Y-%m-%d") for x in date_list]):
    if c_day_str in lc_contour.keys():
        t = lc_contour[c_day_str]
        t[:, 0] = ((t[:, 0] - minlon) / rangelon) * lonlen
        t[:, 1] = ((t[:, 1] - minlat) / rangelat) * latlen
        ls = LineString(t)
        rasterized_lc= rasterio.features.rasterize([ls], out_shape=(latlen, lonlen), all_touched=True)
        mask[iday,:,:] = rasterized_lc

        if iday % 10 == 0:
            print(c_day_str)
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            axs[0].imshow(adt[iday,::-1,:])
            axs[0].set_title(f"{c_day_str}  SSH")
            axs[1].imshow(mask[iday,::-1,:])
            axs[1].set_title(f"{c_day_str}  LC mask")

mask[nans] = np.nan
np.save("/home/olmozavala/Dropbox/MyPresentationsConferencesAndWorkshops/2022/DA_Workshop/da_workshop_2022/imgs/aviso/1993-2013-mask", mask.data)
exit()

##%
viz_obj = EOAImageVisualizer(disp_images=True, output_folder='/data/GOFFISH/AVISO/imgs', lats=lat, lons=lon)
iday = 0
for c_day_str in [x.strftime("%Y-%m-%d") for x in date_list]:
    if c_day_str in lc_contour.keys():
        print(F"Working with day {c_day_str}")
        ls = LineString(lc_contour[c_day_str])
        viz_obj.__setattr__('additional_polygons', [ls])
        viz_obj.plot_2d_data_np(adt[iday,:,:], ['adt'], title=F'LC {c_day_str}', file_name_prefix=F'lc_{c_day_str}')
    iday += 1
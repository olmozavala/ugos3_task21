import xarray as xr
import numpy as np
import h5py
import sys
sys.path.append("eoas_pyutils/")
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import PlotMode
import cmocean as ccm
from shapely.geometry import Point, Polygon
from datetime import timedelta, datetime


#  '0.Value of Xcenter                      '; ...
#  '1.Value of Ycenter                      '; ...
#  '2.Value of Xcentroid Vit                '; ...
#  '3.Value of Ycentroid Vit                '; ...
#  '4.Table of Xcontour Out                 '; ...
#  '5.Table of Ycontour Out                 '; ...
#  '6.Equivalent Radius Out [km]            '; ...
#  '7.Amplitude Out [m]                     '; ...
#  '8.Table of Xcontour Vit                 '; ...
#  '9.Table of Ycontour Vit                '; ...
#  '10.Equivalent Radius Vit [km]           '; ...
#  '11.Amplitude Vit [m]                    '; ...
#  '12.Azimuthal Vit [m/s]                  '; ...
#  '13.Vorticity at center  [1/s]           '; ...
#  '14.Mean EKE [(m/s)^2]                   '; ...
#  '15.Mean Speed [m/s]                     '; ...
#  '16.Mean Vorticity [1/s]                 '; ...
#  '17.Mean Straining deformation rate [1/s]'; ...
#  '18.Mean shearing deformation rate [1/s] '; ...
#  '19.Mean Okubo_Weiss parameter  [1/s]    ';...
#  '20.2D Array azim Vit/km [2]             '];
##

file_name = "/data/GOFFISH/TOEddies/Eddies/adt_1993-01-01.mat"
output_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/PreviousWork/REMI_TOEddies_4_Olmo/TOEddies/output/Detected_Eddies"
df = h5py.File(file_name, 'r')

##
lats = np.squeeze(df['Y'][:,:])
lons = np.squeeze(df['X'][:,:]) - 360

##
def append_eddies(df, eddy_type, centers):
    df_e = df[eddy_type]
    n_eddies = df_e.shape[1]
    for i in range(n_eddies):
        ref_x = df[eddy_type][0, i]
        ref_y = df[eddy_type][1, i]
        centers.append(np.array([np.squeeze(np.array(df[ref_x])), np.squeeze(np.array(df[ref_y]))]))

def append_contours(df, eddy_type, contours):
    df_e = df[eddy_type]
    n_eddies = df_e.shape[1]
    for i in range(n_eddies):
        ref_x = df[eddy_type][4,i]
        ref_y = df[eddy_type][5,i]
        contours.append(Polygon(zip(np.squeeze(np.array(df[ref_x])), np.squeeze(np.array(df[ref_y])))))

# Loop through all the eddies
date_format = "%Y-%m-%d"
start_date = datetime.strptime("2019-01-01",date_format)
cur_date = start_date
end_date = datetime.strptime("2021-12-31",date_format)

while cur_date < end_date:

    cur_date_str = cur_date.strftime(date_format)
    file_name = f"/data/GOFFISH/TOEddies/Eddies/adt_{cur_date_str}.mat"
    print(file_name)

    df = h5py.File(file_name, 'r')
    centers = []
    contours = []
    append_eddies(df, 'Anticyclonic_Cell', centers)
    append_eddies(df, 'Cyclonic_Cell', centers)
    append_contours(df, 'Anticyclonic_Cell', contours)
    append_contours(df, 'Cyclonic_Cell', contours)

    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=False, output_folder=output_folder)
    viz_obj.__setattr__('additional_polygons', np.concatenate([[Point(x) for x in centers], contours]))
    viz_obj.plot_2d_data_np(df['ADT'], ['ADT'], title=cur_date_str, file_name_prefix=cur_date_str, cmap=ccm.cm.curl, plot_mode=PlotMode.RASTER)
    cur_date += timedelta(days=1)
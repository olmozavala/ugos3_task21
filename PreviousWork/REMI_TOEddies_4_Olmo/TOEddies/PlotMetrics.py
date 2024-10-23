import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from os.path import join
import sys
sys.path.append("eoas_pyutils/")
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import BackgroundType

from viz_utils.constants import PlotMode
import cmocean as ccm
from shapely.geometry import Point, Polygon

file_name = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/PreviousWork/REMI_TOEddies_4_Olmo/TOEddies/output/2019-01-01_2021-12-31.nc"
df = xr.open_dataset(file_name)
output_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/PreviousWork/REMI_TOEddies_4_Olmo/TOEddies/output"

date_format = "%Y-%m-%d"
dates_str = pd.to_datetime(df.time).strftime(date_format)
start_date = dates_str[0]
end_date = dates_str[-1]
lats = df['latitude']
lons = df['longitude']

##
rolling_mean_size = 30 # in Days
rm_kernel = np.ones(rolling_mean_size) # in Days
number_eddies =     np.array(list(zip(np.convolve(df['num_eddies_cyc'], rm_kernel, 'valid')/rolling_mean_size, np.convolve(df['num_eddies_anti'], rm_kernel, 'valid')/rolling_mean_size)))
eddy_amplitude =    np.array(list(zip(np.convolve(df['amp_cyc'], rm_kernel, 'valid')/rolling_mean_size       , np.convolve(df['amp_anti'], rm_kernel, 'valid')/rolling_mean_size)))
eddy_eke =          np.array(list(zip(np.convolve(df['eke_cyc'], rm_kernel, 'valid')/rolling_mean_size       , np.convolve(df['eke_anti'], rm_kernel, 'valid')/rolling_mean_size)))
eddy_radius =       np.array(list(zip(np.convolve(df['radius_cyc'], rm_kernel, 'valid')/rolling_mean_size    , np.convolve(df['radius_anti'], rm_kernel, 'valid')/rolling_mean_size)))

geo_num_eddies_cyc = df['histo_number_eddies_cyc']
geo_num_eddies_anti = df['histo_number_eddies_anti']
geo_radius_cyc = df['histo_eddy_radius_cyc']
geo_radius_anti = df['histo_eddy_radius_anti']
geo_amp_cyc =  df['histo_eddy_amp_cyc']
geo_amp_anti = df['histo_eddy_amp_anti']
geo_eke_cyc =  df['histo_eddy_eke_cyc']
geo_eke_anti = df['histo_eddy_eke_anti']

## Plots statistics
fig, axs = plt.subplots(4,1, figsize=(20,7))

x = np.arange(len(number_eddies))
# xticks_labels = [dates_str[int(x)] for x in xticks_pos]
months = ['F','M','A','M','J','J','A','S','O','N']
xticks_labels = ['2019'] + months + ['2020'] + months + ['2021'] + months
xticks_pos = np.linspace(0, len(dates_str)-1, len(xticks_labels))

axs[0].plot(x, number_eddies[:,0], 'b', label='CE')
axs[0].plot(x, number_eddies[:,1], 'r', label='AE')
axs[0].set_ylabel('Number of Eddies')
axs[0].set_ylim(5,30)
axs[0].set_xticks(xticks_pos, labels=xticks_labels, rotation=0)

axs[1].plot(x, eddy_amplitude[:,0]*100, 'b', label='CE')
axs[1].plot(x, eddy_amplitude[:,1]*100, 'r', label='AE')
axs[1].set_ylabel('Eddy Amplitude (cm)')
axs[1].set_ylim(2.5,15)
axs[1].set_xticks(xticks_pos, labels=xticks_labels, rotation=0)
# axs[0].grid()

axs[2].plot(x, eddy_radius[:,0], 'b', label='CE')
axs[2].plot(x, eddy_radius[:,1], 'r', label='AE')
axs[2].set_ylabel('Eddy Radius (km)')
axs[2].set_ylim(50,110)
axs[2].set_xticks(xticks_pos, labels=xticks_labels, rotation=0)
# axs[0].grid()

axs[3].plot(x, eddy_eke[:,0]*1e4, 'b', label='CE')
axs[3].plot(x, eddy_eke[:,1]*1e4, 'r', label='AE')
axs[3].set_ylabel(r"EKE $\frac{m^2}{s^2}$")
axs[3].set_ylim(0,3000)
axs[3].set_xticks(xticks_pos, labels=xticks_labels, rotation=0)
# axs[0].grid()

[axs[x].margins(x=0) for x in range(4)]
[axs[x].legend(loc='upper right') for x in range(4)]
plt.tight_layout(pad=.5)
plt.savefig(join(output_folder,f"{start_date}_{end_date}.png"))
plt.show()

## Plots histogram maps
vizobj = EOAImageVisualizer(disp_images=False, output_folder=output_folder, lats=[lats], lons=[lons],
                            background=BackgroundType.BLUE_MARBLE_HR, auto_colormap=False)
_background = BackgroundType.BLUE_MARBLE_LR  # Select the background to use
# # Num of eddies
cmap = 'rainbow'
vizobj.plot_2d_data_np(geo_num_eddies_cyc, ['Number of CE'], title='Number of CE Eddy', file_name_prefix='num_eddies_ce', cmap=cmap)
vizobj.plot_2d_data_np(geo_num_eddies_anti, ['Number of AE'], title='Number of AE Eddy', file_name_prefix='num_eddies_ae', cmap=cmap)
# Eddy Radius
vizobj.plot_2d_data_np(geo_radius_cyc, ['Eddy Radius CE'],  title='Mean Radius of CE Eddy (km)', file_name_prefix='radius_ce', cmap=cmap)
vizobj.plot_2d_data_np(geo_radius_anti,['Eddy Radius AE'], title='Mean Radius of AE Eddy (km)', file_name_prefix='radius_ae', cmap=cmap)
# Eddy Amplitude
vizobj.plot_2d_data_np(geo_amp_cyc, ['Eddy Amplitude CE'],  title='Mean Amplitude of CE Eddy (cm)', file_name_prefix='amplitude_ce', cmap=cmap)
vizobj.plot_2d_data_np(geo_amp_anti,['Eddy Amplitude AE'], title='Mean Amplitude of AE Eddy (cm)', file_name_prefix='amplitude_ae', cmap=cmap)
# Eddy EKE
vizobj.plot_2d_data_np(geo_eke_cyc, [r"EKE $\frac{m^2}{s^2}$ CE"], title='Mean EKE of CE Eddy', file_name_prefix='eke_ce', cmap=cmap)
vizobj.plot_2d_data_np(geo_eke_anti, [r"EKE $\frac{m^2}{s^2}$ AE"], title='Mean EKE of AE Eddy', file_name_prefix='eke_ae', cmap=cmap)

print("Done!")
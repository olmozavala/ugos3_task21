import requests
import os
from os.path import join
from multiprocessing import Pool

## This program dowloads SSS data for years 2015 to 2022

output_folder = "/Net/work/ozavala/GOFFISH/SSS/SMAP_Global"
years = range(2021,2023)
TOT_PROC = 10

def par_download(proc_id):
    for c_year in years:
        c_output_folder = join(output_folder,str(c_year))
        if not os.path.exists(c_output_folder):
            os.makedirs(c_output_folder)

        for c_day in range(1, 366):
            if c_day % TOT_PROC == proc_id:
                file_name = F"RSS_smap_SSS_L3_8day_running_{c_year}_{c_day:03d}_FNL_v05.0.nc"
                URL = F"https://data.remss.com/smap/SSS/V05.0/FINAL/L3/8day_running/{c_year}/{file_name}"
                try:
                    # ------ One option is to delete previous one
                    # if os.path.exists(output_file):
                    #     os.remove(output_file)
                    # ------- Another option is to download only if it doesn't exist
                    output_file = join(c_output_folder, file_name)
                    if os.path.exists(output_file):
                        continue
                    print(F"Downloading file for day {c_year}-{c_day:03d}: {URL}")
                    response = requests.get(URL)

                    open(output_file, "wb").write(response.content)
                except Exception as e:
                    print(F"Failed for file: {file_name}")

    print("Done!")

p = Pool(TOT_PROC)
p.map(par_download, range(TOT_PROC))


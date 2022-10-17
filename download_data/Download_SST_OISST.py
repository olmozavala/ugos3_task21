import os
from os.path import join
from multiprocessing import Pool
import datetime
from datetime import timedelta
import subprocess

#%% This program dowloads SST using PODAAC pip install podaac-data-subscriber
# https://github.com/podaac/data-subscriber/blob/main/README.md

# root_output_folder = "/Net/work/ozavala/GOFFISH/SST/OISST"
root_output_folder = "/data/GOFFISH/SST/OISST"

TOT_PROC = 1
start_date = datetime.date(2011,12,1)
final_end_date = datetime.date.today()
days_increment = 1
bbox = "-99 -74 17  31"

def par_download(proc_id):
    c_date = start_date
    c_end_date = c_date + timedelta(days=days_increment)
    i = 0
    while c_date < final_end_date:
        year = c_date.year
        output_folder = join(root_output_folder,str(year))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if i % TOT_PROC == proc_id:
            cmd = F"./subset_dataset_oisst.py -s {c_date.strftime('%Y%m%d')} -f {c_end_date.strftime('%Y%m%d')} " \
                  F"-b {bbox} -x MUR-JPL-L4-GLOB-v4.1"
            print(cmd)
            os.system(cmd)
            c_date = c_date + timedelta(days=days_increment)
            c_end_date = c_date + timedelta(days=days_increment)
            print("Done!")
        i += 1

        cmd = F"mv *.nc {output_folder}"
        print(cmd)
        os.system(cmd)

    print(F"Done all from process {proc_id}!")

#%%
p = Pool(TOT_PROC)
p.map(par_download, range(TOT_PROC))

#%%

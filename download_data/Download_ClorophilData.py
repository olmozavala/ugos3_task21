psw = "J>uL|O?-ZGIn"

import requests
import os
from os.path import join
from multiprocessing import Pool
import datetime
from datetime import timedelta
import subprocess

## This program dowloads SST using PODAAC pip install podaac-data-subscriber
# https://github.com/podaac/data-subscriber/blob/main/README.md

output_folder = "/Net/work/ozavala/GOFFISH/CHLORA/CEDA"
# output_folder = "/data/GOFFISH/CHLORA/CEDA"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

TOT_PROC = 1
start_date = datetime.date(2003,1,9)
final_end_date = datetime.date.today()
days_increment = 1
bbox = "-99 -74 17  31"

def par_download(proc_id):
    c_date = start_date
    c_end_date = c_date + timedelta(days=days_increment)
    i = 0
    while c_date < final_end_date:
        if i % TOT_PROC == proc_id:
            url = "http://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/ocean_colour/data/v3.1-release/geographic/netcdf/all_products/daily/v3.1/1997/ESACCI-OC-L3S-OC_PRODUCTS-MERGED-1D_DAILY_4km_GEO_PML_OCx_QAA-19970904-fv3.1.nc?time[0:1:0],chlor_a[0:1:0][0:1:0][0:1:0],lat[0:1:4319],lon[0:1:8639]"
            cmd = F"python remote_nc_reader.py {url} chlor_a"
            print(cmd)
            # os.system(cmd)
            c_date = c_date + timedelta(days=days_increment)
            c_end_date = c_date + timedelta(days=days_increment)
            print("Done!")
        i += 1
        exit()

    # cmd = F"mv -f *.nc {output_folder}"
    # print(cmd)
    # os.system(cmd)
    # print(F"Done all from process {proc_id}!")

p = Pool(TOT_PROC)
p.map(par_download, range(TOT_PROC))

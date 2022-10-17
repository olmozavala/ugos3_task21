import subprocess
import keyring
from calendar import monthrange

username = 'ozavalaromero'
# password = keyring.get_password("aviso",username)  # Use your own user and password for Copernicus

# output_folder = "/Net/work/ozavala/GOFFISH/AVISO"
output_folder = "/home/olmozavala/Downloads/Delete"
lon = [262, 305]  # [-98, -60]
lat = [7.5, 50]

# https://resources.marine.copernicus.eu/product-detail/SEALEVEL_GLO_PHY_L4_MY_008_047/INFORMATION
for year in range(2001, 2023):
    for month in range(1, 13):
        outfile = '%d-%02d.nc' % (year, month)

        args = F'--motu http://my.cmems-du.eu/motu-web/Motu ' \
               F'--service-id SEALEVEL_GLO_PHY_L4_MY_008_047-TDS --product-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D ' \
               F'--longitude-min {lon[0]} --longitude-max {lon[1]} --latitude-min {lat[0]} --latitude-max {lat[1]} ' \
               F'--date-min "{year}-{month:02d}-01 00:00:00" --date-max "{year}-{month:02d}-{monthrange(year, month)[1]} 00:00:00" ' \
               F'--variable sla --variable adt --variable ugos --variable vgos --variable ugosa --variable vgosa --variable err_sla --out-dir {output_folder} --out-name {outfile} ' \
               F'--user {username} --pwd {password}'

        print(args)
        subprocess.call(F"python -m motuclient {args}", shell=True)
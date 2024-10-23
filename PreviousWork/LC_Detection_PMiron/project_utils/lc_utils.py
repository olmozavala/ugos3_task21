import xarray as xr
from scipy.interpolate import interp1d
import os
from os.path import join
from functools import reduce
import numpy as np

# add degree °W and °E and remove negative sign
def label_lon(x, pos):
    'The two args are the value and tick position'
    if x > 0:
        return '%d°E' % x
    else:
        return '%d°W' % -x

# add °N and °S and remove negative sign
def label_lat(x, pos):
    'The two args are the value and tick position'
    if x > 0:
        return '%d°N' % x
    else:
        return '%d°S' % -x

def cm2inch(value):
    return value/2.54

def bathymetry_etopo(lon, lat):
    """
    # load the etopo data inside the specified lon/lat limit
    :param lon: list [min, max] longitude of subdomain
    :param lat: list [min, max] latitude of subdomain
    :return: lon, lat, height
    """

    folder = '../../data/extra'
    data = xr.open_dataset(join(folder, 'ETOPO1_Ice_c_gmt4.grd'))
    subdata = data.sel(x=slice(lon[0], lon[1]), y=slice(lat[0],lat[1]))
    elon = subdata['x'].values
    elat = subdata['y'].values
    ez = subdata['z'].values

    return elon, elat, ez

def get_gom_hycom_files(years):
    root_folder = "/nexsan/archive/GOMu0.04_501/data/netcdf"
    all_paths = []
    for c_year in years:
        file_names = os.listdir(join(root_folder, str(c_year)))
        file_paths = [join(root_folder, str(c_year), file_name) for file_name in file_names if file_name.find(".gz") == -1]
        file_paths.sort()
        all_paths.append(file_paths)
    file_paths = reduce(lambda a,b: a+b, all_paths)
    return file_paths

def get_aviso_files(years, root_folder):
    file_names = os.listdir(root_folder)
    file_paths = [join(root_folder,x) for x in file_names if x[:4] in str(list(years))]
    file_paths.sort()
    return file_paths

def read_aviso_data(years, folder):
    file_paths = get_aviso_files(years, folder)  # From aviso

    for idx, c_file in enumerate(file_paths):
        ds_orig = xr.open_dataset(c_file, decode_times=False)
        # Cropping to the Gulf Only
        ds = ds_orig.sel(latitude=slice(17.5, 31), longitude=slice(-100, -78))

        #     ds_crop = ds.sel(lat=slice(24, 30), lon=slice(-84, -78))  # Cropping by value
        if idx == 0:  # Only get lat and lon the first time
            lon = ds.longitude
            lat = ds.latitude
            adt = np.empty((0, len(lat), len(lon)))  # [time, lat, lon]

        adt = np.ma.concatenate((adt, ds.adt), axis=0)  # [time, lat, lon]

    return adt, lat, lon

def read_aviso_data_adtuv(years, folder):
    file_paths = get_aviso_files(years, folder)  # From aviso

    for idx, c_file in enumerate(file_paths):
        ds_orig = xr.open_dataset(c_file, decode_times=False)
        # Cropping to the Gulf Only
        ds = ds_orig.sel(latitude=slice(17.5, 31), longitude=slice(-100, -78))

        #     ds_crop = ds.sel(lat=slice(24, 30), lon=slice(-84, -78))  # Cropping by value
        if idx == 0:  # Only get lat and lon the first time
            lon = ds.longitude
            lat = ds.latitude
            u = np.empty((0, len(lat), len(lon)))  # [time, lat, lon]
            v = np.empty((0, len(lat), len(lon)))  # [time, lat, lon]
            adt = np.empty((0, len(lat), len(lon)))  # [time, lat, lon]

        u = np.ma.concatenate((u, ds.ugos), axis=0)  # [time, lat, lon]
        v = np.ma.concatenate((v, ds.vgos), axis=0)  # [time, lat, lon]
        adt = np.ma.concatenate((adt, ds.adt), axis=0)  # [time, lat, lon]

    return adt, u, v, lat, lon

def change_units(cc, lon, lat):
    """skimage functions extracts the contour but the paths are image indices
    and not (lon, lat)"""
    flon = interp1d(np.arange(0, len(lon)), lon)
    flat = interp1d(np.arange(0, len(lat)), lat)
    for cci in cc:
        cci[:, 0] = flat(cci[:, 0])
        cci[:, 1] = flon(cci[:, 1])
    return cc

def filter_contour2(cc, gom_path):
    cc = filter_gom2(cc, gom_path)
    cc = filter_loop2(cc)
    return cc

def filter_loop2(cc):
    """Remove contour that are eddies"""
    for i in range(len(cc) - 1, -1, -1):
        cci = cc[i]
        if np.linalg.norm(cci[0] - cci[-1]) < 0.5:  # presence of a loop
            del cc[i]
    return cc


def filter_gom2(cc, gom_path):
    """Remove contour that are outside of the Gom"""
    for i in range(len(cc) - 1, -1, -1):
        cci = cc[i]
        if np.all(gom_path.contains_points(np.array([cci[:, 0], cci[:, 1]]).T)):  # carribbean or atlantic ocean
            del cc[i]
        elif np.all(cci[:, 0] < -89):  # contour in the western gom
            del cc[i]
        elif len(cci) < 25:  # non-useful little contour
            del cc[i]
    return cc

def loop_current_extent2(cc, gom_path):
    min_lon = 0
    max_lat = 0
    for cci in cc:
        # points inside the gom and over 21.5°N to avoid always counting the Yucatan corner
        # around the location where the 17-cm contour start
        in_gom = ~gom_path.contains_points(np.array([cci[:, 1], cci[:, 0]]).T)
        over_21 = cci[:, 1] > 21.5
        conditions = np.logical_and(in_gom, over_21)

        # maximum extent
        #  --- VALIDATE THIS STEP TO FOLLOW LITTERATURE ---
        # index = np.argmax(verts[:, 1][conditions])
        # min_lon = min(min_lon, verts[:, 0][conditions][index])
        # max_lat = max(max_lat, verts[:, 1][conditions][index])

        min_lon = min(min_lon, np.min(cci[:, 1][conditions]))
        max_lat = max(max_lat, np.max(cci[:, 0][conditions]))

    return min_lon, max_lat

def filter_contour(cc):
    filter_gom(cc)
    filter_loop(cc)

def filter_loop(cc):
    "Remove contour that are eddies"
    for level in cc.collections:
        for kp, path in reversed(list(enumerate(level.get_paths()))):
            # go in reversed order due to deletions!
            verts = path.vertices  # (N,2)-shape array of contour line coordinates
            # print(kp)
            # print(verts[0])
            # print(verts[-1])
            if np.linalg.norm(verts[0] - verts[-1]) < 0.1:  # presence of a loop
                del (level.get_paths()[kp])
            # this removes little contour next to bnd in gom could be a problem..
            elif len(verts[:, 0]) < 25:
                del (level.get_paths()[kp])


def filter_gom(cc):
    "Remove contour that are outside of the Gom"
    for level in cc.collections:
        for kp, path in reversed(list(enumerate(level.get_paths()))):
            # small contour not in GOM
            verts = path.vertices
            if np.all(gom_path.contains_points(np.array((verts[:, 0], verts[:, 1])).T)):
                del (level.get_paths()[kp])


def loop_current_extent(cc):
    min_lon = 0
    max_lat = 0
    for level in cc.collections:
        for kp, path in list(enumerate(level.get_paths())):
            verts = path.vertices

            # points inside the gom and over 21.5°N to avoid always counting the Yucatan corner
            # around the location where the 17-cm contour start
            in_gom = ~gom_path.contains_points(np.array((verts[:, 0], verts[:, 1])).T)
            over_21 = verts[:, 1] > 21.5
            conditions = np.logical_and(in_gom, over_21)

            # maximum extent
            #  --- VALIDATE THIS STEP TO FOLLOW LITTERATURE ---
            # index = np.argmax(verts[:, 1][conditions])
            # min_lon = min(min_lon, verts[:, 0][conditions][index])
            # max_lat = max(max_lat, verts[:, 1][conditions][index])

            min_lon = min(min_lon, np.min(verts[:, 0][conditions]))
            max_lat = max(max_lat, np.max(verts[:, 1][conditions]))

    return min_lon, max_lat

def filter_contour2(cc, gom_path):
    cc = filter_gom2(cc, gom_path)
    cc = filter_loop2(cc)
    return cc

def filter_loop2(cc):
    """Remove contour that are eddies"""
    for i in range(len(cc) - 1, -1, -1):
        cci = cc[i]
        if np.linalg.norm(cci[0] - cci[-1]) < 0.5:  # presence of a loop
            del cc[i]
    return cc

def filter_gom2(cc, gom_path):
    """Remove contour that are outside of the Gom"""
    for i in range(len(cc) - 1, -1, -1):
        cci = cc[i]
        # print(cci.shape)
        if np.all(gom_path.contains_points(cci)):  # carribbean or atlantic ocean
            del cc[i]
        elif np.all(cci[:, 1] < -89):  # contour in the western gom
            del cc[i]
        elif len(cci) < 25:  # non-useful little contour
            del cc[i]
    return cc

def change_units(cc, lon, lat):
    """skimage functions extracts the contour but the paths are image indices
    and not (lon, lat)"""
    flon = interp1d(np.arange(0,len(lon)), lon)
    flat = interp1d(np.arange(0,len(lat)), lat)
    newcc = []
    for cci in cc:
        try:
            t = np.zeros(cci.shape)
            t[:,0] = flat(cci[:,0])
            t[:,1] = flon(cci[:,1])
            newcc.append(t)
        except Exception as e:
            print(F"Failed for {cci} error: {e}")
    return newcc

def loop_current_extent3(cc, gom_path):
    min_lon = 0
    max_lat = 0
    for cci in cc:
        # points inside the gom and over 21.5°N to avoid always counting the Yucatan corner
        # around the location where the 17-cm contour start
        in_gom = gom_path.contains_points(list(zip(cci[:, 1], cci[:, 0])))
        over_21 = cci[:, 0] > 21.5
        conditions = np.logical_and(in_gom, over_21)
        #
        # maximum extent
        # --- VALIDATE THIS STEP TO FOLLOW LITTERATURE ---
        # index = np.argmax(verts[:, 1][conditions])
        # min_lon = min(min_lon, verts[:, 0][conditions][index])
        # max_lat = max(max_lat, verts[:, 1][conditions][index])

        min_lon = min(min_lon, np.min(cci[:, 1][conditions]))
        max_lat = max(max_lat, np.max(cci[:, 0][conditions]))

    return min_lon, max_lat
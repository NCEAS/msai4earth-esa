import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import rasterio
from rasterio.crs import CRS

# https://stackoverflow.com/questions/43966350/getting-the-maximum-in-each-rolling-window-of-a-2d-numpy-array
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

import pystac_client

# *********************************************************************

def path_to_lidar(year):
msai4earth-esa/iceplant_detection
    # root for all Santa Barbara County canopy height rasters
    root = '/home/jovyan/msai4earth-esa/iceplant_detection/create_train_test_sets/SantaBarbaraCounty_lidar/'
    fp = os.path.join(root, 
                      'SantaBarbaraCounty_lidar_'+str(year)+'.tif')
    return fp

# **********************************************************************************

# raster = numpy aray
# bands = array or 1
def save_raster(raster, fp, shape, bands_n, crs, transform, dtype):
    bands_array = 1
    if bands_n > 1:
        bands_array = np.arange(1,bands_n+1)
        
    with rasterio.open(
        fp,  # file path
        'w',           # w = write
        driver='GTiff', # format
        height = shape[0], 
        width = shape[1],
        count = bands_n,  # number of raster bands in the dataset
        dtype = dtype,
        crs = crs,
        transform = transform,
    ) as dst:
        dst.write(raster.astype(dtype), bands_array)
    return 
    

# ------------------------------------------------------------------------------

# folder_path = path to folder to save rasters
def save_min_max_rasters(rast_reader, folder_path, year):    
    rast = rast_reader.read([1]).squeeze() # read raster

    maxs = maxf2D(rast, size=(3,3)) # calculate min max and difference
    mins = minf2D(rast, size=(3,3))   
    
    # save rasters
    m = [maxs, mins]
    m_labels = ['maxs_', 'mins_']
    for i in range(0,2):
        fp = os.path.join(folder_path, 'lidar_'+m_labels[i]+ str(year)+'.tif')
        save_raster(m[i], 
                    fp, 
                    rast.shape,
                    1,
                    rast_reader.crs, 
                    rast_reader.transform, 
                    rasterio.uint8)
    return

# ------------------------------------------------------------------------------

def save_avg_rasters(rast_reader, folder_path, year):
    rast = rast_reader.read([1]).squeeze() # read raster
    
    # calculate averages
    w = np.ones(9).reshape(3,3)
    avgs = conf2D(rast, 
             weights=w,
             mode='constant')
    avgs = avgs/9
    
    negative_avg = avgs<0
    avgs[negative_avg] = 0
    
    # save averages
    fp = os.path.join(folder_path, 'lidar_avgs_'+ str(year)+'.tif')
    save_raster(avgs, 
                fp, 
                rast.shape, 
                1,
                rast_reader.crs, 
                rast_reader.transform, 
                rasterio.float32)
    return


# **********************************************************************************


# fp = filepath of csv, must have x and y columns representing coordinates of point
# crs must be the crs of the csv coordinates
def geodataframe_from_csv(fp, crs):
    df_raw = pd.read_csv(fp)
    df = df_raw.drop(['geometry','Unnamed: 0'], axis=1)
    
    # recreate geometry column with shapely Points
    xy = []
    for x,y in zip(df.x,df.y):
        xy.append(Point(x,y))
    df['geometry'] = xy
    df = df.drop(['x','y'], axis=1)

    pts = gpd.GeoDataFrame(df, crs=crs)
    return pts

# ------------------------------------------------------------------------------

def pts_for_lidar_sampling(pts, crs):
    pts_xy = pts.to_crs(crs).geometry.apply(lambda p :(p.x, p.y)).to_list()
    return pts_xy

# ------------------------------------------------------------------------------

def sample_raster(pts_xy, raster_reader):
    sample = raster_reader.sample(pts_xy)
    samples = []
    for x in sample:
        samples.append(x[0])
    return samples

# ------------------------------------------------------------------------------

def crs_from_itemid(itemid):
    # accesing Azure storage using pystac client
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # search for naip scene where the pts were sampled from
    search = catalog.search(
        collections=["naip"],
        ids = itemid
    )
    item = list(search.get_items())[0]
    epsg_code = item.properties['proj:epsg']
    return  CRS.from_epsg(epsg_code)

# **********************************************************************************

def open_and_match(fp, reproject_to):
    rast = rioxr.open_rasterio(fp)
    rast_match = rast.rio.reproject_match(reproject_to)
    return rast_match.squeeze()
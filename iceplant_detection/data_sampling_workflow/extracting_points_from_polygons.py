import os
import pandas as pd
import numpy as np
import rasterio
import shapely

from shapely.geometry import shape
from shapely.geometry import Point

import random
random.seed(10)

import warnings

import utility # custom module


# *********************************************************************

def path_to_polygons(aoi, year):
    """
        Creates a path to shapefile with polygons collected at specified aoi and year. 
        The root of the folder containing the polygons is hardcoded inside the function.
            Parameters:
                        aoi (str): name of aoi in polygon's file name
                        year (int): year of collection in polygon's file name
            Return: fp (str): if the file exists it returns the constructed file path
    """
    # root for all polygons collected on naip scenes
    root = '/home/jovyan/msai4earth-esa/iceplant_detection/data_sampling_workflow/polygons_from_naip_images'
    fp = os.path.join(root, 
                      aoi+'_polygons', 
                      aoi+'_polygons_'+str(year), 
                      aoi+'_polygons_'+str(year)+'.shp')

    # check there is a file at filepath
    if not os.path.exists(fp):
        print('invalid filepath: no file')
        return
    
    return fp

# *********************************************************************

# extracts number of random points within polygon
def random_pts_poly(N, polygon):
    """
        Creates a list of N points sampled randomly from within the given polygon.
            Parameters:
                        N (int): number of random points to sample form polygon
                        polygon (shapely.geometry.polygon.Polygon): 
            Return:
                    points (list of shapely.geometry.point.Point): list of N random points sampled from polygons
    """
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < N:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points  


# *********************************************************************

# rast_reader = rasterio reader of naip image
# polys and naip have to be in same crs, do polys.to_crs(naip.crs,inplace=True) before
def num_pts_proportion(polys, rast_reader, proportion):

    # area of a single pixel from raster resolution
    pixel_size = rast_reader.res[0]*rast_reader.res[1]
    
    # calculating how many pixels are there in the polygon (approx)
    # by dividing the area of poly by area of a single pixel
    return polys.geometry.apply(lambda p: int((p.area/pixel_size)*proportion))

# ---------------------------------------------

def num_pts_sliding(polys, alpha, m):
    
    num_random_pts = alpha * polys.pixels.to_numpy()
    num_random_pts = num_random_pts.astype('int32')
    num_random_pts[num_random_pts>m] = m
    
    return num_random_pts

# *********************************************************************

def sample_raster_from_polys_proportion(polys, itemid, proportion):

    item = utility.get_item_from_id(itemid)
    rast_reader = utility.get_raster_from_item(item)
    
    # convert to same crs as raster to properly calculate area of polygons
    polys_match = polys.to_crs(rast_reader.crs)
    
    # area of a single pixel from raster resolution    
    pixel_size = rast_reader.res[0]*rast_reader.res[1]
    
    # calculating how many pixels are there in the polygon (approx)
    # by dividing the area of poly by area of a single pixel
    num_random_pts = polys_match.geometry.apply(lambda p: int((p.area/pixel_size)*proportion))
    
    return sample_naip(polys_match, num_random_pts, rast_reader, item)

# ---------------------------------------------

def sample_naip_from_polys_sliding(polys, itemid, alpha, m):
    item = utility.get_item_from_id(itemid)
    naip = utility.get_raster_from_item(item)
    
    polys_match = polys.to_crs(naip.crs)
    
    pixel_size = naip.res[0]*naip.res[1]
    polys['pixels'] = polys_match.geometry.apply(lambda p: int((p.area/pixel_size)))
#    polys = polys.sort_values(by=['pixels'], ascending=False).reset_index(drop=True)
        
    num_random_pts = num_pts_sliding(polys_match, alpha, m)
    return sample_naip(polys_match, num_random_pts, naip, item)

# ---------------------------------------------

def sample_naip_from_polys_fixed_n(polys, itemid, n):
    item = utility.get_item_from_id(itemid)
    naip = utility.get_raster_from_item(item)
    
    polys_match = polys.to_crs(naip.crs)
    
    num_random_pts = np.full(polys_raw.shape[0],n)
    return sample_naip(polys_match, num_random_pts, naip, item)

# *********************************************************************

# TO DO: maybe this shouldn't catch warnings here, but in notebook
def raster_sample_proportion_no_warnings(polys, itemid, proportion):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_raster_from_polys_proportion(polys, itemid, proportion)
    return df

# ---------------------------------------------

# TO DO: maybe this shouldn't catch warnings here, but in notebook
def naip_sample_n_no_warnings(polys, itemid, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys_fixedn(polys, itemid, n)
    return df

# ---------------------------------------------

# TO DO: maybe this shouldn't catch warnings here, but in notebook
def naip_sample_sliding_no_warnings(polys, itemid, alpha, m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys_sliding(polys, itemid, alpha, m)
    return df


# *********************************************************************
# polys needs to have at least geometry, iceplant and id columns 

def sample_naip(polys, num_random_pts, naip, item):
    samples = []
    for i in range(0,polys.shape[0]):   # for each polygon in set
        poly = polys['geometry'][i]
        N = num_random_pts[i]

        points = random_pts_poly(N,poly)  # select random points within poly
        sample = pd.DataFrame({
            'geometry': pd.Series(points), 
            'iceplant':pd.Series(np.full(N,polys['iceplant'][i])),  # add iceplant Y/N identification for all pts
            'polygon_id':pd.Series(np.full(N,polys['id'][i]))
                     })

        sample['xy'] = sample.geometry.apply(lambda p :(p.x, p.y))  # separate coords (needed for reasterio.io.DatasetReader sample function)
        data_generator = naip.sample(sample.xy)   # extract band values from naip image 
        data = []
        for i in data_generator: 
            data.append(i)
        data = np.vstack(data)
        data = pd.DataFrame(data, columns=['r','g','b','nir']) # create pd DataFrame from np.array
        
        sample = pd.concat([sample,data],axis=1)  # add band data to points
        samples.append(sample)   # add new samples

    df = pd.concat(samples) # create dataframe from samples list
    
    df['x']= df.geometry.apply(lambda p : p.x)   # coordinate cleaning
    df['y']= df.geometry.apply(lambda p : p.y)
    df.drop('xy',axis=1,inplace=True)
    
    df['year'] = item.datetime.year   # add date to samples
    df['month'] = item.datetime.month
    df['day_in_year'] = utility.day_in_year(item.datetime.day, item.datetime.month, item.datetime.year )
    df['naip_id'] = item.id           # add naip item id to samples

    df[['geometry','x','y',
        'iceplant',
        'r','g','b','nir',
        'year','month','day_in_year',
        'naip_id','polygon_id']]
    return df

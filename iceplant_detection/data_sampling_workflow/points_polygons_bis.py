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
    """
        Counts the approximate number of pixels in a raster covered by each polygon in a list.
        No need to match CRS: to do the count it internally matches the CRS of the polygons and the raster. 
            Parameters:
                        polys (geopandas.geodataframe.GeoDataFrame): 
                            GeoDataFrame with geometry column of type shapely.geometry.polygon.Polygon
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster on which we will "overlay" the polygons to count the pixels covered
            Returns:
                    n_pixels (numpy.ndarray): 
                        approximate number of pixels from raster covered by each polygon
            
    """
def count_pixels_in_polygons(polys, rast_reader):
    
    # convert to same crs as raster to properly calculate area of polygons
    if polys.crs != rast_reader.crs:
        print('matched crs')
        polys = polys.to_crs(rast_reader.crs)
    
    # area of a single pixel from raster resolution    
    pixel_size = rast_reader.res[0]*rast_reader.res[1]
    
    # get approx number of pixels by dividing polygon area by pixel_size
    n_pixels = polys.geometry.apply(lambda p: int((p.area/pixel_size)))
    
    return  n_pixels.to_numpy()

# *********************************************************************
def sample_size_in_polygons(n_pixels, param, sample_fraction=0, max_sample=0, const_sample=0):
    """
        Calculates the number of points to sample from each polygon in a list 
        according to the number of pixels covered by a polygon (given) and 
        one of the following sampling count methods:
            - 'fraction': constant fraction of the number of pixels in each polygon
            - 'sliding': constant fraction of the number of pixels in each polygon, up to a maximum number
            - 'constant': constant number of points in each polygon
            
            Parameters:
                       n_pixels (numpy.ndarray):
                           array with (approximate) number of pixels contained in each polygon 
                        param (str):
                            must be 'fraction', 'sliding' or 'constant', 
                            depending on how you want to calculate the number of points to be sampled from each polygon
                        sample_fraction (float in (0,1)): 
                            fraction of points to sample from each polygon
                        max_sample (int): 
                            maximum number of points to sample from each polygon
                        const_sample (int):
                            constant number of points to sample from each polygon
            Returns:
                    n_pts (numpy.ndarray): 
                        array with number of pts to sample from each polygon
    """
    if param not in ['fraction', 'sliding', 'constant']:
        print('not valid parameter: param must be `fraction`, `sliding` or `constant`' 
        return
    # TO DO: add warning for other parameters
                     
    if param == 'fraction':
        n_pts = sample_fraction * n_pixels
    
    elif param == 'sliding':
        n_pts = sample_fraction * n_pixels
        n_pts[n_pts>max_sample] = max_sample
    
    elif param == 'constant':
        # TO DO: add warning not to sample more points than possible
        n_pts = np.full(n_pixels.shape[0],const_sample)
    
    n_pts = n_pts.astype('int')
    return n_pts

# *********************************************************************

def random_pts_poly(N, polygon):
    """
        Creates a list of N points sampled randomly from within the given polygon.
            Parameters:
                        N (int): number of random points to sample form polygon
                        polygon (shapely.geometry.polygon.Polygon): 
            Return:
                    points (list of shapely.geometry.point.Point): 
                        list of N random points sampled from polygons
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

def sample_raster_from_poly(N, poly, poly_id, poly_class, class_name, rast_reader, rast_band_names):
    points = random_pts_poly(N,poly)  # select random points within poly
    sample = pd.DataFrame({
        'geometry': pd.Series(points), 
        class_name : pd.Series(np.full(N,poly_class)),  # add class identification for all pts
        'polygon_id': pd.Series(np.full(N,poly_id))
                 })

    sample_coords = sample.geometry.apply(lambda p: (p.x, p.y))  # separate coords (needed for reasterio.io.DatasetReader sample function)
    data_generator = rast_reader.sample(sample_coords)   # extract band values from naip image 
    data = []    # TO DO: maybe simplify this?
    for i in data_generator: 
        data.append(i)
    data = np.vstack(data)
    data = pd.DataFrame(data, columns=rast_band_names) # create pd DataFrame from np.array

    sample = pd.concat([sample,data],axis=1)  # add band data to points

    sample['x']= sample.geometry.apply(lambda p : p.x)   # coordinate cleaning
    sample['y']= sample.geometry.apply(lambda p : p.y)
    df.drop('geometry',axis=1,inplace=True)

    return sample

# *********************************************************************

def sample_naip(polys, itemid, param, sample_fraction=0, max_sample=0, const_sample=0):
    item = utility.get_item_from_id(itemid)
    rast_reader = utility.get_raster_from_item(item)        
    polys_match = polys.to_crs(rast_reader.crs)
    
    n_pixels = count_pixels_in_polygons(polys_match, rast_reader)
    n_pts = sample_size_in_polygons(n_pixels, param, sample_fraction, max_sample, const_sample)
    
    samples = []
    for i in range(0,polys.shape[0]):   # for each polygon in set
        poly = polys['geometry'][i]    # TO DO: put all these as parameters inside the function
        poly_id = polys['id'][i]
        poly_class = polys['iceplant'][i]
        class_name = 'iceplant'
        N = n_pts[i]
        rast_band_names = ['r','g','b','nir']
        
        sample = sample_raster_from_poly()
        samples.append(sample)   
        
    df = pd.concat(samples) # create dataframe from samples list
    
    df['year'] = item.datetime.year   # add date to samples  TO DO: get from polys? raster?
    df['month'] = item.datetime.month
    df['day_in_year'] = utility.day_in_year(item.datetime.day, item.datetime.month, item.datetime.year )
    df['naip_id'] = item.id           # add naip item id to samples

    df[['x','y',
        'iceplant',
        'r','g','b','nir',
        'year','month','day_in_year',
        'naip_id','polygon_id']]
    
    return df


# *********************************************************************

def naip_sample_sliding_no_warnings(polys, itemid, sample_fraction, max_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        param = 'sliding'
        df = sample_naip(polys, itemid, param, sample_fraction, max_sample)
    return df

# ---------------------------------------------

def raster_sample_proportion_no_warnings(polys, itemid, sample_fraction):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        param = 'fraction'
        df = sample_naip(polys, itemid, param, sample_fraction)
    return df
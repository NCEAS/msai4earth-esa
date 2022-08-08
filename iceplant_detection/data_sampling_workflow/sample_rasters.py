import os
import pandas as pd
import numpy as np


from shapely.geometry import shape
from shapely.geometry import Point

import random
random.seed(10)

import warnings

#import rioxarray as rioxr
import geopandas as gpd
import rasterio
from rasterio.crs import CRS

from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

import utility # custom module

# *********************************************************************
# TO DO: maybe all these path_to functions should be in a different .py

def path_to_lidar(year):
    """
        Creates a path to the Santa Barbara County canopy hieght raster from given year
        The path to the folder containing the polygons is hardcoded inside the function.
            Parameters:
                        year (int): year of canopy height data
            Return: fp (str): returns the file path to the canopy height raster
    """
    # root for all Santa Barbara County canopy height rasters
    root = '/home/jovyan/msai4earth-esa/iceplant_detection/data_sampling_workflow/SantaBarbaraCounty_lidar/'
    fp = os.path.join(root, 
                      'SantaBarbaraCounty_lidar_'+str(year)+'.tif')
    return fp

# ----------------------------------

def path_to_polygons(aoi, year):
    """
        Creates a path to the shapefile with polygons collected at specified aoi and year. 
        The path to the folder containing the polygons is hardcoded inside the function.
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

# ----------------------------------
def path_to_spectral_pts(aoi, year):
    fp = os.path.join(os.getcwd(), 
                             'temp', 
                            aoi + '_points_'+str(year)+'.csv')
    # TO DO: maybe change to _spectral_points
    return fp


# *********************************************************************

def count_pixels_in_polygons(polys, rast_reader):
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
        print('not valid parameter: param must be `fraction`, `sliding` or `constant`')
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
                        polygon (shapely.geometry.polygon.Polygon): polygon from which to sample points
            Return:
                    points (list of shapely.geometry.point.Point): 
                        list of N random points sampled from polygon
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

def sample_raster_from_poly(N, poly, poly_id, class_name, poly_class, rast_reader, rast_band_names, rast_crs):
    """
        Creates a dataframe of raster bands values at N points randomly sampled from a given polygon.
        Polygon and raster must have SAME CRS for results to be correct. 
        Resulting dataframe includes metadata about the sampled polygon: poly_id, poly_class.
        
            Parameters:
                        N (int): 
                            number of random points to sample form polygon
                        poly (shapely.geometry.polygon.Polygon): 
                            polygon from which to sample points
                        poly_id (int): 
                            id number of the polygon
                        class_name (str): 
                            name of the classification in which polygons outline pixels from same class(ex: 'land_cover')
                        poly_class (int): 
                            class of the data represented by polygon (ex: 1 if polygon is building, 2 if it's water, etc)
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to extract band information at every point sampled from polygon
                        rast_band_names (str list):
                            names of the bands of rast_reader
                        rast_crs (str):
                            CRS of rast_reader
            Returns: 
                    sample (pandas.core.frame.DataFrame): data frame of raster bands' values at the N points sampled from poly.

    """
    # TO DO: add catch when polygon and raster do not intersect
    points = random_pts_poly(N,poly)  # select random points inside poly
    sample = pd.DataFrame({           # make data frame with sampled points
        'geometry': pd.Series(points), 
        class_name : pd.Series(np.full(N,poly_class)),  # add class identification for all pts
        'polygon_id': pd.Series(np.full(N,poly_id))
                 })
    
    # TO DO: substitute four lines by sample_raster_from_pts: input= sample.geometry
    # pts_naip = pts.to_crs(rast_reader_NAIP.crs).geometry
    # sample_raster_from_pts(pts_naip, rast_reader_NAIP, ['r','g','b','nir'])
    sample_coords = sample.geometry.apply(lambda p: (p.x, p.y))  # separate coords (needed for reasterio.io.DatasetReader.sample() )
    data_generator = rast_reader.sample(sample_coords)   # extract band values from raster
    data = np.vstack(list(data_generator))               # make band values into dataframe
    data = pd.DataFrame(data, columns=rast_band_names) 

    sample = pd.concat([sample,data],axis=1)  # add band data to sampled points

    sample['x']= sample.geometry.apply(lambda p : p.x)   # coordinate cleaning
    sample['y']= sample.geometry.apply(lambda p : p.y)
    sample.drop('geometry',axis=1,inplace=True)
    
    sample['pts_crs'] =  rast_crs  # add CRS of points
    
    sample = sample[['x','y','pts_crs','polygon_id', class_name] + rast_band_names] # organize columns

    return sample

# *********************************************************************
def sample_naip_from_polys(polys, class_name, itemid, param, sample_fraction=0, max_sample=0, const_sample=0):
    """
        Creates a dataframe of given NAIP scene's bands values at points sampled randomly from polygons in given list.
        Resulting dataframe includes metadata about the sampled polygons and NAIP raster.
        No need to match polygons and raster CRS, this is done internally.
        
        Parameters:
                        polys (geopandas.geodataframe.GeoDataFrame): 
                            GeoDataFrame with geometry column of type shapely.geometry.polygon.Polygon
                            Index must begin at 0.
                        class_name (str): 
                            name of column in polys GeoDataFrame having the classification in which polygons outline pixels from same class (ex: 'land_cover')
                        itemid (str): 
                            the itemid of a single NAIP scene over which the polygons with be "overlayed" to do the data sampling
                        param (str):
                            must be 'fraction', 'sliding' or 'constant', 
                            depending on how you want to calculate the number of points to be sampled from polygons
                        sample_fraction (float in (0,1)): 
                            fraction of points to sample from each polygon
                        max_sample (int): 
                            maximum number of points to sample from each polygon
                        const_sample (int):
                            constant number of points to sample from each polygon
            Return:
                    df (pandas.core.frame.DataFrame): data frame of raster bands' values at points sampled from polys.

    """    
    item = utility.get_item_from_id(itemid)
    
    rast_reader = utility.get_raster_from_item(item)        
    rast_band_names = ['r','g','b','nir']
    rast_crs = rast_reader.crs.to_dict()['init']
    
    polys_match = polys.to_crs(rast_reader.crs)
    
    n_pixels = count_pixels_in_polygons(polys_match, rast_reader)
    n_pts = sample_size_in_polygons(n_pixels, param, sample_fraction, max_sample, const_sample)
    
    samples = []
    for i in range(0,polys.shape[0]):   # for each polygon in list
        sample = sample_raster_from_poly(n_pts[i], 
                                         polys_match.geometry[i], polys.id[i], 
                                         class_name, polys[class_name][i], 
                                         rast_reader, rast_band_names, rast_crs)                                   
        samples.append(sample)   
    df = pd.concat(samples) # create dataframe from samples list
    
    df['year'] = item.datetime.year   # add date to samples  TO DO: get from polys? raster?
    df['month'] = item.datetime.month
    df['day_in_year'] = utility.day_in_year(item.datetime.day, item.datetime.month, item.datetime.year )
    df['naip_id'] = itemid           # add naip item id to samples
    
    return df


# *********************************************************************


def sample_naip_from_polys_no_warnings(polys, class_name, itemid, param, sample_fraction=0, max_sample=0, const_sample=0):
    """
       Runs sample_naip_from_polys function catching the following warning:
                   /srv/conda/envs/notebook/lib/python3.8/site-packages/pandas/core/dtypes/cast.py:122: ShapelyDeprecationWarning: 
                   The array interface is deprecated and will no longer work in Shapely 2.0. 
                   Convert the '.coords' to a numpy array instead. arr = construct_1d_object_array_from_listlike(values)
        # See https://shapely.readthedocs.io/en/stable/migration.html, section Creating NumPy arrays of geometry objects

            Parameters: see parameters for sample_naip_from_polys function
            Return: see return for sample_naip_from_polys function
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys(polys, class_name, itemid, param, sample_fraction, max_sample, const_sample)
    return df


# *********************************************************************
def geodataframe_from_csv(fp, lon_label, lat_label, crs):
    """
        Transforms a csv with longitude and latitude columns into a GeoDataFrame.
            Parameters:
                        fp (str): 
                            File path to csv containing coordinates of points.
                            The coordiantes must be in separate columns. 
                        lon_label (str): 
                            name of column having longitudes of points
                        lat_label (str): 
                            name of column having latitudes of points
                        crs (rasterio.crs.CRS):
                            crs of points in csv. All points must be in the same crs.
            Returns:
                     geopandas.geodataframe.GeoDataFrame:
                        the csv in given file path converted to a GeoDataFrame with geometry column 
                        of type shapely.geometry.Point constructed from longitude and latitude columns
    """
    df = pd.read_csv(fp)
    if 'geometry' in df.columns:           # rename geometry column if it exists
        df = df = df.rename(columns={'geometry': 'geometry_0'})
    
    # recreate geometry column as shapely Points
    xy = []
    for x,y in zip(df[lon_label],df[lat_label]):
        xy.append(Point(x,y))
    df['geometry'] = xy

    return gpd.GeoDataFrame(df, crs=crs)

# *********************************************************************

def sample_raster_from_pts(pts, rast_reader, rast_band_names):
    """
        Creates a dataframe of raster bands values at the given points.
        Points and raster MUST HAVE SAME CRS for results to be correct. 
            Parameters: 
                pts (geopandas.geoseries.GeoSeries): 
                    GeoSeries of the points  (type shapely.geometry.point.Point) where the samples from the rasters will be taken
                rast_reader (rasterio.io.DatasetReader):
                    reader to the raster from which to sample bands
                rast_band_names (str list):
                    names of the bands of rast_reader
            Return:
                samples (pandas.core.frame.DataFrame): data frame of raster bands' values at the given points

    """
    if rast_reader.count != len(rast_band_names):
        print('# band names != # bands in raster')
        return

    # sample
    sample_coords = pts.apply(lambda p :(p.x, p.y))  
    samples_generator = rast_reader.sample(sample_coords)    
    
    # make band values into dataframe
    samples = np.vstack(list(samples_generator))   
    samples = pd.DataFrame(samples, columns=rast_band_names)
    
    return samples

# *********************************************************************


def min_max_rasters(rast_reader, rast_name, n, folder_path=''):  
    rast = rast_reader.read([1]).squeeze() # read raster values

    maxs = maxf2D(rast, size=(n,n))    # calculate min and max in window
    mins = minf2D(rast, size=(n,n))   
    m = [maxs, mins] 
    
    # if needed, create temp directory to save files 
    if not folder_path:    
        folder_path = os.path.join(os.getcwd(),'temp')  
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # parameters for saving
    m_labels = ['_maxs', '_mins']
    dtypes = [rasterio.dtypes.get_minimum_dtype(maxs), rasterio.dtypes.get_minimum_dtype(mins)]
    
    # save rasters
    for i in range(0,2):
        fp = os.path.join(folder_path, rast_name + m_labels[i]+'.tif')
        utility.save_raster(m[i], 
                    fp, 
                    rast.shape,
                    1,
                    rast_reader.crs, 
                    rast_reader.transform, 
                    dtypes[i])  
    return

# ------------------------------------------------------------------------------

def avg_rasters(rast_reader, rast_name, n, folder_path=''):  
    rast = rast_reader.read([1]).squeeze() # read raster values

    w = np.ones(n*n).reshape(n,n)      # calculate averages in window
    avgs = conf2D(rast, 
             weights=w,
             mode='constant')
    avgs = avgs/(n*n)
    
#    negative_avg = avgs<0   # TO DO: NOT SURE IF THIS IS USED DOWNSTREAM 
#    avgs[negative_avg] = 0  # BUT PROBABLY SHOULD NOT BE HERE
    
    # if needed, create temp directory to save files 
    if not folder_path:   
        folder_path = os.path.join(os.getcwd(),'temp')  
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    # parameters for saving   
    fp = os.path.join(folder_path, rast_name +'_avgs.tif')                
    dtype = rasterio.dtypes.get_minimum_dtype(avgs)
            
    utility.save_raster(avgs,    # save rasters
                fp, 
                rast.shape, 
                1,
                rast_reader.crs, 
                rast_reader.transform, 
                dtype)  
    return
                      
                      
# **********************************************************************************

def open_and_match(fp, reproject_to):
    rast = rioxr.open_rasterio(fp)
    rast_match = rast.rio.reproject_match(reproject_to)
    return rast_match.squeeze()


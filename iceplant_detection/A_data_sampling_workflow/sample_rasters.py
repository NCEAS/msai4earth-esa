import os
import pandas as pd
import numpy as np

import pystac_client 
import planetary_computer as pc

import calendar

from shapely.geometry import shape
from shapely.geometry import Point

import random
#random.seed(10)

import warnings

import rioxarray as rioxr
import geopandas as gpd

import rasterio
from rasterio.crs import CRS

from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

from skimage.morphology import disk
from skimage.filters.rank import entropy

# *********************************************************************


def get_item_from_id(itemid):
    """
        Searches the Planetary Computer's NAIP collection for the item associated with the given itemid.
            Parameters:
                        itemid (str): the itemid of a single NAIP scene
            Returns:
                        item (pystac.item.Item): item associated to given itemid (unsigned)
   """
    # accesing Planetary Computer's storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    search = catalog.search(collections=["naip"], ids = itemid)
    
    # return 1st item in search (assumes itemid IS associaed to some item)
    return list(search.get_items())[0]   # ** TO DO: catch exception

# ---------------------------------------------

def get_raster_from_item(item):
    """
        "Opens" the raster in the given item: returns a rasterio.io.DatasetReader to the raster in item.
            Parameters: item (pystac.item.Item)
            Returns: reader (rasterio.io.DatasetReader) 
    """  
    href = pc.sign(item.assets["image"].href)
    reader = rasterio.open(href)
    return reader

# ---------------------------------------------

# TO DO: not used
def get_crs_from_itemid(itemid):
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

# *********************************************************************

def day_in_year(day,month,year):
    """
        Transforms a date into a day in the year from 1 to 365/366 (takes into account leap years).
            Paratmeters:
                day (int): day of the date
                month (int 1-12): month of the date
                year (int): year of date
            Returns:
                n (int): date as day in year
    """
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    n = 0
    for i in range(month-1):
        n = n+days_in_month[i]
    n = n+day
    if calendar.isleap(year) and month>2:
        n = n+1
    return n


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

# N = number of polygons
# S = total number of pts to sample
def staggered_n_samples(N,S):
    if N>S:
        X = np.zeros(N)
        X[:S]+=1
        return X
    
    n = 0     # width of steps
    P_n = S+1 # number of pts needed to assemble the "ladder" with smallest steps
    while P_n>S:
        n +=1
        q = int(N/n)
        r = int(N%n)
        P_n = n*(q*(q+1)/2) + r
        
    X = []
    for i in range(q,0,-1):
        X.append(np.full(n,i))
    X.append(np.full(r,1))
    X = np.concatenate(X)
    
    # distribute remaining points by filling each level from biggest poly to smallest
    R = S - P_n
    qR = int(R/N)
    rR = int(R%N)
    
    X +=qR
    X[:rR]+=1
    return X

# --------------------------------------------------------------

# *********************************************************************

def sample_size_in_polygons(n_pixels, param, sample_fraction=0, max_sample=0, const_sample=0, total_pts = 0):
    """
        Calculates the number of points to sample from each polygon in a list 
        according to the number of pixels covered by a polygon (given) and 
        one of the following sampling count methods:
            - 'fraction': constant fraction of the number of pixels in each polygon
            - 'sliding': constant fraction of the number of pixels in each polygon, up to a maximum number
            - 'constant': constant number of points in each polygon
            - 'proportional': distribute total_pts proportionally across polygons according to polygon area
            
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
    if param not in ['fraction', 'sliding', 'constant', 'staggered']:
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
        n_pts = np.full(n_pixels.shape[0], const_sample)

    elif param == 'proportional':
        total_pixels = np.sum(n_pixels)
        #n_pts = [ total_pts * x/total_pixels for x in n_pixels]
        n_pts = n_pixels / total_pixels * total_pts
    elif 'staggered':
        X = staggered_n_samples(len(n_pixels), total_pts)
        X.sort()
        df = pd.DataFrame({'n_pixels' : n_pixels}).sort_values(by = ['n_pixels'])
        df['n_sample']= X
        df = df.sort_index()
        n_pts = df.n_sample.to_numpy()
    
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
    
    # select random points inside poly
    points = random_pts_poly(N, poly)  
    
    # make data frame with sampled points
    sample = pd.DataFrame({           
        'geometry': pd.Series(points), 
        class_name : pd.Series(np.full(N, poly_class)), 
        'polygon_id': pd.Series(np.full(N, poly_id))
                 })
    # separate coords (needed for rasterio.io.DatasetReader.sample() )
    sample_coords = sample.geometry.apply(lambda p: (p.x, p.y))  
    data_generator = rast_reader.sample(sample_coords)

    # make band values into dataframe
    data = np.vstack(list(data_generator))               
    data = pd.DataFrame(data, columns=rast_band_names) 
    
    # add band data to sampled points
    sample = pd.concat([sample,data],axis=1)  
    
    kwargs = {'x' : sample.geometry.apply(lambda p : p.x),
             'y' : sample.geometry.apply(lambda p : p.y),
             'pts_crs' : rast_crs}
    sample = sample.assign(**kwargs)    
    sample = sample.drop('geometry', axis=1)

    # organize columns
    sample = sample[['x','y','pts_crs','polygon_id', class_name] + rast_band_names] 

    return sample

# *********************************************************************
def sample_naip_from_polys(polys, class_name, itemid, param, sample_fraction=0, max_sample=0, const_sample=0, total_pts=0):
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
    item = get_item_from_id(itemid)
    
    rast_reader = get_raster_from_item(item)        
    rast_band_names = ['r','g','b','nir']
    rast_crs = rast_reader.crs.to_dict()['init']
    
    polys_match = polys.to_crs(rast_reader.crs)
    
    n_pixels = count_pixels_in_polygons(polys_match, rast_reader)
    n_pts = sample_size_in_polygons(n_pixels, param, sample_fraction, max_sample, const_sample, total_pts)
    
    samples = []
    for i in range(polys.shape[0]):
        if n_pts[i]>0:
            sample = sample_raster_from_poly(n_pts[i], 
                                             polys_match.geometry[i], 
                                             polys.id[i], 
                                             class_name, 
                                             polys[class_name][i], 
                                             rast_reader, 
                                             rast_band_names, 
                                             rast_crs)                                   
            samples.append(sample)   
    # create dataframe from samples list        
    df = pd.concat(samples) 
    
    kwargs = {'year' : item.datetime.year,
             'month' : item.datetime.month,
             'day_in_year' : day_in_year(item.datetime.day, item.datetime.month, item.datetime.year),
             'naip_id' : itemid}
    df = df.assign(**kwargs)     
    df = df.reset_index(drop=True)
    
    return df


# *********************************************************************


def sample_naip_from_polys_no_warnings(polys, class_name, itemid, param, sample_fraction=0, max_sample=0, const_sample=0, total_pts=0):
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
        df = sample_naip_from_polys(polys, class_name, itemid, param, sample_fraction, max_sample, const_sample, total_pts)
    return df


# *********************************************************************
def geodataframe_from_csv(df=None, fp=None, lon_label=None, lat_label=None, crs=None):
    """
        Transforms a csv with longitude and latitude columns into a GeoDataFrame.
            Parameters:
                        fp (str): 
                            File path to csv containing coordinates of points.    # TO DO: update to pandas data frame, don't read in file here
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
    if df is None:
        if fp is not None:
            df = pd.read_csv(fp)
        else:
            return False
    # rename geometry column if it exists        
    if 'geometry' in df.columns:          
        df = df.rename(columns = {'geometry': 'geometry_0'})

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_label],df[lat_label]), crs=crs)

# *********************************************************************

def sample_raster_from_pts(pts, rast_reader, rast_band_names):
    """
        Creates a dataframe of raster bands values at the given points.
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
    
    pts_match = pts.to_crs(rast_reader.crs)

    # sample
    sample_coords = pts_match.apply(lambda p :(p.x, p.y))  
    samples_generator = rast_reader.sample(sample_coords)    
    
    # make band values into dataframe
    samples = np.vstack(list(samples_generator))   
    samples = pd.DataFrame(samples, columns=rast_band_names)
    
    return samples

# *********************************************************************

def open_and_match(fp, reproject_to):
    rast = rioxr.open_rasterio(fp)
    rast_match = rast.rio.reproject_match(reproject_to)
    return rast_match.squeeze()

# ------------------------------------------------------------------------------

def input_raster(rast_reader=None, raster=None, band=1, rast_data=None, crs=None, transf=None):
    
    if rast_reader is not None:
        rast = rast_reader.read([band]).squeeze() # read raster values
        crs = rast_reader.crs
        transf = rast_reader.transform
        
    elif raster is not None:
        if len(raster.shape) == 3: # multiband raster            
            rast = raster[band-1].squeeze()
        elif len(raster.shape) == 2: #one band raster
            rast = raster
        crs = raster.rio.crs
        transf = raster.rio.transform()
    
    elif rast_data is not None:
        rast = rast_data
        crs = crs
        transf = transf
    else:
        return 
    
    return rast, crs, transf

# *********************************************************************
def make_directory(dir_name): 
    """ 
        Checks if the directory with name dir_name (str) exists in the current working directory. 
        If it doesn't, it creates the directory and returns the filepath to it.
    """    
    fp = os.path.join(os.getcwd(),dir_name)  
    if not os.path.exists(fp):
        os.mkdir(fp)
    return fp

# ------------------------------------------------------------------------------
def save_raster(raster, fp, shape, bands_n, crs, transform, dtype):
    """
        Saves an array as a 'GTiff' raster with specified parameters.
        Parameters:
                    raster (numpy.ndarray): array of raster values
                    fp (str): file path where raster will be saved, including file name
                    shape (tuple):shape of raster (height, width) TO DO: SHOULD THIS BE READ DIRECTLY FROM raster??
                    bands_n (integer): number of bands in the raster
                    crs (str): CRS of raster
                    transform (affine.Affine): affine transformation of raster
        Return: None
    """
    bands_array = 1
    if bands_n > 1:
        bands_array = np.arange(1,bands_n+1)
        
    with rasterio.open(
        fp,  # file path
        'w',           # w = write
        driver = 'GTiff', # format
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
def save_raster_checkpoints(rast, crs, transf, rast_name=None, suffix= None, folder_path=None):  

    if rast_name is None:
        rast_name = 'raster'        

    if (folder_path is None) or (os.path.exists(folder_path) == False):  
        folder_path = make_directory('temp')  

    if suffix is None:
        suffix = ''
    else:
        suffix = '_'+suffix
        
    fp = os.path.join(folder_path, rast_name + suffix + '.tif')      

    dtype = rasterio.dtypes.get_minimum_dtype(rast)      

    save_raster(rast, 
                fp, 
                rast.shape,
                1,
                crs, 
                transf, 
                dtype) 
    return 

# *********************************************************************

def min_raster(rast_reader=None, raster=None,  rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):  
    """
        Creates a new raster by replacing each pixel p in given raster R by the minimum value in a nxn window centered at p.
        The raster with minimum values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the minimum values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_maxs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute minimum values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)

    if rasterio.dtypes.get_minimum_dtype(rast)  == 'uint8':
        cval = 255
    else:
        cval = 0
        
    mins = minf2D(rast, size=(n,n), cval=cval)     
    
    save_raster_checkpoints(mins, crs, transf, rast_name, 'mins', folder_path)
    return

# ------------------------------------------------------------------------------

def max_raster(rast_reader=None, raster=None,  rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):  
    """
        Creates a new raster by replacing each pixel p in given raster R by the max value in a nxn window centered at p.
        The raster with maximum values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the maximum values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_maxs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute maximum values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)
        
    maxs = maxf2D(rast, size=(n,n))
    
    save_raster_checkpoints(maxs, crs, transf, rast_name, 'maxs', folder_path)
    return

# ------------------------------------------------------------------------------

def avg_raster(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None): 
    """
        Creates a new raster by replacing each pixel p in given raster R by the avg value in a nxn window centered at p.
        The raster with averege values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the average values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_avgs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute average values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)

    w = np.ones(n*n).reshape(n,n)      
    avgs = conf2D(rast, 
             weights=w,
             mode='constant',
             output='int64')
    avgs = avgs/(n**2)

    save_raster_checkpoints(avgs, crs, transf, rast_name, 'avgs', folder_path)            
    return
                      
# ------------------------------------------------------------------------------

def entropy_raster(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=2, folder_path=None): 
    """
        Creates a new raster by replacing each pixel p in given raster R by the entropy value in a disk of radius n centered at p.
        The raster with entropies values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the average values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_avgs.tif.
                        n (int):
                            radius of disk over which to calculate entropy.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)
    
    entropies = entropy(rast, disk(n))    
    
    save_raster_checkpoints(entropies, crs, transf, rast_name, 'entrs', folder_path)            
    return
# ------------------------------------------------------------------------------

def max_min_avg_rasters(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):
    max_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    min_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    avg_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    return

# *********************************************************************
# *********************************************************************
# *********************************************************************
# --- print proportions of ice plant (1) vs no iceplant (0) in an array with only 0 and 1
# TO DO: change to label_proportions
def iceplant_proportions(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('no-iceplant:iceplant ratio    ',round(counts[0]/counts[1],1),':1')
    n = labels.shape[0]
    perc = [round(counts[0]/n*100,2), round(counts[1]/n*100,2)]
    df = pd.DataFrame({'iceplant':unique,
             'counts':counts,
             'percentage':perc}).set_index('iceplant')
    print(df)
    print()
    

# ---------------------------------------------
# rast is 4 band xarray
def ndvi_xarray(rast):
    red_band = rast.sel(band=1).astype('int16') 
    nir_band = rast.sel(band=4).astype('int16')
    return (nir_band - red_band) / (nir_band + red_band)
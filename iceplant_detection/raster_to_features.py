import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import rasterio
import rioxarray as rioxr
import geopandas as gpd

import planetary_computer as pc

import A_data_sampling_workflow.sample_rasters as sr


# **********************************************************************************************************
# **********************************************************************************************************

def rioxr_from_itemid(itemid, reduce_box = None, reduce_box_crs = None):
    """
        Opens the raster associated with the given itemid. 
        If a reduce_box is given, then it opens the subset of raster determined by teh box.
            Parameters: 
                        itemid (str): the itemid of a scene in the planetary computer data repository
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimeter of the area of interest within the scene
                        reduce_box_crs (str):
                            CRS of reduce_box
            Return: 
                    xarray.core.dataarray.DataArray : rioxarray of scene or a subset of it.
    """
    item = sr.get_item_from_id(itemid)    # locate raster
    href = pc.sign(item.assets["image"].href)
    
    rast = rioxr.open_rasterio(href)           # open raster
    
    if reduce_box != None:
        reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs=reduce_box_crs)    # clip if needed
        reduce = reduce.to_crs(rast.rio.crs)        
        rast = rast.rio.clip_box(*reduce.total_bounds)
    
    rast.attrs['datetime'] = item.datetime    # add date of collection
    
    return rast

# **********************************************************************************************************

def raster_as_df(raster, band_names):
    """
        Transforms the given raster into a dataframe of the pixels with column names equal to the band ndames.
             Parameters:
                       raster (numpy.ndarray): raster values
                       band_names (list): names (str) of band names. order of names must be the same order as bands.
            Returns: 
                    df (pandas.core.frame.DataFrame): dataframe where each pixels is a row and columns are the 
                    rasters's band values at pixel
    """  
    pixels = raster.reshape([len(band_names),-1]).T
    df = pd.DataFrame(pixels, columns=band_names) 
    return df

# **********************************************************************************************************
def normalized_difference_index(df, *args):
    """f
        Calculates the normalized difference index of two columns in the given data frame.
        In doing so it converts the column types to int16 (spectral bands are usually uint8).
            Parameters:
                        df (pandas.core.frame.DataFrame): dataframe from which two columns will be used
                            to calculate a normalized difference index
                        *args: tuple of column indices used as x and y in normalized difference
            Returns:
                    pandas.core.series.Series: the normalized difference index of the selected columns
                    
            Example: for dataframe with columns red, green, blue, nir (in that order)
                     ndvi would be normalized_difference_index(df, 3,0), and
                     ndwi would be normalized_difference_index(df, 1,3)
    """    
    m = args[0]
    n = args[1]
    
    x = df.iloc[:, m].astype('int16')  
    y = df.iloc[:, n].astype('int16')
    return (x-y) / (x+y)

# **********************************************************************************************************

def feature_df_treshold(df, feature_name, thresh, keep_gr, func, *args):
    """
        Adds a new column C to a dataframe using the action of a function and 
        selects only the rows that whose values in C are above a certain threshold.
            Parameters: 
                        df (pandas.core.frame.DataFrame): data frame on which to do the operation
                        feature_name (str): name of new column
                        thresh (float): threshold for new column
                        keep_gr (bool): if keep_gr == True then it keeps the rows with new_column > thresh
                                        if keep_gr == False then it keeps the rows with new_column < thresh
                        func (function): function to calculate new column in dataframe
                        *args: arguments for function 
            Returns:
                    keep (pandas.core.frame.DataFrame):
                        a copy of dataframe with the values of function as a new column and subsetted by threshold
                    deleted_indices (numpy.ndarray): 
                        indices of the rows that were deleted from df 
                        (those with value of function not compatible with threshold condition)s
                        
    """
    # add new column
    kwargs = {feature_name : func(df, *args)}        # TO DO: maybe take these two lines out?
    df = df.assign(**kwargs)
    
    # select rows above threshold, keep indices of deleted rows
    if keep_gr == True:
        keep = df[df[feature_name] > thresh]
        deleted_indices = df[df[feature_name] <= thresh].index
    # select rows below threshold, keep indices of deleted rows
    else : 
        keep = df[df[feature_name] < thresh]
        deleted_indices = df[df[feature_name] >= thresh].index
        
    deleted_indices = deleted_indices.to_numpy()
    
    return keep, deleted_indices

# **********************************************************************************************************

def add_spectral_features(df, ndwi_thresh, ndvi_thresh):
    """
       Finds the rows in df with ndwi values below ndwi_thresh and ndvi values above ndvi_thresh. 
       Keeps track of the rows deleted.
           Parameters:
                       df (pandas.core.frame.DataFrame): dataframe with columns red, green, blue and nir (in that order)
                       ndwi_tresh (float): threshold for ndwi
                       ndvi_tresh (float): threshold for ndvi
           Returns: 
                    is_veg (pandas.core.frame.DataFrame):
                        subset of df in which all rows have ndwi values below ndwi_thresh and ndvi values above ndvi_thresh
                    water_index (numpy.ndarray): 
                        indices of rows in df with ndwi values above ndwi_thresh 
                    not_veg_index (numpy.ndarray): 
                        indices of rows in df with ndwi values below ndwi_thresh and ndvi values below ndvi_thresh
    """
    # remove water pixels
    not_water, water_index = feature_df_treshold(df, 
                                             'ndwi', ndwi_thresh, False, 
                                             normalized_difference_index, 1,3)   
    # remove non-vegetation pixels
    is_veg, not_veg_index = feature_df_treshold(not_water, 
                                                   'ndvi', ndvi_thresh, True, 
                                                   normalized_difference_index, 3,0)
    # return pixels that are vegetation and are not water
    # return indices of water pixels and not vegetation pixels
    return is_veg, water_index, not_veg_index

# ----------------------------------------------------

def add_date_features(df, date): 
    """
        Adds three constant columns to the data frame df with info from date (datetime.datetime): year, month and day_in_year.
    """
    kwargs = {'year' : date.year,
             'month' : date.month,
             'day_in_year' : sr.day_in_year(date.day, date.month, date.year)}
    
    return df.assign(**kwargs)

# **********************************************************************************************************

def indices_to_image(nrows, ncols, indices_list, values, back_value):
    """
        Parameters:
                    nrows (int): number of rows in ouput array
                    ncols (int): number of columns in output array
                    indices_list (list): 
                        list of 1-dimensional np.arrays. 
                        each element in list must be values within [0, nrows*ncols-1], 
                        these represent cells in the output array with same value
                    values (list): the value to assign to each of the arrays in indices_list
                    back_value (int): value of any cell not in the union of indices_list
            Returns:
                reconstruct (numpy.ndarray):
                    array with values in valuesU{back_value} with dimensions nrows*ncols 
                    the cells with 'index' in the array indices_list[i] get assigned the value values[i]
                    the index of a i,j cell of the array is i*nrows+j
            Example:
                Suppose nrows=ncols=3, indices_list = [[2,3,7],[0,1,8]], values = [1,2] and back_value =0. 
                Then the output is the array 
                |2|2|1|
                |1|0|0|
                |0|1|2|
                        
    """
    # background, any pixel not in the union of indices will be given this value
    reconstruct = np.ones((nrows,ncols))*back_value 

    # TO DO: check indices list and values lengths are the same?
    for k in range(0,len(indices_list)):
        i = indices_list[k] / ncols
        i = i.astype(int)
        j = indices_list[k] % ncols
        reconstruct[i,j] = values[k]
    
    return reconstruct

# **********************************************************************************************************

def create_aux_canopyheight_rasters(year):
    # open canopy height raster for given year
    lidar_rast_reader = rasterio.open(sr.path_to_lidar(year))   
    # name of output canopy height raster
    rast_name = 'SB_canopy_height_' + str(year) 

    # if there is no temp folder, create one
    temp_fp = os.path.join(os.getcwd(), 'temp') 
    if not os.path.exists(temp_fp):
        os.mkdir(temp_fp)
    temp_fp = os.path.join(temp_fp, 'aux_canopy_height')
    if not os.path.exists(temp_fp):
        os.mkdir(temp_fp)

    # list of file paths to aux canopy height rasters
    # order of filepaths is: lidar, max, min, avg
    lidar_fps = [sr.path_to_lidar(year)]  
    for tag in ['_maxs', '_mins', '_avgs']:
        lidar_fps.append(os.path.join(temp_fp, rast_name + tag + '.tif'))

    # create any missing aux canopy height rasters
    if not all([os.path.exists(fp) for fp in lidar_fps]):
        # save aux rasters in temp folder
        if not os.path.exists(lidar_fps[1]):  # starts at 1 bc 0 is canopy height raster
            sr.max_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)

        if not os.path.exists(lidar_fps[2]):
            sr.min_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)  

        if not os.path.exists(lidar_fps[3]):
            sr.avg_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)
    return lidar_fps

# **********************************************************************************************************
def finish_processing(status, processed, reason, times_pre, times_class, times_post, veg_pixels, itemid):
    
    processed.append('N')
    times_pre.append(0)
    times_class.append(0)        
    times_post.append(0)
    veg_pixels.append(0)
    
    if status == 'no_data':
        reason.append('no data in intersection')
    elif status == 'no_veg':
        reason.append('no vegeatation in intersection')
    else: 
        reason.append('invalid status') 
    
    return
    
def finish_processing_message(status, itemid):
    if status == 'no_data':
        print('no data at intersection of scene with coastal buffer')  
    elif status == 'no_veg':
        print('no vegetation pixels at intersection of scene data with coastal buffer')
    else: 
        print('invalid status')
        return 
    
    print('FINISHED: ', itemid , '\n', end="\r")
    return
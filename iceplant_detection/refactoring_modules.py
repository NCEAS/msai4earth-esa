import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
import rioxarray as rioxr
import geopandas as gpd

import planetary_computer as pc

import data_sampling_workflow.utility as utility

# **********************************************************************************************************
# **********************************************************************************************************

def rioxr_from_itemid(itemid, reduce_box = False, reduce_box_crs = False):
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
    item = utility.get_item_from_id(itemid)    # locate raster
    href = pc.sign(item.assets["image"].href)
    
    rast = rioxr.open_rasterio(href)           # open raster
    
    if reduce_box != False:
        reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs=reduce_box_crs)    # clip if needed
        reduce = reduce.to_crs(rast.rio.crs)        
        rast = rast.rio.clip_box(*reduce.total_bounds)
    
    rast.attrs['datetime'] = item.datetime    # add date of collection
    
    return rast

# **********************************************************************************************************

#raster = nd.array
def raster_as_df(raster, band_names):
    """
             Parameters:
                       raster (numpy.ndarray): # bands is 1st in shape
                       band_names (list): 
            Returns: 
                    df (pandas.core.frame.DataFrame):
    """  
    pixels = raster.reshape([len(band_names),-1]).T
    df = pd.DataFrame(pixels, columns=band_names) 
    return df

# **********************************************************************************************************
def normalized_difference_index(df, *args):
    """
    
        Returns:
                pandas.core.series.Series
    """    
    m = args[0]
    n = args[1]
    
    x = df.iloc[:, m].astype('int16')  
    y = df.iloc[:, n].astype('int16')
    return (x-y) / (x+y)

# **********************************************************************************************************

def feature_df_treshold(df, feature_name, thresh, keep_gr, func, *args):
    
    df[feature_name] = func(df, *args)
    
    if keep_gr == True:
        keep = df[df[feature_name] > thresh]
        deleted_indices = df[df[feature_name] <= thresh].index
    else : 
        keep = df[df[feature_name] < thresh]
        deleted_indices = df[df[feature_name] >= thresh].index
        
    deleted_indices = deleted_indices.to_numpy()
    
    return keep, deleted_indices

# **********************************************************************************************************

def add_date_features(df, date): 
    df['year'] = date.year
    df['month'] = date.month
    df['day_in_year'] = utility.day_in_year(date.day, date.month, date.year)
    return df

# **********************************************************************************************************

def indices_to_image(nrows, ncols, indices_list, values, back_value):
    # background, any pixel not in the union of indices will be given this value
    reconstruct = np.ones((nrows,ncols))*back_value 

    # TO DO: check indices list and values lengths are the same?
    for k in range(0,len(indices_list)):
        i = indices_list[k] / ncols
        i = i.astype(int)
        j = indices_list[k] % ncols
        reconstruct[i,j] = values[k]
    
    return reconstruct


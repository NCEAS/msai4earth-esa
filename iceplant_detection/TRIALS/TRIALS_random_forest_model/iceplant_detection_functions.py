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

def href_and_window(itemid, reduce_box):
    """
        For the raster associated with itemid, returns location (href) and view onto subset (win) specified by reduce_box. 
        This is an internal function used by other functions in this module.
            Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
            Returns: 
                        href (str): 
                            location of raster associated with itemid
                        win (rasterio.windows.Window): 
                            View onto the rectangular subset of raster associated with itemid, 
                            subset is specified by reduce_box. 
                            See https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
    """  
    item = utility.get_item_from_id(itemid)
    # sign and open item
    href = pc.sign(item.assets["image"].href)  # should use item, not itemid
    reader = rasterio.open(href)

    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326") 
    reduce = reduce.to_crs(reader.crs)
    win = reader.window(*reduce.total_bounds)
    return href, win

# ---------------------------------

def open_window_in_scene(itemid, reduce_box):
    """
        Extract array with raster values (all bands) in the subset of the NAIP scene with itemid outliend by reduce_box.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
            Returns: 
                        numpy.ndarray: 
                            raster values of all the bands (r,g,b,nir) in the subset of NAIP scene.
    """             
    href, win = href_and_window(itemid, reduce_box)         # should use item, not itemid
    return rasterio.open(href).read([1,2,3,4], window=win)

# ---------------------------------

def rgb_window_in_scene(itemid, reduce_box):
    """
        Extract array with raster values (only red, green and blue bands) in the subset of the NAIP scene with itemid outliend by reduce_box.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
            Returns: 
                        numpy.ndarray: 
                            raster values of the red, green and blue bands in the subset of NAIP scene.
    """   
    href, win = href_and_window(itemid, reduce_box)
    return rasterio.open(href).read([1,2,3], window=win)

# ---------------------------------

# ** TO DO: give this a better name
def small_raster(itemid, reduce_box):
    """
        Extract a subset of a NAIP scene as a rioxarray raster.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimeter of the area that will be extracted from the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
            Returns: rast_small (xarray.core.dataarray.DataArray): 
                            subset of NAIP scene with itemid outlined by reduce_box as rioxarray raster
    """   
    item = utility.get_item_from_id(itemid)
    href = pc.sign(item.assets["image"].href)
    
    rast = rioxr.open_rasterio(href)
    
    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326")
    reduce = reduce.to_crs(rast.rio.crs)    
    
    rast_small = rast.rio.clip_box(*reduce.total_bounds)
    return rast_small

# ---------------------------------

def plot_window_in_scene(itemid, reduce_box, figsize=15):
    """
        Plots a rectangular subset of a specified NAIP scene.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
                        figsize (int): size of graph
            Returns: None.
    """             
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(np.moveaxis(rgb_window_in_scene(itemid, reduce_box),0,-1))
    plt.show()
    return

# ---------------------------------

# # ** TO DO: probably delete?
# def predict_over_subset(itemid, reduce_box, model):
#     """
#         Apply a classification model (trained on 4 features: red, green, blue and nir values of pixels) to each pixel within a rectangular subset of a specified NAIP scene.
#              Parameters:
#                         itemid (str): 
#                             the itemid of a single NAIP scene
#                         reduce_box (shapely.geometry.polygon.Polygon): 
#                             box outlining the perimter of the area of interest within the NAIP scene with itemid.
#                             Coordiantes of the box's vertices must be given in EPSG:4326 crs.
#                         model (sklearn.ensemble.RandomForestClassifier):
#                             classification model trained on 4 features: red, green, blue and nir values of pixels. 
#                             Features must be in that order.
#             Returns: 
#                          numpy.ndarray: 
#                              2D array with the classifications of each pixel in the subset of NAIP raster in itemid outlined by reduce_box
#     """
#     image = open_window_in_scene(itemid, reduce_box)
#     # reshape image into a np.array where each row is a pixel and the columns are the bands
#     pixels = image.reshape([4,-1]).T
#     predictions_class = model.predict(pixels)
#     # turn back into original raster dimensions
#     return predictions_class.reshape([image.shape[1],-1])

# ---------------------------------

# # ** TO DO: probably delete?
# # ** TO DO: change name to specify it's over a subset
# # ** TO DO: shouldn't be a different function predictions over whole NAIP scene
# def mask_ndvi_and_predict(itemid, reduce_box, model, thresh=0.05): 

#     image = open_window_in_scene(itemid, reduce_box)
#     veg = select_ndvi_df(image, thresh)
#     index = veg.index
#     features = np.array(veg)
    
#     # get predictions from model and make them into a df
#     predictions_class = model.predict(features)
#     c = {'prediction':predictions_class}
#     df = pd.DataFrame(c, index = index)
    
#     # transform predictions df back into binary image 
#     nrows = image.shape[1]
#     ncols = image.shape[2]
#     index = df[df.prediction == 1].index.to_numpy()
    
#     return indices_backto_image(nrows, ncols, index)

# ---------------------------------

def plot_preds_vs_original(itemid, reduce_box, predictions, year, model_name = ' ', figsize=(30,40)):
    """
        Plots a rectangular subset of a specified NAIP scene next to the per-pixel-classifications given my model.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
                        predictions (numpy.ndarray): 
                            2D array with the classifications of each pixel in the subset of NAIP raster in itemid outlined by reduce_box
                        year (int): 
                            year when NAIP scene was collected
                        model_name (str): 
                            name of model to appear on graph's title
                        figsize (int tuple): 
                            size of graph
                        
            Returns: None.
    """  
    original = np.moveaxis(rgb_window_in_scene(itemid, reduce_box),0,-1)
    fig, ax = plt.subplots(1,2,figsize=figsize)

    ax[0].imshow(predictions)
    ax[0].set_title("PREDICTIONS NAIP"+str(year)+ " - " + model_name)

    ax[1].imshow(original)
    ax[1].set_title("NAIP "+str(year)+" original image")

    plt.show()
    return


# **********************************************************************************************************
# **********************************************************************************************************
 
def ndvi(image):
    """
        Pixel-by-pixel NDVI calculation of an image with four bands.
            Parameters:
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order.
            Returns: 
                    numpy.ndarray: 
                        array of size m,n, the result of calculating the NDVI to image pixel-by-pixel
    """ 
    x = image.astype('int16')  # TO DO: add sth in documentation about data type 
    return (x[3,...] - x[0,...])/(x[3,...] + x[0,...])

# ---------------------------------
def ndwi(image):
    """
        Pixel-by-pixel NDWI calculation of an image with four bands.
            Parameters:
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order.
            Returns: 
                    numpy.ndarray: 
                        array of size m,n, the result of calculating the NDWI to image pixel-by-pixel
    """ 
    x = image.astype('int16')
    return (x[1,...] - x[3,...])/(x[1,...] + x[3,...])

# ---------------------------------
def ndvi_thresh(image, thresh=0.05):
    # TO DO: change to apply ndvi or ndwi
    """
        Identifies which pixels in a 4-band image have NDVI above a given threshold.
            Parameters:
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order.
                        thresh (float in (-1,1)): 
                            NDVI threshold
            Returns: 
                    numpy.ndarray: 
                        array of size m,n in which pixels of image with ndvi<thresh have 0 value and pixels with ndvi>=thresh have value 1.
    """  
    x = ndvi(image)
    low_ndvi = x<thresh
    x[low_ndvi] = 0
    x[~low_ndvi] = 1
    return x

# ---------------------------------
# # TO DO: MAYBE DELETE?? seems like we only need the previous one
# def select_ndvi_image(itemid, reduce_box, thresh=0.05):
#     """
#         Identifies which pixels in a rectangular subset of a specified NAIP scene have NDVI above a given threshold.
#              Parameters:
#                         itemid (str): 
#                             the itemid of a single NAIP scene
#                         reduce_box (shapely.geometry.polygon.Polygon): 
#                             box outlining the perimter of the area of interest within the NAIP scene with itemid.
#                             Coordiantes of the box's vertices must be given in EPSG:4326 crs.
#                         thresh (float in (-1,1)): 
#                             NDVI threshold
#             Returns: 
#                     numpy.ndarray: 
#                         array of size m,n in which pixels of image with ndvi<thresh have 0 value and pixels with ndvi>=thresh have value 1.
#     """  
#     image = open_window_in_scene(itemid, reduce_box)
#     return ndvi_thresh(image,thresh)


# ---------------------------------
def select_ndvi_df(image, thresh=0.05):
    """
        Identifies which pixels in a 4-band image have NDVI above a given threshold and returns information as data frame.
             Parameters:
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order.
                        thresh (float in (-1,1)): 
                            NDVI threshold
            Returns: 
                        above_thresh (pandas.DataFrame): 
                            A data frame with all the pixels in image having ndvi>thresh.
                            Each pixel is a row in the data frame and the columns contain a pixel's spectral information.
                            Data frame has five columns: red, green, blue, nir and NDVI
    """  
    pixels = image.reshape([4,-1]).T
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    
    x = ndvi(image)
    df['ndvi'] = x.reshape(x.shape[0]*x.shape[1])
    
    above_thresh = df[df.ndvi>thresh]
    # TO DO: ?? what's this>>
    #vegetation.drop(labels=['ndvi'], axis=1, inplace=True)  # this is uncommented for TRIALS_model_with_lidar
    return above_thresh


# **********************************************************************************************************
# **********************************************************************************************************
def spectral_df(image):
    """
        Transforms a 4-band image into a data frame with columns red, green, blue, nir and NDVI
            Parameters:
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order.def 
            Returns:
                        df (pandas.DataFrame):
                            A data frame containing the spectral information of every pixel in the image as rows.
                            Data frame has five columns: red, green, blue, nir and NDVI
    """
    pixels = image.reshape([4,-1]).T  
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    
    x = ndvi(image)    # add ndvi
    df['ndvi'] = x.reshape(x.shape[0]*x.shape[1])
    return df

# ---------------------------------

# TO DO: unsure this is ever used
def add_date_features(df, item):
    """
        Adds a NAIP scene's date information as columns to a data frame.
            Parameters:
                        item (pystac.item.Item): 
                            item associated to a NAIP image 
                        df (pandas.DataFrame): 
                            within the workflow this is a data frame in which each row is data for a pixel in image. 
            Returns:
                        df (pandas.DataFrame):
                            a copy of the given data frame augmented with three columns with constant values.
                            the columns contain information about the date of collection of the NAIP scene given by item.
                            the columsn are: 
                                year (int: year of scene colletion), month (int: month of scene collection), 
                                and day_in_year (int from 1 to 365: day in year which scene was collected)
    """    
    df['year'] = item.datetime.year
    df['month'] = item.datetime.month
    df['day_in_year'] = utility.day_in_year(item.datetime.day, item.datetime.month, item.datetime.year)
    return df

# ---------------------------------

# TO DO: probably should not mask ndvi here?
# TO DO: move item after image
# TO DO: pass date features as paramenters, don't collect from items
# TO DO: maybe just splig into three functions? select ndvi and add_date_features
def features_over_aoi(item, image, thresh=0.05):
    """
       Selects which pixels in a 4-band image have NDVI above a given threshold and organizes them as a dataframe in which each row is the spectral and date-of-collection information of the pixel.
            Parameters:
                        item (pystac.item.Item): 
                            item associated to the NAIP scene containing image
                        image (numpy.ndarray): 
                            a (4,m,n) array. The first bands are red, green, blue, and NIR, in that specific order
                        thresh (float in (-1,1)): 
                            NDVI threshold
            Returns:
                        df (pandas.DataFrame): 
                            A data frame with all the pixels in image having ndvi>thresh.
                            Each pixel is a row in the data frame and the columns contain a pixel's spectral and date-of collection information.
                            Data frame has 8 columns (in this order): 
                                red, green, blue, nir, NDVI, year, month and day_in_year
    """
    df = select_ndvi_df(image, thresh)
    
    df['year'] = item.datetime.year
    df['month'] = item.datetime.month
    df['day_in_year'] = utility.day_in_year(item.datetime.day, item.datetime.month, item.datetime.year)

    # order features
    df = df[['r','g','b','nir','ndvi','year','month','day_in_year']] 
    return df


# 
# **********************************************************************************************************
# **********************************************************************************************************

def indices_backto_image(nrows, ncols, index):
    """
        Creates a binary array of dimensions nrows*ncols in which cells with value of 1 correspond to the given list of indices.
            Parameters:
                nrows (int): number of rows in output array
                ncols (int): number of columns in output array
                index (array of ints): must be values within [0, nrows*ncols-1]
            Returns:
                reconstruct (numpy.ndarray):
                    array with values 0 and 1 of dimensions nrows*ncols in which cells with value of 1 correspond to the given list of indices.
            Example:
                Suppose nrows=ncols=3 and index = [2,3,7]. Then the output is the array 
                |0|0|1|
                |1|0|0|
                |0|1|0|
    """
    # transform indices to coordinates
    i = index / ncols
    i = i.astype(int)
    j = index % ncols
    
    # fill in array with 1 on index's coordinates 0 elsewhere
    reconstruct = np.zeros((nrows,ncols))
    reconstruct[i,j] = 1
    return reconstruct

# **********************************************************************************************************
# **********************************************************************************************************

# TO DO: FIGURE OUT IF WE WILL KEEP THIS OR SPLIT IT INTO SIMPLER FUNCTIONS
# TO DO: add water category?
def preds_to_image_3labels(nrows, ncols, index, predictions):
    preds = pd.DataFrame(predictions, 
                         columns=['is_iceplant'], 
                         index = index)
    is_iceplant_index = preds[preds.is_iceplant == 1].index
    non_iceplant_index = preds[preds.is_iceplant == 0].index
    
    # initialize raster
    reconstruct = np.ones((nrows,ncols))*2 # 2 = poins that did not go into model

    # 1 = classified as iceplant
    i = is_iceplant_index / ncols
    i = i.astype(int)
    j = is_iceplant_index % ncols
    reconstruct[i,j] = 1

    # 0 = classified as not iceplant
    i = non_iceplant_index / ncols
    i = i.astype(int)
    j = non_iceplant_index % ncols
    reconstruct[i,j] = 0

    return reconstruct

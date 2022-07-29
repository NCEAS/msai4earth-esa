import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
import rioxarray as rioxr
import geopandas as gpd

import pystac_client 
import planetary_computer as pc

import calendar

# **********************************************************************************************************
# **********************************************************************************************************

# SAME AS IN POINTS FORM POLYGONS 
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

    search = catalog.search(
        collections=["naip"],
        ids = itemid)
    
    # return 1st item in search (assumes itemid IS associaed to some item)
    item = list(search.get_items())[0]   # ** TO DO: catch exception
    return item

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
    item = get_item_from_id(itemid)
    # sign and open item
    href = pc.sign(item.assets["image"].href)
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
                            aster values of all the bands (r,g,b,nir) in the subset of NAIP scene.
    """             
    href, win = href_and_window(itemid, reduce_box)
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
    item = get_item_from_id(itemid)
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

# ** TO DO: probably delete?
def predict_over_subset(itemid, reduce_box, model):
    """
        Apply a classification model (trained on 4 features: red, green, blue and nir values of pixels) to each pixel within a rectangular subset of a specified NAIP scene.
             Parameters:
                        itemid (str): 
                            the itemid of a single NAIP scene
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimter of the area of interest within the NAIP scene with itemid.
                            Coordiantes of the box's vertices must be given in EPSG:4326 crs.
                        model (sklearn.ensemble.RandomForestClassifier):
                            classification model trained on 4 features: red, green, blue and nir values of pixels. 
                            Features must be in that order.
            Returns: 
                         numpy.ndarray: 
                             2D array with the classifications of each pixel in the subset of NAIP raster in itemid outlined by reduce_box
    """
    image = open_window_in_scene(itemid, reduce_box)
    # reshape image into a np.array where each row is a pixel and the columns are the bands
    pixels = image.reshape([4,-1]).T
    predictions_class = model.predict(pixels)
    # turn back into original raster dimensions
    return predictions_class.reshape([image.shape[1],-1])

# ---------------------------------

# ** TO DO: probably delete?
# ** TO DO: change name to specify it's over a subset
# ** TO DO: shouldn't be a different function predictions over whole NAIP scene
def mask_ndvi_and_predict(itemid, reduce_box, model, thresh=0.05): 

    image = open_window_in_scene(itemid, reduce_box)
    veg = select_ndvi_df(image, thresh)
    index = veg.index
    features = np.array(veg)
    
    # get predictions from model and make them into a df
    predictions_class = model.predict(features)
    c = {'prediction':predictions_class}
    df = pd.DataFrame(c, index = index)
    
    # transform predictions df back into binary image 
    nrows = image.shape[1]
    ncols = image.shape[2]
    index = df[df.prediction == 1].index.to_numpy()
    
    return indices_backto_image(nrows, ncols, index)

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

# image is a (4,m,n) np array in which bands are r,g,b,nir
def ndvi(image):
    x = image.astype('int16')
    return (x[3,...] - x[0,...])/(x[3,...] + x[0,...])

# ---------------------------------

def ndvi_thresh(image, thresh=0.05):
    x = ndvi(image)
    low_ndvi = x<thresh
    x[low_ndvi] = 0
    x[~low_ndvi] = 1
    return x

# ---------------------------------

def select_ndvi_image(itemid, reduce_box, thresh=0.05):
    image = open_window_in_scene(itemid, reduce_box)
    return ndvi_thresh(image,thresh)


# **********************************************************************************************************
# **********************************************************************************************************

# image is a (4,m,n) np array in which bands are r,g,b,nir
def spectral_df(image):
    pixels = image.reshape([4,-1]).T
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    
    x = ndvi(image)
    df['ndvi'] = x.reshape(x.shape[0]*x.shape[1])
    return df
# ---------------------------------

def add_date_features(df, item):
        df['year'] = item.datetime.year
        df['month'] = item.datetime.month
        df['day_in_year'] = day_in_year(item.datetime.day, item.datetime.month, item.datetime.year)
        return df
    
# ---------------------------------

# TO DO: probably should not mask ndvi here
def features_over_aoi(item, image, thresh=0.05):

    veg = select_ndvi_df(image, thresh)
    
#    veg['ndvi']=(veg.nir.astype('int16') - veg.r.astype('int16'))/(veg.nir.astype('int16') + veg.r.astype('int16'))

    veg['year'] = item.datetime.year
    veg['month'] = item.datetime.month
    veg['day_in_year'] = day_in_year(item.datetime.day, item.datetime.month, item.datetime.year)

    # order features
    veg = veg[['r','g','b','nir','ndvi','year','month','day_in_year']] 
    return veg

# ---------------------------------

def select_ndvi_df(image, thresh=0.05):
    pixels = image.reshape([4,-1]).T
    df = pd.DataFrame(pixels, columns=['r','g','b','nir'])
    
    x = ndvi(image)
    df['ndvi'] = x.reshape(x.shape[0]*x.shape[1])
    
    vegetation = df[df.ndvi>thresh]
    #vegetation.drop(labels=['ndvi'], axis=1, inplace=True)  # this is uncommented for TRIALS_model_with_lidar
    return vegetation

# ---------------------------------

def indices_backto_image(nrows, ncols, index):
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

def day_in_year(day,month,year):
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    n = 0
    for i in range(0,month-1):
        n = n+days_in_month[i]
    n = n+day
    if calendar.isleap(year) and month>2:
        n = n+1
    return n

# **********************************************************************************************************
# **********************************************************************************************************


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
    reconstruct[i,j] = 2

    return reconstruct

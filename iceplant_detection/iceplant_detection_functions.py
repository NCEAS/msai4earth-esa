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
    # accesing Azure storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    search = catalog.search(
        collections=["naip"],
        ids = itemid)
    
    item = list(search.get_items())[0]
    # sign and open item
    return item

# ---------------------------------------------

def get_raster_from_item(item):
    href = pc.sign(item.assets["image"].href)
    ds = rasterio.open(href)
    return ds

# ---------------------------------

def href_and_window(itemid, reduce_box):
    item = get_item_from_id(itemid)
    # sign and open item
    href = pc.sign(item.assets["image"].href)
    ds = rasterio.open(href)

    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326")
    reduce = reduce.to_crs(ds.crs)
    win = ds.window(*reduce.total_bounds)
    return href, win

# **********************************************************************************************************
# **********************************************************************************************************


def small_raster(href, reduce_box):
    item = get_item_from_id(itemid)
    # sign and open item
    href = pc.sign(item.assets["image"].href)
    
    rioxr.open_rasterio(href)
    rgb_small = rgb.rio.clip_box(*reduce.total_bounds)
    return rgb_small

# *******************************************************************************************************


# ---------------------------------

def open_window_in_scene(itemid, reduce_box):
    href, win = href_and_window(itemid, reduce_box)
    return rasterio.open(href).read([1,2,3,4], window=win)

# ---------------------------------

def rgb_window_in_scene(itemid, reduce_box):
    href, win = href_and_window(itemid, reduce_box)   
    return rasterio.open(href).read([1,2,3], window=win)

# **********************************************************************************************************

def plot_window_in_scene(itemid, reduce_box, figsize=15):
    
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(np.moveaxis(rgb_window_in_scene(itemid, reduce_box),0,-1))
    plt.show()
    return

# ---------------------------------

def plot_preds_vs_original(predictions, itemid, aoi, year, model_name = '-',figsize=(30,40)):
    
    original = np.moveaxis(rgb_window_in_scene(itemid, aoi),0,-1)
    fig, ax = plt.subplots(1,2,figsize=figsize)

    ax[0].imshow(predictions)
    ax[0].set_title("PREDICTIONS "+str(year)+ model_name)

    ax[1].imshow(original)
    ax[1].set_title(str(year)+" original image")

    plt.show()
    return

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

def predict_over_subset(itemid, reduce_box, model):
    image = open_window_in_scene(itemid, reduce_box)
    # reshape image into a np.array where each row is a pixel and the columns are the bands
    pixels = image.reshape([4,-1]).T
    predictions_class = model.predict(pixels)
    # turn back into original raster dimensions
    return predictions_class.reshape([image.shape[1],-1])

# ---------------------------------

# modelmust only take r, g, b, nir as featuers (IN THAT ORDER)
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

def preds_to_image_3labels(nrows, ncols, index, predictions):
    preds = pd.DataFrame(predictions, 
                         columns=['is_iceplant'], 
                         index = index)
    is_iceplant_index = preds[preds.is_iceplant == 1].index
    non_iceplant_index = preds[preds.is_iceplant == 0].index
    
    # initialize empty raster
    reconstruct = np.zeros((nrows,ncols))

    # add is_iceplant as 1's
    i = is_iceplant_index / ncols
    i = i.astype(int)
    j = is_iceplant_index % ncols
    reconstruct[i,j] = 1

    # add non_iceplant as 2's
    i = non_iceplant_index / ncols
    i = i.astype(int)
    j = non_iceplant_index % ncols
    reconstruct[i,j] = 2

    return reconstruct

# # ---------------------------------

# def add_date_features(item, df):

# #    veg = ipf.select_ndvi_df(image)
    
# #    veg['ndvi']=(veg.nir.astype('int16') - veg.r.astype('int16'))/(veg.nir.astype('int16') + veg.r.astype('int16'))

#     df['year'] = item.datetime.year
#     df['month'] = item.datetime.month
#     df['day_in_year'] = day_in_year(item.datetime.day, item.datetime.month, item.datetime.year)

#     veg = veg[['r','g','b','nir','ndvi','year','month','day_in_year']] # order features
#     return df

# # **********************************************************************************************************
# # FROM naip_flights.ipynb

# def query_geom(geom, year):

#     date_range = str(year)+'-01-01/'+str(year)+'-12-31'

#     catalog = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1")

#     search = catalog.search(
#         collections=["naip"], 
#         intersects=geom, 
#         datetime=date_range)
    
#     items =list(search.get_items()) 
#     if len(items)==0:
#         return None
#     return items


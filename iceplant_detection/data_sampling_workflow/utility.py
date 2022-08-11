import pystac_client 
import planetary_computer as pc
import rasterio

import calendar
import numpy as np
import pandas as pd

import os

# **********************************************************************************************************

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
    for i in range(0,month-1):
        n = n+days_in_month[i]
    n = n+day
    if calendar.isleap(year) and month>2:
        n = n+1
    return n

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
    

# **********************************************************************************

def save_raster(raster, fp, shape, bands_n, crs, transform, dtype):
    """
        Saves an array as a 'GTiff' raster with specified parameters.
        Parameters:
                    raster (numpy.ndarray): array of raster values
                    fp (str): file path where raster will be saved
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

# **********************************************************************************

def make_directory(dir_name): 
    """ 
        Checks if the directory with name dir_name (str) exists in the current working directory. 
        If it doesn't, it creates the directory and returns the filepath to it.
    """    
    fp = os.path.join(os.getcwd(),dir_name)  
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp

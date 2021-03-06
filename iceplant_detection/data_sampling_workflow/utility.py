import pystac_client 
import planetary_computer as pc
import rasterio

import calendar
import numpy as np
import pandas as pd

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
    


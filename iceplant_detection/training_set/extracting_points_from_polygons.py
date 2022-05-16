import pandas as pd
import numpy as np
import rasterio
import shapely

from shapely.geometry import shape
from shapely.geometry import Point

import pystac_client
import planetary_computer as pc

import random
random.seed(10)

# *********************************************************************
# *********************************************************************

def get_item_from_id(itemid):
    # accesing Azure storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    # campus point naip scene
    search = catalog.search(
        collections=["naip"],
        ids = itemid)
    
    item = list(search.get_items())[0]
    # sign and open item
    return item

# ---------------------------------------------

def get_raster_from_item(item):
    href = pc.sign(item.assets["image"].href)
    naip = rasterio.open(href)
    return naip

# ---------------------------------------------

# naip = rasterio reader of naip image
# polys and naip have to be in same crs, do polys.to_crs(naip.crs,inplace=True) before
def num_random_points(polys, naip, proportion=0.2):
    pixel_size = naip.res[0]*naip.res[1]
    

    # calculating how many pixels are there in the polygon (approx), by dividing the area of poly by area of a single pixel
    return polys.geometry.apply(lambda p: int((p.area/pixel_size)*proportion))

# ---------------------------------------------

def random_pts_poly(number, polygon):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points  

# ---------------------------------------------

def sample_naip(polys, num_random_pts, naip, item):
    samples = []
    for i in range(0,polys.shape[0]):   # for each polygon in set
    #for i in range(0,3):
        poly = polys['geometry'][i]
        N = num_random_pts[i]

        points = random_pts_poly(N,poly)  # select random points within poly
        sample = pd.DataFrame({
            'geometry': pd.Series(points), 
            'iceplant':pd.Series(np.full(N,polys['iceplant'][i]))  # add iceplant Y/N identification for all pts
                     })

        sample['xy']=sample.geometry.apply(lambda p :(p.x, p.y))   # separate coords for reasterio.io.DatasetReader sample function
        data_generator = naip.sample(sample.xy)   # extract band values from naip image 
        data = []
        for i in data_generator:
            data.append(i)
        data = np.vstack(data)
        data = pd.DataFrame(data, columns=['r','g','b','nir']) # create pd DataFrame from np.array

        sample = pd.concat([sample,data],axis=1)  # add band data to points
        samples.append(sample)   # add new samples

    df = pd.concat(samples) # create dataframe from samples list
    
    df['x']= df.geometry.apply(lambda p : p.x)   # some df cleaning
    df['y']= df.geometry.apply(lambda p : p.y)
    df.drop('xy',axis=1,inplace=True)
    df = df[['geometry','x','y','iceplant','r','g','b','nir']]
    
    df['year'] = item.datetime.year   # add date to samples
    df['month'] = item.datetime.month
    df['day'] = item.datetime.day 
    return df

# ---------------------------------------------

def sample_naip_from_polys(polys_raw, itemid, proportion=0.2):
    item = get_item_from_id(itemid)
    naip = get_raster_from_item(item)
    
    polys = polys_raw.to_crs(naip.crs)
    num_random_pts = num_random_points(polys,naip)
    return sample_naip(polys, num_random_pts, naip, item)



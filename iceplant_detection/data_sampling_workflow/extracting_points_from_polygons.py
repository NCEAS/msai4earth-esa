import os

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


import calendar
import warnings



# *********************************************************************

def path_to_polygons(aoi,year):
    # root for all polygons collected on naip scenes
    root = '/home/jovyan/msai4earth-esa/iceplant_detection/create_train_test_sets/polygons_from_naip_images'
    fp = os.path.join(root, 
                      aoi+'_polygons', 
                      aoi+'_polygons_'+str(year), 
                      aoi+'_polygons_'+str(year)+'.shp')
    return fp


# *********************************************************************

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
    naip = rasterio.open(href)
    return naip



# *********************************************************************

# extracts at most number of random points within polygon
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
# polys needs to have at least geometry, iceplant and id columns 

def sample_naip(polys, num_random_pts, naip, item):
    samples = []
    for i in range(0,polys.shape[0]):   # for each polygon in set
        poly = polys['geometry'][i]
        N = num_random_pts[i]

        points = random_pts_poly(N,poly)  # select random points within poly
        sample = pd.DataFrame({
            'geometry': pd.Series(points), 
            'iceplant':pd.Series(np.full(N,polys['iceplant'][i])),  # add iceplant Y/N identification for all pts
            'polygon_id':pd.Series(np.full(N,polys['id'][i]))
                     })

        sample['xy'] = sample.geometry.apply(lambda p :(p.x, p.y))  # separate coords (needed for reasterio.io.DatasetReader sample function)
        data_generator = naip.sample(sample.xy)   # extract band values from naip image 
        data = []
        for i in data_generator: 
            data.append(i)
        data = np.vstack(data)
        data = pd.DataFrame(data, columns=['r','g','b','nir']) # create pd DataFrame from np.array
        
        sample = pd.concat([sample,data],axis=1)  # add band data to points
        samples.append(sample)   # add new samples

    df = pd.concat(samples) # create dataframe from samples list
    
    df['x']= df.geometry.apply(lambda p : p.x)   # coordinate cleaning
    df['y']= df.geometry.apply(lambda p : p.y)
    df.drop('xy',axis=1,inplace=True)
    
    df['year'] = item.datetime.year   # add date to samples
    df['month'] = item.datetime.month
    df['day_in_year'] = day_in_year(item.datetime.day, item.datetime.month, item.datetime.year )
    df['naip_id'] = item.id           # add naip item id to samples

    df[['geometry','x','y',
        'iceplant',
        'r','g','b','nir',
        'year','month','day_in_year',
        'naip_id','polygon_id']]
    return df

# *********************************************************************

# naip = rasterio reader of naip image
# polys and naip have to be in same crs, do polys.to_crs(naip.crs,inplace=True) before
def num_pts_area_proportion(polys, naip, proportion):
    
    pixel_size = naip.res[0]*naip.res[1]
    
    # calculating how many pixels are there in the polygon (approx)
    # by dividing the area of poly by area of a single pixel
    return polys.geometry.apply(lambda p: int((p.area/pixel_size)*proportion))

# ---------------------------------------------

def sample_naip_from_polys_proportion(polys_raw, itemid, proportion):
    item = get_item_from_id(itemid)
    naip = get_raster_from_item(item)
    
    polys = polys_raw.to_crs(naip.crs)
    
    num_random_pts = num_pts_area_proportion(polys, naip, proportion)
    return sample_naip(polys, num_random_pts, naip, item)

# ---------------------------------------------

def naip_sample_proportion_no_warnings(polys, itemid, proportion):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys_proportion(polys, itemid, proportion)
    return df



# *********************************************************************

def sample_naip_from_polys_fixedn(polys_raw, itemid, n):
    item = get_item_from_id(itemid)
    naip = get_raster_from_item(item)
    
    polys = polys_raw.to_crs(naip.crs)
    
    num_random_pts = np.full(polys_raw.shape[0],n)
    return sample_naip(polys, num_random_pts, naip, item)

# ---------------------------------------------

def naip_sample_n_no_warnings(polys, itemid, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys_fixedn(polys, itemid, n)
    return df

# *********************************************************************
# alpha = proportion of pixels in the smallest polygon to be sampled
# diff = difference between #pixels sampled from smallest polygon and #pixels sampled from biggest polygon

def num_pts_sliding_proportion(polys, alpha, diff):
    
    n1 = polys.pixels[polys.shape[0]-1]
    nN = polys.pixels[0]
    beta = (diff + alpha*n1)/nN
    step = (alpha - beta)/polys.shape[0]
    
    proportions = np.arange(beta,alpha,step)       
    #print(proportions)
    num_random_pts = proportions * polys.pixels.to_numpy()
    #print(num_random_pts)
    num_random_pts = num_random_pts.astype('int16')
    #print(num_random_pts)
    
    return num_random_pts

# ---------------------------------------------

def num_pts_sliding(polys, alpha, m):
    
    num_random_pts = alpha * polys.pixels.to_numpy()
    num_random_pts = num_random_pts.astype('int32')
    num_random_pts[num_random_pts>m] = m
    
    return num_random_pts

# ---------------------------------------------

def sample_naip_from_polys_sliding(polys_raw, itemid, alpha, m):
    item = get_item_from_id(itemid)
    naip = get_raster_from_item(item)
    
    polys = polys_raw.to_crs(naip.crs)
    
    pixel_size = naip.res[0]*naip.res[1]
    polys['pixels'] = polys.geometry.apply(lambda p: int((p.area/pixel_size)))
    polys = polys.sort_values(by=['pixels'], ascending=False).reset_index(drop=True)
        
    num_random_pts = num_pts_sliding(polys, alpha, m)
    return sample_naip(polys, num_random_pts, naip, item)

# ---------------------------------------------

def naip_sample_sliding_no_warnings(polys, itemid, alpha, m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys_sliding(polys, itemid, alpha, m)
    return df


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
    

# *********************************************************************

def day_in_year(day,month,year):
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    n = 0
    for i in range(0,month-1):
        n = n+days_in_month[i]
    n = n+day
    if calendar.isleap(year) and month>2:
        n = n+1
    return n
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import rasterio
import rioxarray as rioxr
from rasterio.crs import CRS

# https://stackoverflow.com/questions/43966350/getting-the-maximum-in-each-rolling-window-of-a-2d-numpy-array
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

import utility


# *********************************************************************
def geodataframe_from_csv(fp, lon_label, lat_label, crs):
    """
        Transforms a csv with longitude and latitude columns into a GeoDataFrame.
            Parameters:
                        fp (str): 
                            File path to csv containing coordinates of points.
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
    df = pd.read_csv(fp)
    if 'geometry' in df.columns:           # rename geometry column if it exists
        df = df = df.rename(columns={'geometry': 'geometry_0'})
    
    # recreate geometry column as shapely Points
    xy = []
    for x,y in zip(df[lon_label],df[lat_label]):
        xy.append(Point(x,y))
    df['geometry'] = xy

    return gpd.GeoDataFrame(df, crs=crs)

# *********************************************************************
"""
    Creates a dataframe of raster bands values at the given points.
    Points and raster MUST HAVE SAME CRS for results to be correct. 
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
# pts have to be in same crs as rast_reader
# pts are shapely.Points
def sample_raster_from_pts(pts, rast_reader, rast_band_names):
    if rast_reader.count != len(rast_band_names):
        print('# band names != # bands in raster'
        return

    # sample
    sample_coords = pts.apply(lambda p :(p.x, p.y))  
    samples_generator = rast_reader.sample(sample_coords)    
    
    # make band values into dataframe
    samples = np.vstack(list(samples_generator))   
    samples = pd.DataFrame(samples, columns=rast_band_names)
    
    return samples
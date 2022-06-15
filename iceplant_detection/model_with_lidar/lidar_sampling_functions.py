import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

# **********************************************************************************


def save_raster(raster, fp, shape, bands_n, crs, transform, dtype):
    with rasterio.open(
        fp,  # file path
        'w',           # w = write
        driver='GTiff', # format
        height = shape[0], 
        width = shape[1],
        count = bands_n,  # number of raster bands in the dataset
        dtype = dtype,
        crs = crs,
        transform = transform,
    ) as dst:
        dst.write(raster.astype(dtype), bands_n)
    return 
    
# ------------------------------------------------------------------------------

# folder_path = path to folder to save rasters
def save_min_max_diff_rasters(rast_reader, folder_path):    
    rast = rast_reader.read([1]).squeeze() # read raster

    maxs = maxf2D(rast, size=(3,3)) # calculate min max and difference
    mins = minf2D(rast, size=(3,3))   
    diffs = maxs - mins
    
    # save rasters
    m = [maxs, mins, diffs]
    m_labels = ['maxs', 'mins', 'diffs']
    for i in range(0,3):
        fp = os.path.join(folder_path, 'lidar_'+m_labels[i]+'.tif')
        save_raster(m[i], 
                    fp, 
                    rast.shape,
                    1,
                    rast_reader.crs, 
                    rast_reader.transform, 
                    rasterio.uint8)
    return

# ------------------------------------------------------------------------------

def save_avg_rasters(rast_reader, folder_path):
    rast = rast_reader.read([1]).squeeze() # read raster
    
    # calculate averages
    w = np.ones(9).reshape(3,3)
    avgs = conf2D(rast, 
             weights=w,
             mode='constant')
    avgs = avgs/9
    
    # save averages
    fp = os.path.join(folder_path, 'lidar_avgs.tif')
    save_raster(avgs, 
                fp, 
                rast.shape, 
                1,
                rast_reader.crs, 
                rast_reader.transform, 
                rasterio.float32)
    return


# **********************************************************************************


# fp = filepath of csv, must have x and y columns representing coordinates of point
# crs must be the crs of the csv coordinates
def geodataframe_from_csv(fp, crs):
    df_raw = pd.read_csv(fp)
    df = df_raw.drop(['geometry','Unnamed: 0'], axis=1)
    
    # recreate geometry column with shapely Points
    xy = []
    for x,y in zip(df.x,df.y):
        xy.append(Point(x,y))
    df['geometry'] = xy
    df = df.drop(['x','y'], axis=1)

    pts = gpd.GeoDataFrame(df, crs=crs)
    return pts

# ------------------------------------------------------------------------------
# lidar = lidar raster, opened via rasterio.open(fp)
def sample_from_lidar(pts_raw, lidar):
    pts = pts_raw.to_crs(lidar.crs)
    pts['xy'] = pts.geometry.apply(lambda p :(p.x, p.y)) #this is the format needed to sample 
    
    sample = lidar.sample(pts.xy.to_list())
    # *** SIMPLIFY ***
    samples = []
    for x in sample:
        samples.append(x[0])

    return samples 
# ------------------------------------------------------------------------------

def min_max_samples_from_lidar(pts_raw, lidar):
    pts = pts_raw.to_crs(lidar.crs)

    rast = lidar.to_np #****
    
    win_size = (3,3)
    min_lidar = minf2D(rast, size=win_size)
    max_lidar = maxf2D(rast, size=win_size)
    diff = max_lidar - max_lidar

    
    w = np.ones(9).reshape(3,3)
    w
    avg = conf2D(a, 
             weights=w,
             mode='constant')
    avg/9
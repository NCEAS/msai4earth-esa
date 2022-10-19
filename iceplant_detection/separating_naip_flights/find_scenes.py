import pystac_client

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box


# geometry = geometry that NAIP scenes must intersect
# year = year of the NAIP scenes you want
# returns a list of the items of NAIP scenes from year that intersect geometry
def query_geom(geom, year):

    date_range = str(year)+'-01-01/'+str(year)+'-12-31'

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=["naip"], 
        intersects=geom, 
        datetime=date_range)
    
    items =list(search.get_items()) 
    if len(items)==0:
        return None
    return items

# --------------------------------------------------------------------------------------------------------
def make_bbox(item):
    c = item.properties['proj:bbox']
    return box(c[0],c[1],c[2],c[3])

# --------------------------------------------------------------------------------------------------------
def group_flight(df,date):
    same_date = df[df['date']==date]
    
    crs_list = same_date.crs.unique()
    polygons = []
    for crs in crs_list:
        same_crs = same_date[same_date['crs']==crs]
        area = shapely.ops.unary_union(same_crs.bbox)
        gdf = gpd.GeoDataFrame({'geometry':[area]}, 
                               crs='EPSG:'+str(crs))
        gdf.to_crs('EPSG:4326',inplace=True)
        polygons.append(gdf.geometry[0])

        flight = shapely.ops.unary_union(polygons)

    return flight

# --------------------------------------------------------------------------------------------------------

def flight_paths(df):
    dates = df.date.unique()
    flights = []
    for date in dates:
        flights.append(group_flight(df,date))
    gdf = gpd.GeoDataFrame({'date':dates, 'geometry':flights},
                     crs = 'EPSG:4326')
    return gdf
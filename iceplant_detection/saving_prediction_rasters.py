## Save prediction rasters over aois

import rioxarray as rioxr
import planetary_computer as pc

def small_raster(itemid, reduce_box):
    item = ipf.get_item_from_id(itemid)
    href = pc.sign(item.assets["image"].href)
    
    rast = rioxr.open_rasterio(href)
    
    reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs="EPSG:4326")
    reduce = reduce.to_crs(rast.rio.crs)    
    
    rast_small = rast.rio.clip_box(*reduce.total_bounds)
    return rast_small

# *******************************************************************************************

for i in range(0,4):
    itemid = aois.iloc[i].itemid_18
    geom = aois.iloc[i].geometry
    
    preds = predict_over_aoi(itemid ,geom, rfc)
    
    small = small_raster(itemid, geom)
   
    h = preds.shape[0]
    w = preds.shape[1]
    
    # ---- save predictions ----
    fp = os.path.join(os.getcwd(),'trial_rasters','scene_'+str(i)+'_2018_iceplant.tif')    
    with rasterio.open(
        fp,  # file path
        'w',           # w = write
        driver='GTiff', # format
        height = h, 
        width = w,
        count = 1,  # number of raster bands in the dataset
        dtype = rasterio.uint8,
        crs = small.rio.crs,
        transform = small.rio.transform(),
    ) as dst:
        dst.write(preds.astype(rasterio.uint8), 1)
        
# *******************************************************************************************
        
# for i in range(0,4):
#     geom = aois.iloc[i].geometry
#     itemid = aois.iloc[i].itemid_18
#     item = ipf.get_item_from_id(itemid)
#     rast = ipf.get_raster_from_item(item)

#     aoi = ipf.open_window_in_scene(itemid ,geom)
#     print(    aoi.shape)
    
#     h = aoi.shape[1]
#     w = aoi.shape[2]
    
#     fp = os.path.join(os.getcwd(),'trial_rasters','scene_'+str(i)+'_2018_naip.tif')    
#     with rasterio.open(
#         fp,  # file path
#         'w',           # w = write
#         driver='GTiff', # format
#         height = h, 
#         width = w,
#         count = 4,  # number of raster bands in the dataset
#         dtype = aoi.dtype,
#         crs = rast.crs,
#         transform = rast.transform,
#     ) as dst:
#         dst.write(aoi, indexes =[1,2,3,4])


# fp = os.path.join(os.getcwd(),'trial_rasters','scene_'+str(0)+'_2018_naip.tif')    
# rgb = rioxr.open_rasterio(fp)
# rgb.plot.imshow(figsize=(8,8))

# get the washed out image
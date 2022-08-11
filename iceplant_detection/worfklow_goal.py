itemid = 'ca_m_3411934_sw_11_060_20200521'

raster = rm.rioxr_from_itemid(itemid)

is_veg, water_index, not_veg_index = rm.add_spectral_features(df = rm.raster_as_df(raster.to_numpy(), ['r','g','b','nir']), 
                                                           ndwi_thresh = 0.3, 
                                                           ndvi_thresh = 0.05) 
is_veg.drop('ndwi', axis=1, inplace=True)


is_veg = rm.add_date_features(is_veg, raster.datetime)

print('time to make features df: ', (time.time()-t0))
# time to make features df:  10.682304620742798

# -------------------------------------------------------

itemid = 'ca_m_3411934_sw_11_060_20200521'

raster = rioxr_from_itemid(itemid)

# keep raster or save these arguments as dict?
# Q: maybe not because rioxarray is not holding the raster in memory?   ## A: decided to keep raster since it is used in lidar-model fro cropping canopy height rasters
date = raster.date
shape = raster.shape
raster.rio.crs,
raster.rio.transform()

pixels = raster_as_df(raster.to_numpy(),  ['r','g','b','nir'])
# delete raster here

# Q: maybe this uses too much memory... a bit slow? maybe use numpy?   # A: comparable performance with previous method
not_water, water_index = feature_df_treshold(pixels, 'ndwi', 0.3, False, normalized_difference_index, 1,3)   
is_veg, not_veg_index = feature_df_treshold(not_water, 'ndvi', 0.05, True, normalized_difference_index, 3,0)
# Q: drop ndwi?  A: yes, not included in previous model, only for pre-processing

is_veg = add_date_features(is_veg, date)

# convert all auxiliary lidar rasters as df and sample with veg index
# add lidar to spectral + dates   ## df_lidar_veg = df_lidar.iloc[veg.index]

reconstruct = indices_to_image(12500, 10580, [water_index, non_veg_index], [3,2], back_value=1)

# -------------------------------------------------------

itemid = 'ca_m_3411934_sw_11_060_20200521'
item = ut.get_item_from_id(itemid)
pixels = ut.get_raster_from_item(item).read([1,2,3,4])
pixels_df = ipf.image_as_df( pixels, # np.array,
                          column_names)

not_water, water_index = ipf.select_ndwi_df(pixels_df)
is_veg, not_veg_index = ipf.select_ndvi_df(not_water)  # these should proable be same function with parameter
is_veg = ipf.add_date_features(is_veg, item)

# convert all auxiliary lidar rasters as df

df_lidar_veg = df_lidar.iloc[veg.index]




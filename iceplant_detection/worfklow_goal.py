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



# -------------------------------------------------------

itemid = 'ca_m_3411934_sw_11_060_20200521'

raster = rioxr_from_itemid(itemid)

# keep raster or save these arguments as dict?
# maybe not because rioxarray is not holding the raster in memory?
date = raster.date
shape = raster.shape
raster.rio.crs,
raster.rio.transform()

pixels = raster_as_df(raster.to_numpy(),  ['r','g','b','nir'])
# delete raster here

# maybe this uses too much memory... a bit slow? maybe use numpy?
not_water, water_index = feature_df_treshold(pixels, 'ndwi', 0.3, False, normalized_difference_index, 1,3)   
is_veg, not_veg_index = feature_df_treshold(not_water, 'ndvi', 0.05, True, normalized_difference_index, 3,0)
# TO DO: drop ndwi?

is_veg = add_date_features(is_veg, date)

# convert all auxiliary lidar rasters as df and sample with veg index
# add lidar to spectral + dates   ## df_lidar_veg = df_lidar.iloc[veg.index]

reconstruct = indices_to_image(12500, 10580, [water_index, non_veg_index], [3,2], back_value=1)



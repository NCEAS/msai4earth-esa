def count_pixels_in_polygons(polys, itemid):
    item = utility.get_item_from_id(itemid)
    rast_reader = utility.get_raster_from_item(item)
    
    # convert to same crs as raster to properly calculate area of polygons
    polys_match = polys.to_crs(rast_reader.crs)
    
    # area of a single pixel from raster resolution    
    pixel_size = rast_reader.res[0]*rast_reader.res[1]
    
    return  polys_match.geometry.apply(lambda p: int((p.area/pixel_size)))

# ---------------------------------------------

# n_pixels = np.array
def num_random_pts(n_pixels, param, sample_fraction=0, max_sample=0, const_sample=0):
    if param not in ['proportion', 'sliding', 'constant']:
        return
                     
    if param == 'proportion':
        num_random_pts = sample_fraction * n_pixels
    
    else if param == 'sliding':
        num_random_pts = sample_fraction * n_pixels
        num_random_pts[num_random_pts>max_sample] = max_sample
    
    else if param == 'constant':
        num_random_pts = np.full(n_pixels.shape[0],const_sample)
    
    num_random_pts = num_random_pts.astype('int32')
    return num_random_pts

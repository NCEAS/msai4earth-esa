# DATA SAMPLING FOR ICE PLANT DETECTION MODEL


## Intro


This folder contains a series of five notebooks that create a dataset of georeferenced points of known ice plant and non-ice plant locations across time on the Santa Barbara County coast. 

Each point in the final dataset has information about its location, collection date, and spectral and canopy height features at that point. The spectral features are extracted from [National Agriculture Imagery Program (NAIP)](https://naip-usdaonline.hub.arcgis.com) images, and the canopy height data is obtained from the [California Forest Observatory (CFO)](https://forestobservatory.com) canopy height datasets. The goal is to use the resulting dataset to train machine learning models to identify ice plant locations on NAIP images.  

All code was developed and tested in [Microsoft's Planetary Computer](https://planetarycomputer.microsoft.com) coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools.



## Data Access


[NAIP images](https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook) are part of Microsoft's Planetary Computer's data catalog and can be directly accessed through their API.

After creating a CFO account, the California Forest Observatory canopy height dataset can be downloaded through [CFO's API](https://github.com/forestobservatory/cfo-api).

The starting point for the data sampling is a set of polygons that outline confirmed ice plant and non-iceplant locations. These are available in the polygons_from_naip_images_folder. To create these polygons, JB and CGG outlined the confirmed iceplant and non-iceplant locations on all available NAIP images over four regions regularly spaced along the Santa Barbara County coast: Carpinteria State Beach, the University of California Santa Barbara campus, Gaviota State Park, and Point Conception. The ice plant locations were based on field observations and digitized records of ice plant locations from GBIF and Calflora. To create the polygons, the NAIP images were loaded from the Planetary Computer's data repository on QGIS using the [Open STAC API Browser plugin](https://planetarycomputer.microsoft.com/docs/overview/qgis-plugin/).



## Notebook Workflow

The notebooks are numbered in the order they should be run. 

* `1_sample_pts_from_polygons`

Creates datasets of random points extracted from polygons in the 'polygons_form_naip_images' folder, together with the corresponding spectral and date features from NAIP images.

* `2_download_CFO_canopy_height_raster`

Creates a canopy height raster layer for Santa Barbara County from the CFO canopy height data for the state of California. These rasters are not deleted in any of the next notebooks.

* `3_add_canopy_height_features`

Adds canopy height features to the points sampled in the first notebook.

* `4_assemble_data_samples`

Assembles all the csv files produced by the previous notebooks into a single dataframe. It either saves the data frame as a single csv or splits it into training and test sets. It also includes statistics of the combined dataset.



## Features in Final Dataset


Each point in the final dataset has the following associated features:

    - x, y: coordinates of point p
    - pts_crs: CRS of coordinates x, y
    - naip_id: itemid of the NAIP from which p was sampled from
    - year, month, day_in_year: year, month and day of the year when the NAIP image was collected
    - polygon_id: id of the polygon from which p was sampled from
    - iceplant: whether point p corresponds to a confirmed iceplant location or a confirmed non-iceplant location (0 = non-iceplant, 1 = iceplant)
    - r, g, b, nir: Red, Green, Blue and NIR bands values of NAIP scene with naip_id at cooridnates of point p
    - ndvi: computed for each point using the Red and NIR bands
    - aoi: name of the area of interest where the points were sampled from
    - lidar: canopy height at point p on year of point collection 
    - max_lidar, min_lidar: max/min canopy height in a 3x3 window centered at point p
    - max_min_lidar: difference between max_lidar and min_lidar features
    - avg_lidar: avg canopy height in a 3x3 window centered at point p
    

## Custom Libraries

### sample_rasters.py
Functions to:
   - access NAIP data in the Planetary Computer repository using pystac_client and planetary_computer libraries.
   - sample random points inside a polygon    
   - sample raster values at a list of points
   - extract spectral and calendar values from NAIP iamges at specific coordinates
   - create and save auxiliary rasters to sample avg_lidar, max_lidar and min_lidar features using methods from scipy.ndimage 
   - convert points in csv to geodataframe




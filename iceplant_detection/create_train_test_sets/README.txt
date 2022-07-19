DATA SAMPLING FOR ICE PLANT DETECTION MODEL
-------------------------------------------

1. Intro
--------

This folder contains a series of five notebooks that create a dataset of georeferenced points of known iceplant and non-iceplant locations across time on the Santa Barbara County coast. 

Each point in the final dataset has information about its location, collection date, and spectral and canopy height features at that point (and time of collection). The spectral features are extracted from National Agriculture Imagery Program (NAIP) images, and the canopy height data is obtained from the California Forest Observatory (CFO) canopy height datasets. The goal is to use the resulting dataset to train machine learning models to identify iceplant locations on NAIP images.  

All code was developed and tested in Microsoft's Planetary Computer coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools.


2. Data Access
--------------

NAIP images are part of Microsoft's Planetary Computer's data catalog and can be directly accessed through their API. See https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook

The California Forest Observatory canopy height dataset can be downloaded through CFO's API [*] after you create a CFO account. See https://github.com/forestobservatory/cfo-api

The starting point for the data sampling is a set of polygons that outline confirmed iceplant and non-iceplant locations. These are available in the polygons_from_naip_images_folder. To create these polygons, JB and CGG outlined the confirmed iceplant and non-iceplant locations on all available NAIP images over four regions regularly spaced along the Santa Barbara County coast: Carpinteria State Beach, the University of California Santa Barbara campus, Gaviota State Park, and Point Conception. The ice plant locations were based on field observations and digitized records of ice plant locations from GBIF and Calflora. To create the polygons, the NAIP images were loaded from the Planetary Computer's data repository on QGIS using the Open STAC API Browser plugin (see https://planetarycomputer.microsoft.com/docs/overview/qgis-plugin/).


3. Notebook Workflow
--------------------
The notebooks are numbered in the order they should be run. 

* 1_sample_pts_from_polygons
Creates an initial dataset of random points extracted from polygons in the 'polygons_form_naip_images' folder together with the corresponding spectral and date features from NAIP images.

* 2_download_CFO_canopy_height_raster
Creates a canopy height raster layer for Santa Barbara County from the CFO canopy height data for the state of California. These rasters are not deleted in any of the next notebooks.

* 3_add_canopy_height_features
Adds canopy height features to the points sampled in the first notebook.

* 4_make_single_csv
Assembles all the csvs produced by the previous notebook into a single dataframe and saves it as a csv. It also includes statistics of the combined dataset. At this point the dataset is complete.

* 5_create_train_test_set
Divides the dataset created in the previous notebook into train and tests sets by sampling the same specified percentage of points per scene to go into the training set. This is an effort to keep the training and test sets unbiased towards scenes that have more points sampled from them. 


4. Features in Final Dataset
----------------------------

Each point in the final dataset has the following associated features:

    1. geometry: coordinates of point p (in the CRS of the NAIP with itemid naip_id)
    2. naip_id: itemid of the NAIP from which p was sampled from
    3,4,5,6. year, month, day_in_year: year, month and day of the year when the NAIP image was collected
    7. polygon_id: id of the polygon from which p was sampled from
    8. iceplant: whether point p corresponds to a confirmed iceplant location or a confirmed non-iceplant location (0 = non-iceplant, 1 = iceplant)
    9,10,11,12. r, g, b, nir: Red, Green, Blue and NIR bands values of NAIP scene with naip_id at cooridnates of point p
    13. ndvi: computed for each point using the Red and NIR bands
    14. aoi: name of the area of interest where the points were sampled from
    15. lidar: canopy height at point p on year of point collection 
    16,17. max_lidar, min_lidar: max/min canopy height in a 3x3 window centered at point p
    18. max_min_lidar: difference between max_lidar and min_lidar features
    19. avg_lidar: avg canopy height in a 3x3 window centered at point p
    
Features 1-14 are generated in the first notebook 1_sample_pts_from_polygons. Features 15-19 are added in the third notebook 3_add_canopy_height_features. 
    

5. Custom Libraries
-------------------






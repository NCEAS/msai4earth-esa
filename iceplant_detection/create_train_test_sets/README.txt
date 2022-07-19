DATA SAMPLING FOR ICE PLANT DETECTION MODEL
-------------------------------------------

1. Intro
--------

This folder contains a series of five notebooks that create a dataset of georeferenced points of known iceplant and non-iceplant locations across time on the Santa Barbara County coast. 

Each point in the final dataset has information about its location, collection date, and spectral and canopy height features at that point (and time of collection). The spectral features are extracted from National Agriculture Imagery Program (NAIP) images, and the canopy height data is obtained from the California Forest Observatory (CFO) canopy height datasets. The goal is to use the resulting dataset to train machine learning models to identify iceplant locations on NAIP images.  

All code was developed and tested in Microsoft's Planetary Computer coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools.


2. Data Access
--------------

NAIP images are part of Microsoft's Planetary Computer's data catalog and can be directly accessed through their API. (See https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook)

The California Forest Observatory canopy height dataset can be downloaded through CFO's API [*] after you create a CFO account. (See https://github.com/forestobservatory/cfo-api)

The starting point for the data sampling is a set of polygons that outline confirmed iceplant and non-iceplant locations. These are available in the polygons_from_naip_images_folder. To create these polygons, JB and CGG outlined the confirmed iceplant and non-iceplant locations on all available NAIP images over four regions regularly spaced along the Santa Barbara County coast: Carpinteria State Beach, the University of California Santa Barbara campus, Gaviota State Park, and Point Conception. The ice plant locations were based on field observations and digitized records of ice plant locations from GBIF and Calflora. To create the polygons, the NAIP images were loaded from the Planetary Computer's data repository on QGIS using the Open STAC API Browser plugin.


3. Notebook Workflow
--------------------



4. Features in Final Dataset
----------------------------
 
Each point in the final dataset has the following associated features:

    - geometry: coordinates of point p (in the CRS of the NAIP with itemid naip_id)
    - naip_id: itemid of the NAIP from which p was sampled from
    - year, month, day_in_year: year, month and day of the year when the NAIP image was collected
    - polygon_id: id of the polygon from which p was sampled from
    - iceplant: whether point p corresponds to a confirmed iceplant location or a confirmed non-iceplant location (0 = non-iceplant, 1 = iceplant)
    - r, g, b, nir: Red, Green, Blue and NIR bands values of NAIP scene with naip_id at at cooridnates of point p
    - ndvi: computed for each point using the Red and NIR bands
    - aoi: name of the area of interest where the points were sampled from
    


5. Creating Train and Test Sets
-------------------------------






what, why, and the how of the project.


What your application does,
Why you used the technologies you used,
Some of the challenges you faced and features you hope to implement in the future.
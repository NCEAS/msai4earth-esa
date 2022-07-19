DATA SAMPLING FOR ICE PLANT DETECTION MODEL
-------------------------------------------

1. Intro

This folder contains a series of five notebooks that create a dataset of georeferenced points of known iceplant and non-iceplant locations across time on the Santa Barbara County coast. The goal is to use the resulting dataset to train machine learning models. 

Each point in the final dataset has information about its location, collection date, and spectral and canopy height features at that point (and time of collection). The spectral features are extracted from National Agriculture Imagery Program (NAIP) [*] images, and the canopy height data is obtained from the California Forest Observatory (CFO) [*]. 




2. Polygons for Iceplant Locations


3. Notebook Workflow


4. Features in Final Dataset
 
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

The authors first outlined the confirmed locations as polygons on all available NAIP images over four regions regularly spaced along the Santa Barbara County coast: Carpinteria State Beach, the University of California, Santa Barbara campus, Gaviota State Park, and Point Conception. The polygons outlining confirmed ice plant outgrows based on field observations and digitized records of ice plant locations from GBIF [*] and Calflora [*].




what, why, and the how of the project.


What your application does,
Why you used the technologies you used,
Some of the challenges you faced and features you hope to implement in the future.
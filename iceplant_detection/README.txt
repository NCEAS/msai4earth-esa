ICE PLANT DETECTION MODEL
-------------------------

1. About
--------

This is a working repository for an NCEAS project that looks for areas with ice plant (Carpobrotus edulis) along the Santa Barbara County coast. The project analyses aerial images collected by the National Agriculture Imagery Program (NAIP) from 2012 to 2020. 


All code is being developed and tested in Microsoft's Planetary Computer coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools (see https://planetarycomputer.microsoft.com).


4. Data
-------

NAIP images are part of Microsoft's Planetary Computer's data catalog and can be directly accessed through their API. See https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook

The California Forest Observatory canopy height dataset can be downloaded through CFO's API [*] after you create a CFO account. See https://github.com/forestobservatory/cfo-api



2. TRIALS 
---------

All folders starting with TRIALS_ include notebooks with trial code that is no longer in use or has been integrated to the custom libraries (.py files).

Notebooks starting with TRIALS_# on this folder are workflows on which we are currently working on.


3. Finished code
----------------



    

5. Custom Libraries
-------------------
There are two custom libraries implemented for the data sampling:

* lidar_sampling_functions.py:
Custom functions to:
    - create and save auxiliary rasters to sample avg_lidar, max_lidar and min_lidar features using methods from scipy.ndimage 
    - convert points in csv to geodataframe
    - sample raster values at a list of points
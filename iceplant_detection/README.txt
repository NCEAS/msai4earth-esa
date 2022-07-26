DATA SAMPLING FOR ICE PLANT DETECTION MODEL
-------------------------------------------

1. About
--------



All code was developed and tested in Microsoft's Planetary Computer coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools.


2. TRIALS 
---------

3. Finished
-----------

4. Data
-------
    

5. Custom Libraries
-------------------
There are two custom libraries implemented for the data sampling:

* lidar_sampling_functions.py:
Custom functions to:
    - create and save auxiliary rasters to sample avg_lidar, max_lidar and min_lidar features using methods from scipy.ndimage 
    - convert points in csv to geodataframe
    - sample raster values at a list of points
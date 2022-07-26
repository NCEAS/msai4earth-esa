ICE PLANT DETECTION MODEL
-------------------------

1. About
--------

This is a working repository for an NCEAS project that uses machine learning techniques to look for areas with ice plant (Carpobrotus edulis) along the Santa Barbara County coast. In this project we analyse aerial images collected by the National Agriculture Imagery Program (NAIP) from 2012 to 2020. 


All code is being developed and tested in Microsoft's Planetary Computer coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools (see https://planetarycomputer.microsoft.com).


2. TRIALS 
---------
All folders starting with TRIALS_ include notebooks with trial code that is no longer in use or has been integrated to the custom libraries (.py files).

Notebooks in this folder starting with TRIALS_# are workflows on which we are currently working on:

* TRIALS_8: classify pixels in a NAIP scene into iceplant, non-iceplant and low-ndvi using a random forest classification model trained on the spectral features of the training set generated in data_sampling_workflow

* TRIALS_10: classify pixels in a NAIP scene into iceplant and non-iceplant using a random forest classification model trained on the spectral and canopy height features of the training set generated in data_sampling_workflow

* TRIALS_11: same as TRIALS_10, but classifies into three categories 

* TRIALS_12: assign to each pixel within an area of interest (subset of NAIP scene) the probability with which the random forest classifies it as iceplant (see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba) 



3. Workflows
------------

* data_sampling_workflow

* separating_naip_flights

    

5. Custom Libraries
-------------------
There are two custom libraries implemented for the data sampling:

* lidar_sampling_functions.py:
Custom functions to:
    - create and save auxiliary rasters to sample avg_lidar, max_lidar and min_lidar features using methods from scipy.ndimage 
    - convert points in csv to geodataframe
    - sample raster values at a list of points
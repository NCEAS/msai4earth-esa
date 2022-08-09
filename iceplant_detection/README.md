# ICE PLANT DETECTION MODEL


## About

This is a working repository for an NCEAS project that uses machine learning techniques to look for areas with ice plant (Carpobrotus edulis) along the Santa Barbara County coast. In this project we analyse aerial images collected by the [National Agriculture Imagery Program (NAIP)](https://naip-usdaonline.hub.arcgis.com) from 2012 to 2020. 

All code is being developed and tested in [Microsoft's Planetary Computer](https://planetarycomputer.microsoft.com) coding environment. The Planetary Computer is a cloud-based analysis platform with an expansive catalog of environmental data and a development environment based on open-source tools.

[//]: # (--------------------)

## Active Code 

Notebooks in the current folder starting with TRIALS_# are workflows on which we are working on at the moment:

* TRIALS_8: classify pixels in a NAIP scene into iceplant, non-iceplant and low-ndvi using a random forest classification model trained on **only the spectral and date** features of the training set generated in data_sampling_workflow

* TRIALS_11: classify pixels in a NAIP scene into iceplant, non-iceplant and low-ndvi using a random forest classification model trained on the **spectral, date and canopy height** features of the training set generated in data_sampling_workflow

* TRIALS_12: assign to each pixel within an area of interest (subset of NAIP scene) the [probability](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba)  with which the random forest classifies it as iceplant (see 



[//]: # (--------------------)

## Stable Code

These are folders containing stable code:

#### `data_sampling_workflow`

This folder contains a series of notebooks that creates a dataset of georeferenced points of known iceplant and non-iceplant locations across time on the Santa Barbara County coast. Each point in the final dataset has information about its location, collection date, and spectral and canopy height features at that point. The resulting dataset (or a subset) is used to train machine learning models to identify iceplant locations on NAIP images.  


### `separating_naip_flights`:

This folder contains a notebook that, given a shapefile (not too complex) returns:

   1. a list of the NAIP scenes that cover the shapefile
   2. a shapefile with the NAIP scenes bounding boxes aggregated by date of collection
    
The coastal_buffer folder contains a shapefile of a rough outline of the Santa Barbara County coast that can be used to run the notebook. 

[//]: # (--------------------)

## Archived Code

The notebooks in all folders starting with TRIALS_ are no longer in use.

[//]: # (--------------------)

## Areas of Interest

The `areas_of_interest` folder contains a shapefile with a list of aeras of interest and the following corresponding attributes: name of aoi, region (Dangermond, Goleta, Santa Barbara), item ids for the NAIP images where the aoi is located, include all years since 2012.  


[//]: # (--------------------)

## Custom Libraries

Custom libraries implemented to run the notebooks in this repository. The locations are:

	iceplant_detection/data_sampling_workflow/
		sample_rasters.py
		utility.py
        
	iceplant_detection/
		iceplant_detection_functions.py
		model_prep_and_evals.py

Descriptions:

### `sample_rasters.py`
Custom functions to:
   - sample random points inside a polygon    
   - sample raster values at a list of points
   - extract spectral and calendar values from NAIP iamges at specific coordinates
   - create and save auxiliary rasters to sample avg_lidar, max_lidar and min_lidar features using methods from scipy.ndimage 
   - convert points in csv to geodataframe

### `utility.py`
Mainly functions to access NAIP data in the Planetary Computer repository using pystac_client and planetary_computer libraries.
    
### `iceplant_detection_functions.py`
Functions to:
   - get a subset of a NAIP scene via the Planetary Computer's API
   - calculate NDVI and of a given raster with bands r,g,b,nir
   - create a dataframe with spectral and date features for each pixel in a NAIP scene or subset of one
   - apply a classification model to each pixel of a NAIP scene
   - convert model binary predictions back to image

### `model_prep_and_evals.py`
Functions to:
   - Divide a dataset into train/test sets with equal proportion of points from each NAIP scene
   - Nicely print statistics about the proportion of iceplant in training/test sets
   - Nicely print threshold metrics for predictions (accuracy, sensitivity, specificity, precsion, etc)

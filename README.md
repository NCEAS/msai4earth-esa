# msai4earth-esa

This repository gather codes for the joint project between NCEAS and Microsoft to develop a new approach Ecosystem assessment.

**IUCN_trials**
R notebooks exploring the spatial metrics from [IUCN Red List Assessments package](https://cran.r-project.org/web/packages/redlistr/vignettes/redlistr-vignette.html). 

**examples**
Examples of accessing and masking Landsat, GBIF and othe raster data.

**ecosystem_change**
- **aridity_layer**
  - ca_aridity_TRIALS.ipynb
  Different examples of how to manipulate data from the [Gridmet dataset](https://planetarycomputer.microsoft.com/dataset/gridmet).
 
  - ca_aridity_FUNCTIONS.ipynb
  Workflow to generate and compare CA aridity index and moisture domains maps. Based on [Gridmet dataset](https://planetarycomputer.microsoft.com/dataset/gridmet).
 
 
- **climate_layer**
  - tmax_TRIALS.ipynb
  Different examples of how to manipulate maximum average temperature data from the [Dayment Annual dataset](https://planetarycomputer.microsoft.com/dataset/daymet-annual-na)
  - clim_regions_TRIALS.ipynb
  Trial of grouping pixels into regions by thickening them usimg opencv and exporting them as geodataframe.
  - clim_outliers_TRIALS.ipynb
  Identifying some pixels with possible outlier values.
  
  - clim_avg_FUNCTIONS.ipynb
  Workflow to generate and compare 30-year average temperature normals and temperature regions maps. Based on [Dayment Annual dataset](https://planetarycomputer.microsoft.com/dataset/daymet-annual-na)
  - regions_FUNCTIONS.ipynb
  
 

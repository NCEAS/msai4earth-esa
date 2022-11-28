## This script imports the ouptut csv file of the classification validation and process it to
## create a long format geojson for further inspection of false positive and negative
##
## Julien Brun (brun@nceas.ucsb.edu)


librarian::shelf(tidyverse, janitor, sf)

# Read the data in and select relevant columns
validation_results_2020 <- read_csv("naip_iceplant_2020/validation_results_2020.csv") %>%
  janitor::clean_names() %>%
  select(plotid, center_lon, center_lat, email, pl_class, 
         starts_with("Category"), 
         starts_with("Validation")
         )


# Make long format
iceplant_long <- validation_results_2020 %>%
  pivot_longer(
    cols = starts_with("Category"),
    names_to = "category",
    names_prefix = "category_",
    values_to = "rank",
    values_drop_na = TRUE
  ) %>%
  filter(rank > 0) %>%
  select(-rank) %>%
  pivot_longer(
    cols = starts_with("Validation"),
    names_to = "validation",
    names_prefix = "validation_finished_",
    values_to = "confidence",
    values_drop_na = TRUE
  ) %>% 
  filter(confidence > 0) %>%
  select(-confidence)


# filter iceplant that have been classified as other vegetation
iceplant_other <- iceplant_long %>%
  filter(pl_class==1 & category=="non_iceplant_vegetation") %>%
  sf::st_as_sf(coords=c("center_lon","center_lat"), crs=4326) %>%
  st_write(dsn = "naip_iceplant_2020/iceplant_other_2020.geojson", 
           layer = "iceplant_other_2020.geojson",
           delete_dsn = T)

# filter iceplant that have been classified as iceplant
iceplant_iceplant <- iceplant_long %>%
  filter(pl_class==1 & category=="iceplant") %>%
  sf::st_as_sf(coords=c("center_lon","center_lat"), crs=4326) %>%
  st_write(dsn = "naip_iceplant_2020/iceplant_iceplant_2020.geojson", 
           layer = "iceplant_iceplant_2020.geojson",
           delete_dsn = T)




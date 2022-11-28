

librarian::shelf(tidyverse, janitor, naniar, sf)

# import the data
data_assessmt <- st_read("naip_iceplant_2020/validation_results_spectral_2020/validation_results_spectral_2020.shp") %>%
  replace_with_na_if(.predicate = is.numeric,
                     condition = ~.x == -9999)


# classification categories
#  0 = other_vegetation
#  1 = iceplant
#  2 = low_ndvi
#  3 = water

# add false positive, false negative flag for ice_plant
data_assessmt_flag <- data_assessmt %>%
  mutate(category_flag = case_when(
    ref_class == 1 & map_class == 1 ~ "iceplant_true_pos", # TRUE positive
    ref_class == 1 & map_class == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    ref_class == 0 & map_class == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    ref_class == 2 & map_class == 0 ~ "iceplant_falseneg_low", # FALSE negative others
    ref_class == 0 & map_class == 2 ~ "iceplant_falsepos_low", # FALSE positive others
    ref_class == 3 & map_class == 0 ~ "iceplant_falseneg_water", # FALSE negative others
    ref_class == 0 & map_class == 3 ~ "iceplant_falsepos_water", # FALSE positive others
    ref_class == 0 & map_class == 0 ~ "other_true_pos", # TRUE positive
    ref_class == 2 & map_class == 2 ~ "low_true_pos", # TRUE positive
    ref_class == 3 & map_class == 3 ~ "water_true_pos", # TRUE positive
    TRUE                      ~ "not_flagged"
  ))


# Subset the "pure" spectral signatures
truepos_signatures <- data_assessmt_flag %>% 
  filter(str_detect(category_flag, "true_pos")) 

# Compute the average "pure" spectral signatures
# truepos_signatures_mean <- truepos_signatures %>%
#   group_by(category_flag, .drop = FALSE) %>%
#   summarise_if(is.numeric, mean, na.rm = TRUE) %>% 
#   as.data.frame() %>%
#   select(-geometry)

truepos_signatures_mean_sd <- truepos_signatures %>%
  group_by(category_flag, .drop = FALSE) %>%
  summarise(across(where(is.numeric), 
                   list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
                   .names = "{.col}.{.fn}")) %>% 
  as.data.frame() %>%
  select(-geometry)


ggplot(truepos_signatures) + 
  geom_point(aes(x=category_flag, y = r), color = "red") +
  geom_point(aes(x=category_flag, y = g), color = "green") +
  geom_point(aes(x=category_flag, y = b), color = "blue") +
  geom_point(aes(x=category_flag, y = nir), color = "dark red") +
  geom_point(aes(x=category_flag, y = lidar), color = "grey") +
  coord_flip()

# Compute the "pure" spectral signatures
iceplant_false_signatures <- data_assessmt_flag %>% 
  filter(str_detect(category_flag, "_pos")) %>%
  group_by(category_flag, .drop = FALSE) %>%
  summarise_if(is.numeric, mean, na.rm = TRUE) %>% 
  as.data.frame() %>%
  select(-geometry)

ggplot(truepos_signatures) + 
  geom_point(aes(x= category_flag, y = r), color = "red") +
  geom_point(aes(x= category_flag, y = g), color = "green") +
  geom_point(aes(x= category_flag, y = b), color = "blue") +
  geom_point(aes(x= category_flag, y = nir), color = "dark red") +
  geom_point(aes(x= category_flag, y = lidar), color = "grey") +
  coord_flip()

### Inspecting spectral signature of different classification confusions


librarian::shelf(tidyverse, janitor, naniar, sf)

# file path
data_dir <- "/Users/brun/Data/msai4earth/naip_iceplant_2020/validation_results_spectral_2020"

# import the data
data_assessmt <- st_read(file.path(data_dir, "validation_results_spectral_2020.shp")) %>%
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



#### COMPUTE MEAN FOR ICEPLANT SIGNATURES ####

# Subset the "pure" spectral signatures
iceplant_signatures <- data_assessmt_flag %>% 
  filter(str_detect(category_flag, "iceplant")) 

# Compute the average "pure" spectral signatures
iceplant_signatures_mean <- iceplant_signatures %>%
  select(-c(year, month, day_in_yea, pl_which_r, low_confid, map_class, ref_class)) %>% # remove unwanted columns
  group_by(category_flag, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(mean = mean),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry)  # remove geometry


# make it a long format so ggplot is happy
iceplant_signatures_mean_long <- iceplant_signatures_mean %>%
  pivot_longer(cols = -c(category_flag, count),
               names_to = "channel_mean",
               values_to = "mean") %>%
  filter(!str_detect(channel_mean, "lidar")) %>% # Something is funky with the lidar data
  mutate(channel_mean = factor(channel_mean, 
                                levels = c("b", "g", "r", "nir", 
                                           "lidar", "min_lidar", "avg_lidar", "max_lidar")))





#### COMPUTE SD FOR ICEPLANT SIGNATURES ####

# Compute the average "pure" spectral signatures
iceplant_signatures_sd <- iceplant_signatures %>%
  select(-c(year, month, day_in_yea, pl_which_r, low_confid, map_class, ref_class)) %>% # remove unwanted columns
  group_by(category_flag, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(sd = sd),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry)  # remove geometry


# make it a long format so ggplot is happy
iceplant_signatures_sd_long <- iceplant_signatures_sd %>%
  pivot_longer(cols = -c(category_flag, count),
               names_to = "channel_sd",
               values_to = "sd") %>%
  filter(!str_detect(channel_sd, "lidar")) %>%
  mutate(channel_sd = factor(channel_sd, 
                          levels = c("b", "g", "r", "nir", 
                                     "lidar", "min_lidar", "avg_lidar", "max_lidar")))


#### Join mean and sd ####
iceplant_signatures_long <- iceplant_signatures_mean_long %>%
  left_join( iceplant_signatures_sd_long, by=c("category_flag", "channel_mean" = "channel_sd")) %>%
  select(-count.y) %>%
  rename(count = count.x,
         channel = channel_mean)

# plot the spectral signatures
ggplot(iceplant_signatures_long) + 
  geom_line(aes(x = channel, y = mean, color = category_flag, group = category_flag)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag, group = category_flag)) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()


# plot the spectral signatures
ggplot(iceplant_signatures_long) + 
  geom_ribbon(aes(x = channel, ymin = mean - 1.96 * sd / sqrt(count), ymax = mean + 1.96 * sd / sqrt(count), group = category_flag, fill = category_flag, alpha=0.1)) +  # confidence interval
  geom_line(aes(x = channel, y = mean, color = category_flag, group = category_flag)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag, group = category_flag)) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()




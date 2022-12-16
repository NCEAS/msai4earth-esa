### Inspecting spectral signature of different classification confusions
###

librarian::shelf(tidyverse, janitor, naniar, sf, spatialEco)

# file path
data_dir <- "~/Data/msai4earth/naip_iceplant_2020/validation_results_spectral_2020"

# import the data

# Previous data with visual check of classes
data_assessmt <- st_read(file.path(data_dir, "validation_results_spectral_2020.shp")) %>%
  replace_with_na_if(.predicate = is.numeric,
                     condition = ~.x == -9999)

# Extract the coordinates explicitly
data_assessmt_coord <- st_coordinates(data_assessmt) %>% 
  as_tibble() %>% setNames(c("lon","lat"))

data_assessmt <- cbind(data_assessmt,data_assessmt_coord)

# Read new data LSWE
data_lswe <- read.csv(file.path(data_dir, "LSWE_validation_points_results.csv")) %>%
  select(-c(X,class, which_raster, geometry)) %>%
  clean_names()

# join the two
data_assessmt_all <- left_join(data_assessmt, data_lswe, by = c("lon", "lat")) %>%
  mutate(class_change_flag = ifelse(lswe_result == map_class, 0, 1))


# classification categories
#  0 = other_vegetation
#  1 = iceplant
#  2 = low_ndvi
#  3 = water


#### Inspect spectral separability using Jeffries-Matusita distance ####

# create data frame of spectral signatures
spectral_signatures <- data_assessmt_all %>%
  select(r:nir) %>%
  st_drop_geometry()


# create the vector with classes
class_vect <- data_assessmt_all$ref_class %>% 
  factor(., labels=c("other_vegetation", "iceplant",
                           "low_ndvi", "water"))

# compute the spectral separability
spectral.separability(spectral_signatures, class_vect, jeffries.matusita = TRUE)

#                  other_vegetation iceplant low_ndvi    water
# other_vegetation         0.000000 1.213863 1.074290 1.349336
# iceplant                 1.213863 0.000000 1.285014 1.409361
# low_ndvi                 1.074290 1.285014 0.000000 1.251939
# water                    1.349336 1.409361 1.251939 0.000000




#### Add false positive, false negative flag for ice_plant ####
data_assessmt_flag <- data_assessmt_all %>%
  mutate(category_flag_lidar = case_when(
    ref_class == 1 & map_class == 1 ~ "iceplant_true_pos", # TRUE positive
    ref_class == 1 & map_class == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    ref_class == 0 & map_class == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    ref_class == 2 & map_class == 0 ~ "iceplant_falseneg_low", # FALSE negative low
    ref_class == 0 & map_class == 2 ~ "iceplant_falsepos_low", # FALSE positive low
    ref_class == 3 & map_class == 0 ~ "iceplant_falseneg_water", # FALSE negative water
    ref_class == 0 & map_class == 3 ~ "iceplant_falsepos_water", # FALSE positive water
    ref_class == 0 & map_class == 0 ~ "other_true_pos", # TRUE positive
    ref_class == 2 & map_class == 2 ~ "low_true_pos", # TRUE positive
    ref_class == 3 & map_class == 3 ~ "water_true_pos", # TRUE positive
    TRUE                      ~ "not_flagged"
  ),
  category_flag_lwse = case_when(
    ref_class == 1 & lswe_result == 1 ~ "iceplant_true_pos", # TRUE positive
    ref_class == 1 & lswe_result == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    ref_class == 0 & lswe_result == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    ref_class == 2 & lswe_result == 0 ~ "iceplant_falseneg_low", # FALSE negative low
    ref_class == 0 & lswe_result == 2 ~ "iceplant_falsepos_low", # FALSE positive low
    ref_class == 3 & lswe_result == 0 ~ "iceplant_falseneg_water", # FALSE negative water
    ref_class == 0 & lswe_result == 3 ~ "iceplant_falsepos_water", # FALSE positive water
    ref_class == 0 & lswe_result == 0 ~ "other_true_pos", # TRUE positive
    ref_class == 2 & lswe_result == 2 ~ "low_true_pos", # TRUE positive
    ref_class == 3 & lswe_result == 3 ~ "water_true_pos", # TRUE positive
    TRUE                      ~ "not_flagged"
  )) %>%
  select(-c(year, month, day_in_yea, pl_which_r, low_confid, map_class, ref_class, analysis_d)) # remove unused columns

# Add and id column
# data_assessmt_flag$id <- 1:nrow(data_assessmt_flag)
data_assessmt_flag <- data_assessmt_flag %>%
  relocate(plotid, .before = everything())


#### Quick stats between the two model versions ####

# Subset the iceplant related spectral signatures using old classification
iceplant_signatures_lidar <- data_assessmt_flag %>% 
  filter(str_detect(category_flag_lidar, "iceplant"))

iceplant_signatures_lidar %>% group_by(category_flag_lidar, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_lidar) *100) %>%
  st_drop_geometry()

# category_flag_lidar     count percent
# 1 iceplant_falseneg_low      14   11.0 
# 2 iceplant_falseneg_other     2    1.57
# 3 iceplant_falseneg_water     5    3.94
# 4 iceplant_falsepos_low      20   15.7 
# 5 iceplant_falsepos_other    37   29.1 
# 6 iceplant_true_pos          49   38.6 

# Subset the iceplant related spectral signatures using lwse
iceplant_signatures_lwse <- data_assessmt_flag %>% 
  filter(str_detect(category_flag_lwse, "iceplant"))

iceplant_signatures_lwse %>% group_by(category_flag_lwse, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_lwse) *100) %>%
  st_drop_geometry()

# category_flag_lwse      count percent
# 1 iceplant_falseneg_low      18   16.1 
# 2 iceplant_falseneg_other     6    5.36
# 3 iceplant_falseneg_water     9    8.04
# 4 iceplant_falsepos_low      20   17.9 
# 5 iceplant_falsepos_other    14   12.5 
# 6 iceplant_true_pos          45   40.2 




#### COMPUTE MEAN FOR ICEPLANT SIGNATURES ####


# Compute the average "pure" spectral signatures
iceplant_signatures_mean <- iceplant_signatures_lwse %>%
  group_by(category_flag_lwse, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(mean = mean),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry)  # remove geometry


# make it a long format so ggplot is happy
iceplant_signatures_mean_long <- iceplant_signatures_mean %>%
  pivot_longer(cols = -c(category_flag_lwse, count, plotid),
               names_to = "channel_mean",
               values_to = "mean") %>%
  filter(!str_detect(channel_mean, "lidar")) %>% # Something is funky with the lidar data
  mutate(channel_mean = factor(channel_mean, 
                                levels = c("b", "g", "r", "nir", 
                                           "lidar", "min_lidar", "avg_lidar", "max_lidar")))


#### COMPUTE SD FOR ICEPLANT SIGNATURES ####

# Compute the average "pure" spectral signatures
iceplant_signatures_sd <- iceplant_signatures_lwse %>%
  group_by(category_flag_lwse, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(sd = sd),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry)  # remove geometry


# make it a long format so ggplot is happy
iceplant_signatures_sd_long <- iceplant_signatures_sd %>%
  pivot_longer(cols = -c(category_flag_lwse, count, plotid),
               names_to = "channel_sd",
               values_to = "sd") %>%
  filter(!str_detect(channel_sd, "lidar")) %>%
  mutate(channel_sd = factor(channel_sd, 
                          levels = c("b", "g", "r", "nir", 
                                     "lidar", "min_lidar", "avg_lidar", "max_lidar")))


#### Join mean and sd ####
iceplant_signatures_long <- iceplant_signatures_mean_long %>%
  left_join( iceplant_signatures_sd_long, by=c("category_flag_lwse", "channel_mean" = "channel_sd")) %>%
  select(-count.y) %>%
  rename(count = count.x,
         channel = channel_mean)

# plot the spectral signatures
ggplot(iceplant_signatures_long) + 
  geom_line(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()


# plot the spectral signatures
ggplot(iceplant_signatures_long) + 
  geom_ribbon(aes(x = channel, ymin = mean - 1.96 * sd / sqrt(count), ymax = mean + 1.96 * sd / sqrt(count), group = category_flag_lwse, fill = category_flag_lwse, alpha=0.1)) +  # confidence interval
  geom_line(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()

iceplant_signatures_true <- 
  iceplant_signatures_long %>% filter(category_flag_lwse == "iceplant_true_pos")


# other vegetation false positive #
iceplant_false_otherveg_long <- data_assessmt_flag %>% 
  filter(category_flag_lwse == "iceplant_falsepos_other") %>%
  pivot_longer(cols = -c(id,naip_id, email, category_flag_lwse ,geometry),
               names_to = "channel",
               values_to = "dn") %>%
  filter(!str_detect(channel, "lidar")) %>%
  mutate(channel = factor(channel, 
                             levels = c("b", "g", "r", "nir", 
                                        "lidar", "min_lidar", "avg_lidar", "max_lidar")),
         dn = ifelse(dn < 0, 0, dn)
         )


# plot the spectral signatures
ggplot(iceplant_false_otherveg_long) + 
  geom_line(aes(x = channel, y = dn, group = id, color = id)) +
  # geom_point(aes(x = channel, y = dn, group = id, color = id)) +
  geom_line(data = iceplant_signatures_true, aes(x = channel, y = mean, group = id.x, color = "red", linewidth = 1.5)) +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()



#### Compute the euclidean distance between iceplant mean and false positive other veg and low NDVI####

# Get the mean in another df
iceplant_signatures_mean_true <- iceplant_signatures_mean %>% 
  filter(category_flag_lwse == "iceplant_true_pos") %>%
  select(r:nir)

# focus on the false positive other vegetation
iceplant_false_otherveg_dist <- iceplant_signatures %>%
  filter(category_flag_lwse == "iceplant_falsepos_other") %>% 
  mutate(eu_dist = sqrt((r-iceplant_signatures_mean_true$r)^2 + 
                          (g-iceplant_signatures_mean_true$g)^2 + 
                          (b-iceplant_signatures_mean_true$b)^2 + 
                          (nir-iceplant_signatures_mean_true$nir)^2)) %>%
  arrange(desc(eu_dist))

st_write(iceplant_false_otherveg_dist, file.path(data_dir,"iceplant_false_otherveg_dist.geojson"))

# focus on the false positive low NDVI
iceplant_false_low_dist <- iceplant_signatures %>%
  filter(category_flag_lwse == "iceplant_falsepos_low") %>% 
  mutate(eu_dist = sqrt((r-iceplant_signatures_mean_true$r)^2 + 
                          (g-iceplant_signatures_mean_true$g)^2 + 
                          (b-iceplant_signatures_mean_true$b)^2 + 
                          (nir-iceplant_signatures_mean_true$nir)^2)) %>%
  arrange(desc(eu_dist))

st_write(iceplant_false_low_dist, file.path(data_dir,"iceplant_false_low_dist.geojson"))

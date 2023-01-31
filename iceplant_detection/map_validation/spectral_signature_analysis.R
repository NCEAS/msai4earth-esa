############################################################################
### Inspecting spectral signature of different classification confusions ###
############################################################################

### brun@nceas.ucsb.edu 


# Loading the libraries
librarian::shelf(tidyverse, janitor, naniar, sf, spatialEco, patchwork)

# file path
data_dir <- "~/Data/msai4earth/naip_iceplant_2020/validation_results_spectral_2020"
data_dir_3070 <- file.path(data_dir, "model_3070")

# import the data

# Previous data with visual check of classes
data_assessmt <- st_read(file.path(data_dir, "validation_results_spectral_2020.shp")) %>%
  replace_with_na_if(.predicate = is.numeric,
                     condition = ~.x == -9999)

# Extract the coordinates explicitly
data_assessmt_coord <- st_coordinates(data_assessmt) %>% 
  as_tibble() %>% 
  setNames(c("lon","lat"))

data_assessmt <- cbind(data_assessmt, data_assessmt_coord)

# Read LSWE model output 
data_lswe <- read.csv(file.path(data_dir, "LSWE_validation_points_results.csv")) %>%
  select(-c(X,class, which_raster, geometry)) %>%
  clean_names()

# join the two
data_assessmt_all <- left_join(data_assessmt, data_lswe, by = c("lon", "lat")) %>%
  mutate(class_change_flag = ifelse(lswe_result == map_class, 0, 1))


# Read 3070 model output (2023-01-20)
data_3070 <- read.csv(file.path(data_dir_3070, "ice_veg_validation_pts_model3070.csv")) %>%
  clean_names() %>%
  tibble::rowid_to_column("plotid")  # add the plotid

# Read training data for 3070
training_3070 <- read.csv(file.path(data_dir_3070, "model3070_train_2020.csv")) %>%
  clean_names() %>%
  tibble::rowid_to_column("plotid")  # add the plotid


# classification categories
#
#  0 = other_vegetation
#  1 = iceplant
#  2 = low_ndvi
#  3 = water


########################################################################
#### Inspect spectral separability using Jeffries-Matusita distance ####
########################################################################

### LWSE
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


### 3070
# # create data frame of spectral signatures
# spectral_signatures_3070_spec <- data_3070 %>%
#   select(starts_with(c("r", "g", "b", "nir"))) %>%
#   select(-ref_class)
# 
# # create the vector with classes (should be the same, but to be safe)
# class_vect_3070 <- data_3070$ref_class %>% 
#   factor(., labels=c("other_vegetation", "iceplant",
#                      "low_ndvi", "water"))

# Sanity check, should be TRUE
# identical(class_vect, class_vect_3070)

# compute the spectral separability 
# spectral.separability(spectral_signatures_3070_spec, class_vect_3070, jeffries.matusita = TRUE)



###############################################################
#### Add false positive, false negative flag for ice_plant ####
###############################################################

data_assessmt_flag <- data_assessmt_all %>%
  mutate(category_flag_lidar = case_when(
    ref_class == 1 & map_class == 1 ~ "iceplant_true_pos", # TRUE positive
    ref_class == 1 & map_class == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    ref_class == 0 & map_class == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    ref_class == 1 & map_class == 2 ~ "iceplant_falseneg_low", # FALSE negative low
    ref_class == 2 & map_class == 1 ~ "iceplant_falsepos_low", # FALSE positive low
    ref_class == 1 & map_class == 3 ~ "iceplant_falseneg_water", # FALSE negative water
    ref_class == 3 & map_class == 1 ~ "iceplant_falsepos_water", # FALSE positive water
    ref_class == 0 & map_class == 0 ~ "other_true_pos", # TRUE positive
    ref_class == 2 & map_class == 2 ~ "low_true_pos", # TRUE positive
    ref_class == 3 & map_class == 3 ~ "water_true_pos", # TRUE positive
    TRUE                      ~ "not_flagged"
  ),
  category_flag_lwse = case_when(
    ref_class == 1 & lswe_result == 1 ~ "iceplant_true_pos", # TRUE positive
    ref_class == 1 & lswe_result == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    ref_class == 0 & lswe_result == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    ref_class == 1 & lswe_result == 2 ~ "iceplant_falseneg_low", # FALSE negative low
    ref_class == 2 & lswe_result == 1 ~ "iceplant_falsepos_low", # FALSE positive low
    ref_class == 1 & lswe_result == 3 ~ "iceplant_falseneg_water", # FALSE negative water
    ref_class == 3 & lswe_result == 1 ~ "iceplant_falsepos_water", # FALSE positive water
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


data_assessmt_flag_3070 <- data_3070 %>%
  mutate(category_flag_3070 = case_when(
  ref_class == 1 & mdl3070_class == 1 ~ "iceplant_true_pos", # TRUE positive
  ref_class == 1 & mdl3070_class == 0 ~ "iceplant_falseneg_other", # FALSE negative others
  ref_class == 0 & mdl3070_class == 1 ~ "iceplant_falsepos_other", # FALSE positive others
  ref_class == 1 & mdl3070_class == 2 ~ "iceplant_falseneg_low", # FALSE negative low
  ref_class == 2 & mdl3070_class == 1 ~ "iceplant_falsepos_low", # FALSE positive low
  ref_class == 1 & mdl3070_class == 3 ~ "iceplant_falseneg_water", # FALSE negative water
  ref_class == 3 & mdl3070_class == 1 ~ "iceplant_falsepos_water", # FALSE positive water
  ref_class == 0 & mdl3070_class == 0 ~ "other_true_pos", # TRUE positive
  ref_class == 2 & mdl3070_class == 2 ~ "low_true_pos", # TRUE positive
  ref_class == 3 & mdl3070_class == 3 ~ "water_true_pos", # TRUE positive
  TRUE ~ "not_flagged")
  )





####################################################
#### Quick stats between the model versions ####
####################################################

# Subset the iceplant related spectral signatures using old classification
iceplant_signatures_lidar <- data_assessmt_flag %>% 
  filter(str_detect(category_flag_lidar, "iceplant"))

iceplant_signatures_lidar %>% group_by(category_flag_lidar, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_lidar) *100) %>%
  st_drop_geometry()


# Subset the iceplant related spectral signatures using lwse
iceplant_signatures_lwse <- data_assessmt_flag %>% 
  filter(str_detect(category_flag_lwse, "iceplant"))

iceplant_signatures_lwse %>% group_by(category_flag_lwse, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_lwse) *100) %>%
  st_drop_geometry()

# category_flag_lwse      count percent
# 2 iceplant_falseneg_other     6    8.22
# 3 iceplant_falsepos_low       5    6.85
# 4 iceplant_falsepos_other    14   19.2 
# 6 iceplant_true_pos          45   61.6 

# Subset the iceplant related spectral signatures using lwse
iceplant_signatures_3070 <- data_assessmt_flag_3070 %>% 
  filter(str_detect(category_flag_3070, "iceplant")) 

# Compute the stats
iceplant_signatures_3070 %>% 
  group_by(category_flag_3070, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_3070)*100) 

# # A tibble: 6 × 3
# category_flag_3070      count percent
# 1 iceplant_falseneg_other    14   23.0 
# 2 iceplant_falsepos_low       2    3.28
# 3 iceplant_falsepos_other     6    9.84
# 4 iceplant_true_pos          39   63.9 



##############################################
#### COMPUTE MEAN FOR ICEPLANT SIGNATURES ####
##############################################


# Compute the average "pure" spectral signatures
iceplant_signatures_mean <- iceplant_signatures_lwse %>%
  group_by(category_flag_lwse, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(mean = mean),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry) %>% # remove geometry
  select(-plotid) # does not mean anything

iceplant_signatures_mean_3070 <- iceplant_signatures_3070 %>%
  group_by(category_flag_3070, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(mean = mean),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%
  select(-c(plotid, x, x_2, y, year, month, day_in_year, lswe_class, ref_class, mdl3070_class)) # does not mean anything

# make it a long format so ggplot is happy
iceplant_signatures_mean_long <- iceplant_signatures_mean %>%
  select(r, g, b, nir, category_flag_lwse, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_lwse, count),
               names_to = "channel_mean",
               values_to = "mean") %>%
  filter(!str_detect(channel_mean, "lidar")) %>% # Something is funky with the lidar data
  mutate(channel_mean = factor(channel_mean, 
                                levels = c("b", "g", "r", "nir")
                               )
         )

iceplant_signatures_mean_long_3070 <- iceplant_signatures_mean_3070 %>%
  select(r, g, b, nir, category_flag_3070, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_3070, count),
               names_to = "channel_mean",
               values_to = "mean") %>%
  mutate(channel_mean = factor(channel_mean, 
                               levels = c("b", "g", "r", "nir")
                               )
         )



############################################
#### COMPUTE SD FOR ICEPLANT SIGNATURES ####
############################################


# Compute the sd "pure" spectral signatures
iceplant_signatures_sd <- iceplant_signatures_lwse %>%
  group_by(category_flag_lwse, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(sd = sd),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  as.data.frame() %>%
  select(-geometry) %>%  # remove geometry
  select(-plotid)
  
# Compute the sd "pure" spectral signatures for 3070
iceplant_signatures_sd_3070 <- iceplant_signatures_3070 %>%
  group_by(category_flag_3070, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(sd = sd),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  select(-plotid) # does not mean anything 


# make it a long format so ggplot is happy
iceplant_signatures_sd_long <- iceplant_signatures_sd %>%
  select(r, g, b, nir, category_flag_lwse, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_lwse, count),
               names_to = "channel_sd",
               values_to = "sd") %>%
  filter(!str_detect(channel_sd, "lidar")) %>%
  mutate(channel_sd = factor(channel_sd, 
                          levels = c("b", "g", "r", "nir")
                          )
         )

# make it a long format so ggplot is happy for 3070
iceplant_signatures_sd_long_3070 <- iceplant_signatures_sd_3070 %>%
  select(r, g, b, nir, category_flag_3070, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_3070, count),
               names_to = "channel_sd",
               values_to = "sd") %>%
  filter(!str_detect(channel_sd, "lidar")) %>%
  mutate(channel_sd = factor(channel_sd, 
                             levels = c("b", "g", "r", "nir")
                             )
         )



#### Join mean and sd ####
iceplant_signatures_long <- iceplant_signatures_mean_long %>%
  left_join(iceplant_signatures_sd_long, by=c("category_flag_lwse", "channel_mean" = "channel_sd")) %>%
  select(-count.y) %>%
  rename(count = count.x,
         channel = channel_mean)

iceplant_signatures_long_3070 <- iceplant_signatures_mean_long_3070 %>%
  left_join(iceplant_signatures_sd_long_3070, by=c("category_flag_3070", "channel_mean" = "channel_sd")) %>%
  select(-count.y) %>%
  rename(count = count.x,
         channel = channel_mean)




##############
#### PLOT ####
##############


# plot the spectral signatures
g_mean <- ggplot(iceplant_signatures_long) + 
  geom_line(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral mean signatures") +
  theme_bw() + theme(legend.position="none")

# plot the spectral signatures
g_mean_3070 <- ggplot(iceplant_signatures_long_3070) + 
  geom_line(aes(x = channel, y = mean, color = category_flag_3070, group = category_flag_3070)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_3070, group = category_flag_3070)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral mean signatures 3070") +
  theme_bw()

g_mean + g_mean_3070 


# plot the spectral signatures
g_all <- ggplot(iceplant_signatures_long) + 
  geom_ribbon(aes(x = channel, ymin = mean - 1.96 * sd / sqrt(count), ymax = mean + 1.96 * sd / sqrt(count), group = category_flag_lwse, fill = category_flag_lwse, alpha=0.1)) +  # confidence interval
  geom_line(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_lwse, group = category_flag_lwse)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant Mean spectral signatures") +
  theme_bw() + theme(legend.position="none")

g_all_3070 <-ggplot(iceplant_signatures_long_3070) + 
  geom_ribbon(aes(x = channel, ymin = mean - 1.96 * sd / sqrt(count), ymax = mean + 1.96 * sd / sqrt(count), group = category_flag_3070, fill = category_flag_3070, alpha=0.1)) +  # confidence interval
  geom_line(aes(x = channel, y = mean, color = category_flag_3070, group = category_flag_3070)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_3070, group = category_flag_3070)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant Mean spectral signatures") +
  theme_bw() 

g_all + g_all_3070

iceplant_signatures_true <- 
  iceplant_signatures_long %>% 
  filter(category_flag_lwse == "iceplant_true_pos")

iceplant_signatures_true_3070 <- 
  iceplant_signatures_long_3070 %>% 
  filter(category_flag_3070 == "iceplant_true_pos")


# other vegetation false positive #
iceplant_false_otherveg_long <- data_assessmt_flag %>% 
  select(plotid, r, g, b, nir, category_flag_lwse) %>%
  filter(category_flag_lwse == "iceplant_falsepos_other") %>%
  pivot_longer(cols = -c(plotid, category_flag_lwse ,geometry),
               names_to = "channel",
               values_to = "dn") %>%
  mutate(channel = factor(channel, 
                             levels = c("b", "g", "r", "nir")),
         dn = ifelse(dn < 0, 0, dn)
         ) %>%
  st_drop_geometry()
  


# plot the spectral signatures
ggplot(iceplant_false_otherveg_long) + 
  geom_line(aes(x = channel, y = dn, group = plotid, color = plotid)) +
  # geom_point(aes(x = channel, y = dn, group = id, color = id)) +
  # geom_line(data = iceplant_signatures_true, aes(x = channel, y = mean, color = "red", linewidth = 1.5)) +
  ggtitle("Iceplant spectral signatures") +
  theme_bw()



########################################################################################################
#### Compute the euclidean distance between iceplant mean and false positive other veg and low NDVI ####
########################################################################################################

# Get the mean in another df
iceplant_signatures_mean_true <- iceplant_signatures_mean %>% 
  filter(category_flag_lwse == "iceplant_true_pos") %>%
  select(r:nir)


iceplant_signatures_mean_true_3070 <- iceplant_signatures_mean_3070 %>%
  filter(category_flag_3070 == "iceplant_true_pos")



# focus on the false positive other vegetation
iceplant_false_otherveg_dist <- iceplant_signatures_lwse %>%
  filter(category_flag_lwse == "iceplant_falsepos_other") %>% 
  mutate(eu_dist = sqrt((r-iceplant_signatures_mean_true$r)^2 + 
                          (g-iceplant_signatures_mean_true$g)^2 + 
                          (b-iceplant_signatures_mean_true$b)^2 + 
                          (nir-iceplant_signatures_mean_true$nir)^2),
         r_diff = r-iceplant_signatures_mean_true$r,
         g_diff = g-iceplant_signatures_mean_true$g,
         b_diff = b-iceplant_signatures_mean_true$b,
         nir_diff = nir-iceplant_signatures_mean_true$nir
         ) %>%
  arrange(desc(eu_dist))


# focus on the false positive other vegetation
iceplant_dist_3070 <- iceplant_signatures_3070 %>%
  # select(-c(plotid, x, x_2, y, year, month, day_in_year, lswe_class, ref_class, mdl3070_class, pts_crs)) %>%
  # filter(category_flag_3070 == "iceplant_falsepos_other") %>% 
  mutate(eu_dist_spectral = sqrt((r-iceplant_signatures_mean_true_3070$r)^2 + 
                          (g-iceplant_signatures_mean_true_3070$g)^2 + 
                          (b-iceplant_signatures_mean_true_3070$b)^2 + 
                          (nir-iceplant_signatures_mean_true_3070$nir)^2),
         eu_dist_entr = sqrt((r_entr-iceplant_signatures_mean_true_3070$r_entr)^2 + 
                                   (g_entr-iceplant_signatures_mean_true_3070$g_entr)^2 + 
                                   (b_entr-iceplant_signatures_mean_true_3070$b_entr)^2 + 
                                   (nir_entr-iceplant_signatures_mean_true_3070$nir_entr)^2),
         r_diff_spect = r-iceplant_signatures_mean_true_3070$r,
         g_diff_spect = g-iceplant_signatures_mean_true_3070$g,
         b_diff_spect = b-iceplant_signatures_mean_true_3070$b,
         nir_diff_spect = nir-iceplant_signatures_mean_true_3070$nir_entr,
         r_diff_entr = r_entr-iceplant_signatures_mean_true_3070$r_entr,
         g_diff_entr = g_entr-iceplant_signatures_mean_true_3070$g_entr,
         b_diff_entr = b_entr-iceplant_signatures_mean_true_3070$b_entr,
         nir_diff_entr = nir_entr-iceplant_signatures_mean_true_3070$nir_entr
  ) 



# Compute stat for the false positive other vegetation
iceplant_falsepos_otherveg_dist_3070 <- iceplant_dist_3070 %>%
  filter(category_flag_3070 == "iceplant_falsepos_other") %>%
  select(-x) %>%
  rename(x= x_2) %>%
  arrange(desc(eu_dist_spectral)) #farthest is the best

# Compute stat for the false negative other vegetation
iceplant_falseneg_otherveg_dist_3070 <- iceplant_dist_3070 %>%
  filter(category_flag_3070 == "iceplant_falseneg_other") %>%
  select(-x) %>%
  rename(x= x_2) %>%
  arrange((eu_dist_spectral)) # closest is the best

# write file
write_csv(iceplant_falsepos_otherveg_dist_3070, 
         file.path(data_dir_3070,"iceplant_falsepos_otherveg_dist_3070.csv"))

write_csv(iceplant_falseneg_otherveg_dist_3070, 
          file.path(data_dir_3070,"iceplant_falseneg_otherveg_dist_3070.csv"))

# write file
st_write(iceplant_false_otherveg_dist, 
         file.path(data_dir,"iceplant_falsepos_otherveg_dist.geojson"),
         delete_dsn = TRUE)


# # focus on the false positive low NDVI
# iceplant_false_low_dist <- iceplant_signatures %>%
#   filter(category_flag_lwse == "iceplant_falsepos_low") %>% 
#   mutate(eu_dist = sqrt((r-iceplant_signatures_mean_true$r)^2 + 
#                           (g-iceplant_signatures_mean_true$g)^2 + 
#                           (b-iceplant_signatures_mean_true$b)^2 + 
#                           (nir-iceplant_signatures_mean_true$nir)^2)) %>%
#   arrange(desc(eu_dist))
# 
# st_write(iceplant_false_low_dist, file.path(data_dir,"iceplant_false_low_dist.geojson"))

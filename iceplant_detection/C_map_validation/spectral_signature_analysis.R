############################################################################
### Inspecting spectral signature of different classification confusions ###
############################################################################

### brun@nceas.ucsb.edu 


# Loading the libraries
librarian::shelf(tidyverse, janitor, naniar, sf, spatialEco, patchwork)

# file path
data_dir <- "~/Data/msai4earth/naip_iceplant_2020/validation_results_spectral_2020"
# data_dir_3070 <- file.path(data_dir, "model_3070")
data_dir_ae5fp <- file.path(data_dir, "modelAE5_FP")
filename_validation_ae5fp <- "ceo-AE5FP_2020_model_map_validation-sample-data-2023-02-13.csv" 
filename_features_ae5fp <- "features-ceo-AE5FP_2020_model_map_validation-sample-data.csv" 

# import the data

# Read AE5_FP model output (2023-01-31)
data_ae5fp <- read_csv(file.path(data_dir_ae5fp, filename_validation_ae5fp)) %>%
  clean_names() %>%
  select(-c(imagery_attributions, sampleid))

# Read AE5_FP features (2023-01-31)
features_ae5fp <- read_csv(file.path(data_dir_ae5fp, filename_features_ae5fp)) %>%
  clean_names() 


# join the tables
full_data_ae5fp <- bind_cols(data_ae5fp,features_ae5fp) %>%
  drop_na(category)# note: could not do a join due to some mismatch in the decimal numbers; assuming the tables are in same order

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
spectral_signatures <- full_data_ae5fp %>%
  select(b, g, r, nir) %>%
  st_drop_geometry() %>%
  as.data.frame()


# create the vector with classes
class_vect <- full_data_ae5fp$category %>% 
  factor(., labels=c("other_vegetation", "iceplant",
                           "low ndvi (impervious surface)", "water"))

# compute the spectral separability
spectral.separability(spectral_signatures, class_vect, jeffries.matusita = TRUE)

#                                other_vegetation iceplant    low ndvi     water
# other_vegetation                      0.000000   1.284327   1.264195 1.413526
# iceplant                              1.284327   0.000000   1.144103 1.279324
# low ndvi.                             1.264195.  1.144103   0.000000 1.364555
# water                                 1.413526   1.279324   1.364555 0.000000


## Previous set of points
#                  other_vegetation iceplant low_ndvi    water
# other_vegetation         0.000000 1.213863 1.074290 1.349336
# iceplant                 1.213863 0.000000 1.285014 1.409361
# low_ndvi                 1.074290 1.285014 0.000000 1.251939
# water                    1.349336 1.409361 1.251939 0.000000



###############################################################
#### Add false positive, false negative flag for ice_plant ####
###############################################################

data_assessmt_flag_ae5fp <- full_data_ae5fp %>%
  mutate(category_flag_ae5fp = case_when(
    category == "iceplant"                      & pl_class == 1 ~ "iceplant_true_pos", # TRUE positive
    category == "iceplant"                      & pl_class == 0 ~ "iceplant_falseneg_other", # FALSE negative others
    category == "other_vegetation"              & pl_class == 1 ~ "iceplant_falsepos_other", # FALSE positive others
    category == "iceplant"                      & pl_class == 2 ~ "iceplant_falseneg_low", # FALSE negative low
    category == "low ndvi (impervious surface)" & pl_class == 1 ~ "iceplant_falsepos_low", # FALSE positive low
    category == "iceplant"                      & pl_class == 3 ~ "iceplant_falseneg_water", # FALSE negative water
    category == "water"                         & pl_class == 1 ~ "iceplant_falsepos_water", # FALSE positive water
    category == "other_vegetation"              & pl_class == 0 ~ "other_true_pos", # TRUE positive
    category == "low ndvi (impervious surface)" & pl_class == 2 ~ "low_true_pos", # TRUE positive
    category == "water"                         & pl_class == 3 ~ "water_true_pos", # TRUE positive
  TRUE ~ "not_flagged")
  )





####################################################
#### Quick stats between the model versions ####
####################################################



# Subset the iceplant related spectral signatures using lwse
# iceplant_signatures_lwse <- data_assessmt_flag %>% 
#   filter(str_detect(category_flag_lwse, "iceplant"))
# 
# iceplant_signatures_lwse %>% group_by(category_flag_lwse, .drop = FALSE) %>%
#   summarize(count = n(),
#             percent = n()/nrow(iceplant_signatures_lwse) *100) %>%
#   st_drop_geometry()

# category_flag_lwse      count percent
# 2 iceplant_falseneg_other     6    8.22
# 3 iceplant_falsepos_low       5    6.85
# 4 iceplant_falsepos_other    14   19.2 
# 6 iceplant_true_pos          45   61.6 

# Subset the iceplant related spectral signatures using lwse
iceplant_signatures_ae5fp <- data_assessmt_flag_ae5fp %>% 
  filter(str_detect(category_flag_ae5fp, "iceplant")) 

# Compute the stats
iceplant_signatures_ae5fp %>% 
  group_by(category_flag_ae5fp, .drop = FALSE) %>%
  summarize(count = n(),
            percent = n()/nrow(iceplant_signatures_ae5fp)*100) 

# category_flag_ae5fp     count percent
# 1 iceplant_falseneg_low       1   0.667
# 2 iceplant_falseneg_other     1   0.667
# 3 iceplant_falsepos_low      11   7.33 
# 4 iceplant_true_pos         137  91.3 


# model 3070
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

iceplant_signatures_mean_ae5fp <- iceplant_signatures_ae5fp %>%
  group_by(category_flag_ae5fp, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(mean = mean),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%
  select(-c(plotid, x, y, year, month, day_in_year)) # does not mean anything

# make it a long format so ggplot is happy

iceplant_signatures_mean_long_ae5fp <- iceplant_signatures_mean_ae5fp %>%
  select(r, g, b, nir, category_flag_ae5fp, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_ae5fp, count),
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
iceplant_signatures_sd_ae5fp <- iceplant_signatures_ae5fp %>%
  group_by(category_flag_ae5fp, .drop = FALSE) %>%  # Create groups
  summarise(across(where(is.numeric),   # compute the mean
                   list(sd = sd),
                   na.rm = TRUE,
                   .names = "{.col}"),
            count = n()) %>%  # add the count of the groups
  select(-plotid) # does not mean anything 


# make it a long format so ggplot is happy
iceplant_signatures_sd_long_ae5fp <- iceplant_signatures_sd_ae5fp %>%
  select(r, g, b, nir, category_flag_ae5fp, count) %>% # select the raw channels for now
  pivot_longer(cols = -c(category_flag_ae5fp, count),
               names_to = "channel_sd",
               values_to = "sd") %>%
  filter(!str_detect(channel_sd, "lidar")) %>%
  mutate(channel_sd = factor(channel_sd, 
                             levels = c("b", "g", "r", "nir")
                             )
         )



#### Join mean and sd ####
iceplant_signatures_long_ae5fp <- iceplant_signatures_mean_long_ae5fp %>%
  left_join(iceplant_signatures_sd_long_ae5fp, by=c("category_flag_ae5fp", "channel_mean" = "channel_sd")) %>%
  select(-count.y) %>%
  rename(count = count.x,
         channel = channel_mean)




##############
#### PLOT ####
##############


# plot the spectral signatures
g_mean_ae5fp <- ggplot(iceplant_signatures_long_ae5fp) + 
  geom_line(aes(x = channel, y = mean, color = category_flag_ae5fp, group = category_flag_ae5fp)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_ae5fp, group = category_flag_ae5fp)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant spectral mean signatures ae5fp") +
  theme_bw()

g_mean_ae5fp 


# plot the spectral signatures

g_all_ae5fp <-ggplot(iceplant_signatures_long_ae5fp) + 
  geom_ribbon(aes(x = channel, ymin = mean - 1.96 * sd / sqrt(count), ymax = mean + 1.96 * sd / sqrt(count), group = category_flag_ae5fp, fill = category_flag_ae5fp, alpha=0.1)) +  # confidence interval
  geom_line(aes(x = channel, y = mean, color = category_flag_ae5fp, group = category_flag_ae5fp)) + 
  geom_point(aes(x = channel, y = mean, color = category_flag_ae5fp, group = category_flag_ae5fp)) +
  ylim(70, 170) +
  labs(color = "Category") +
  ggtitle("Iceplant Mean spectral signatures") +
  theme_bw() 

g_all_ae5fp


iceplant_signatures_true_ae5fp <- 
  iceplant_signatures_long_ae5fp %>% 
  filter(category_flag_3070 == "iceplant_true_pos")





########################################################################################################
#### Compute the euclidean distance between iceplant mean and false positive other veg and low NDVI ####
########################################################################################################

# Get the mean in another df
iceplant_signatures_mean_true_ae5fp <- iceplant_signatures_mean_ae5fp %>%
  filter(category_flag_ae5fp == "iceplant_true_pos")


# focus on the false positive other vegetation
iceplant_dist_ae5fp <- iceplant_signatures_ae5fp %>%
  # select(-c(plotid, x, x_2, y, year, month, day_in_year, lswe_class, ref_class, mdl3070_class, pts_crs)) %>%
  # filter(category_flag_3070 == "iceplant_falsepos_other") %>% 
  mutate(eu_dist_spectral = sqrt((r-iceplant_signatures_mean_true_ae5fp$r)^2 + 
                          (g-iceplant_signatures_mean_true_ae5fp$g)^2 + 
                          (b-iceplant_signatures_mean_true_ae5fp$b)^2 + 
                          (nir-iceplant_signatures_mean_true_ae5fp$nir)^2),
         eu_dist_entr5 = sqrt((r_entr5-iceplant_signatures_mean_true_ae5fp$r_entr5)^2 + 
                                   (g_entr5-iceplant_signatures_mean_true_ae5fp$g_entr5)^2 + 
                                   (b_entr5-iceplant_signatures_mean_true_ae5fp$b_entr5)^2 + 
                                   (nir_entr5-iceplant_signatures_mean_true_ae5fp$nir_entr5)^2),
         r_diff_spect = r-iceplant_signatures_mean_true_ae5fp$r,
         g_diff_spect = g-iceplant_signatures_mean_true_ae5fp$g,
         b_diff_spect = b-iceplant_signatures_mean_true_ae5fp$b,
         nir_diff_spect = nir-iceplant_signatures_mean_true_ae5fp$nir_entr5,
         r_diff_entr5 = r_entr5-iceplant_signatures_mean_true_ae5fp$r_entr5,
         g_diff_entr5 = g_entr5-iceplant_signatures_mean_true_ae5fp$g_entr5,
         b_diff_entr5 = b_entr5-iceplant_signatures_mean_true_ae5fp$b_entr5,
         nir_diff_entr5 = nir_entr5-iceplant_signatures_mean_true_ae5fp$nir_entr5
  ) 



# Compute stat for the false positive low ndvi
iceplant_falsepos_otherveg_dist_ae5fp <- iceplant_dist_ae5fp %>%
  filter(category_flag_ae5fp == "iceplant_falsepos_low") %>%
  arrange(desc(eu_dist_spectral)) #farthest is the best

# Compute stat for the false negative other vegetation
iceplant_falseneg_otherveg_dist_ae5fp <- iceplant_dist_ae5fp %>%
  filter(category_flag_ae5fp == "iceplant_falseneg_other") %>%
  arrange((eu_dist_spectral)) # closest is the best

# write file
write_csv(iceplant_falsepos_otherveg_dist_ae5fp, 
         file.path(data_dir_ae5fp,"iceplant_falsepos_otherveg_dist_ae5fp.csv"))

write_csv(iceplant_falseneg_otherveg_dist_ae5fp, 
          file.path(data_dir_ae5fp,"iceplant_falseneg_otherveg_dist_ae5fp.csv"))

# # write file
# st_write(iceplant_false_otherveg_dist, 
#          file.path(data_dir,"iceplant_falsepos_otherveg_dist.geojson"),
#          delete_dsn = TRUE)


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

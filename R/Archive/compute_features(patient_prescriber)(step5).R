### STEP 5
### Feature engineering (patient-based demographics / prescriber info / prescriber info)

### INPUT: FULL_OPIOID_2018_ONE_FEATURE.csv, FULL_OPIOID_2018_ATLEASTTWO_FEATURE.csv
### OUTPUT: FULL_OPIOID_2018_FEATURE.csv

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2019

################################################################################

FULL_ONE <- read.csv(paste("FULL_OPIOID_", year ,"_ONE_FEATURE.csv", sep="")) # 84 features
FULL_ATLEASTTWO_1 <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_1_FEATURE.csv", sep=""))
FULL_ATLEASTTWO_2 <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_2_FEATURE.csv", sep=""))
FULL_ATLEASTTWO_3 <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_3_FEATURE.csv", sep=""))
FULL_ATLEASTTWO_4 <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_4_FEATURE.csv", sep=""))

FULL_ONE <- FULL_ONE %>% mutate(patient_zip = as.character(patient_zip))
FULL_ATLEASTTWO_1 <- FULL_ATLEASTTWO_1 %>% mutate(patient_zip = as.character(patient_zip))
FULL_ATLEASTTWO_2 <- FULL_ATLEASTTWO_2 %>% mutate(patient_zip = as.character(patient_zip))
FULL_ATLEASTTWO_3 <- FULL_ATLEASTTWO_3 %>% mutate(patient_zip = as.character(patient_zip))
FULL_ATLEASTTWO_4 <- FULL_ATLEASTTWO_4 %>% mutate(patient_zip = as.character(patient_zip))

FULL <- bind_rows(FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4)

rm(FULL_ONE)
rm(FULL_ATLEASTTWO_1)
rm(FULL_ATLEASTTWO_2)
rm(FULL_ATLEASTTWO_3)
rm(FULL_ATLEASTTWO_4)

################################################################################
############################### PATIENT-BASED ##################################
################################################################################

### Patient/prescriber/pharmacy HPI quartile
HPI <- read.csv("../CA/HPI.csv")
HPI <- HPI %>% mutate(Zip = as.character(Zip))
HPI_patient <- HPI %>% rename(patient_zip = Zip, patient_HPIQuartile = HPIQuartile) %>% select(-c(HPI))
HPI_prescriber <- HPI %>% rename(prescriber_zip = Zip, prescriber_HPIQuartile = HPIQuartile) %>% select(-c(HPI))
HPI_pharmacy <- HPI %>% rename(pharmacy_zip = Zip, pharmacy_HPIQuartile = HPIQuartile) %>% select(-c(HPI))


### Patient zip demographics
ZIP_DEMO <- read.csv("../CA/California_DemographicsByZip2020.csv")
ZIP_DEMO <- ZIP_DEMO %>% 
  rename(Zip = X......name) %>% 
  mutate(patient_zip = as.character(Zip),
         zip_pop = as.numeric(gsub(",", "", population)),
         zip_pop_density = population_density_sq_mi,
         median_household_income = as.numeric(gsub("[\\$,]", "", ifelse(median_household_income == "($1)", "0", gsub("\\(", "-", gsub("\\)", "", median_household_income))))),
         # educational_attainment_no_diploma = as.numeric(gsub("%", "", educational_attainment_no_diploma)),
         # educational_attainment_high_school = as.numeric(gsub("%", "", educational_attainment_high_school)),
         # educational_attainment_some_college = as.numeric(gsub("%", "", educational_attainment_some_college)),
         # educational_attainment_bachelors = as.numeric(gsub("%", "", educational_attainment_bachelors)),
         # educational_attainment_graduate = as.numeric(gsub("%", "", educational_attainment_graduate)),
         family_poverty_pct = as.numeric(gsub("%", "", family_poverty_pct)),
         unemployment_pct = as.numeric(gsub("%", "", unemployment_pct))
         ) %>%
  select(patient_zip, city_name, zip_pop, zip_pop_density, median_household_income, family_poverty_pct, unemployment_pct)
# median_household_income: set zip with median_household_income = $1 to 0.


### Combine
FULL <- FULL %>% 
  mutate(patient_zip = as.character(patient_zip),
         prescriber_zip = as.character(prescriber_zip),
         pharmacy_zip = as.character(pharmacy_zip)) %>%
  left_join(HPI_patient, by = "patient_zip") %>%
  left_join(HPI_prescriber, by = "prescriber_zip") %>%
  left_join(HPI_pharmacy, by = "pharmacy_zip") %>%
  mutate(patient_HPIQuartile = coalesce(patient_HPIQuartile, prescriber_HPIQuartile, pharmacy_HPIQuartile),
         prescriber_HPIQuartile = coalesce(prescriber_HPIQuartile, patient_HPIQuartile, pharmacy_HPIQuartile),
         pharmacy_HPIQuartile = coalesce(pharmacy_HPIQuartile, patient_HPIQuartile, prescriber_HPIQuartile)) %>% # fill NA 
  left_join(ZIP_DEMO, by = 'patient_zip')
  
# FULL_subset <- FULL %>% filter(is.na(prescriber_HPIQuartile))

rm(HPI)
rm(HPI_patient)
rm(HPI_pharmacy)
rm(HPI_prescriber)
rm(ZIP_DEMO)

# TEST <- FULL[1:20,]

################################################################################
############################# SPATIAL-INTENSITY ################################
################################################################################
# number (or proportion) of patients with opioid prescription in the LAST 6 MONTHS in pt_zip, 
# average per person MME (sum all MME/ pt_zip population) in pt_zip, number of pills/ pt_zip population in the LAST 6 MONTHS in pt_zip.

# file_list <- paste0('Patient_zip/PATIENT_ZIP_', year, '_', 0:20, '_TEMP.csv')
# PATIENT_ZIP_df_list <- lapply(file_list, read.csv)
# PATIENT_ZIP <- do.call(rbind, PATIENT_ZIP_df_list)
# 
# FULL <- FULL %>% left_join(PATIENT_ZIP, by = c("patient_zip", "date_filled"))
# 
# rm(PATIENT_ZIP_df_list)
# rm(PATIENT_ZIP)

################################################################################

previous_year = year - 1
FULL_PREVIOUS <- read.csv(paste("../RX_", previous_year, ".csv", sep=""))
FULL_PREVIOUS_OPIOID <- FULL_PREVIOUS %>% filter(class == 'Opioid')
rm(FULL_PREVIOUS)
# TEST <- FULL_PREVIOUS_OPIOID[1:100,]

PATIENT_ZIP <- FULL_PREVIOUS_OPIOID %>%
  group_by(patient_zip) %>%
  summarize(patient_zip_yr_num_prescriptions = n(),
            patient_zip_yr_num_patients = n_distinct(patient_id),
            patient_zip_yr_num_pharmacies = n_distinct(pharmacy_id),
            patient_zip_yr_avg_MME = mean(daily_dose, na.rm = TRUE),
            patient_zip_yr_avg_days = mean(days_supply, na.rm = TRUE),
            patient_zip_yr_avg_quantity = mean(quantity, na.rm = TRUE)) %>%
  mutate(patient_zip_yr_num_prescriptions_quartile = ntile(patient_zip_yr_num_prescriptions, 4),
         patient_zip_yr_num_patients_quartile = ntile(patient_zip_yr_num_patients, 4),
         patient_zip_yr_num_pharmacies_quartile = ntile(patient_zip_yr_num_pharmacies, 4),
         patient_zip_yr_avg_MME_quartile = ntile(patient_zip_yr_avg_MME, 4),
         patient_zip_yr_avg_days_quartile = ntile(patient_zip_yr_avg_days, 4),
         patient_zip_yr_avg_quantity_quartile = ntile(patient_zip_yr_avg_quantity, 4))

FULL <- FULL %>% left_join(PATIENT_ZIP, by = c("patient_zip"))

rm(PATIENT_ZIP)

################################################################################
############################## PRESCRIBER-BASED ################################
################################################################################

### Mean monthly (past 30 days)
### Opioid prescribing volume, length and MME, number of patients/prescriptions
### imported from python (compressed into zip for storage)
# file_list <- paste0('Prescriber/PRESCRIBER_', year,'_', 0:20, '_TEMP.csv') 
# PRESCRIBER_df_list <- lapply(file_list, read.csv)
# PRESCRIBER <- do.call(rbind, PRESCRIBER_df_list)
# 
# FULL <- FULL %>% left_join(PRESCRIBER, by = c("prescriber_id", "date_filled"))

# write.csv(FULL, paste("FULL_OPIOID_", year ,"_FEATURE.csv", sep=""), row.names = FALSE)

# rm(PRESCRIBER_df_list)
# rm(PRESCRIBER)

################################################################################
### Average MME by provider in previous year
# requires processing previous year data

PRESCRIBER <- FULL_PREVIOUS_OPIOID %>%
  group_by(prescriber_id) %>%
  summarize(prescriber_yr_num_prescriptions = n(),
            prescriber_yr_num_patients = n_distinct(patient_id),
            prescriber_yr_num_pharmacies = n_distinct(pharmacy_id),
            prescriber_yr_avg_MME = mean(daily_dose, na.rm = TRUE),
            prescriber_yr_avg_days = mean(days_supply, na.rm = TRUE),
            prescriber_yr_avg_quantity = mean(quantity, na.rm = TRUE)) %>%
  mutate(prescriber_yr_num_prescriptions_quartile = ntile(prescriber_yr_num_prescriptions, 4),
         prescriber_yr_num_patients_quartile = ntile(prescriber_yr_num_patients, 4),
         prescriber_yr_num_pharmacies_quartile = ntile(prescriber_yr_num_pharmacies, 4),
         prescriber_yr_avg_MME_quartile = ntile(prescriber_yr_avg_MME, 4),
         prescriber_yr_avg_days_quartile = ntile(prescriber_yr_avg_days, 4),
         prescriber_yr_avg_quantity_quartile = ntile(prescriber_yr_avg_quantity, 4))

FULL <- FULL %>% left_join(PRESCRIBER, by = c("prescriber_id"))

rm(PRESCRIBER)

################################################################################
############################## PHARMACY-BASED ##################################
################################################################################

PHARMACY <- FULL_PREVIOUS_OPIOID %>% 
  group_by(pharmacy_id) %>% 
  summarize(pharmacy_yr_num_prescriptions = n(),
            pharmacy_yr_num_patients = n_distinct(patient_id),
            pharmacy_yr_num_prescribers = n_distinct(prescriber_id),
            pharmacy_yr_avg_MME = mean(daily_dose, na.rm = TRUE),
            pharmacy_yr_avg_days = mean(days_supply, na.rm = TRUE),
            pharmacy_yr_avg_quantity = mean(quantity, na.rm = TRUE)) %>%
  mutate(pharmacy_yr_num_prescriptions_quartile = ntile(pharmacy_yr_num_prescriptions, 4),
         pharmacy_yr_num_patients_quartile = ntile(pharmacy_yr_num_patients, 4),
         pharmacy_yr_num_prescribers_quartile = ntile(pharmacy_yr_num_prescribers, 4),
         pharmacy_yr_avg_MME_quartile = ntile(pharmacy_yr_avg_MME, 4),
         pharmacy_yr_avg_days_quartile = ntile(pharmacy_yr_avg_days, 4),
         pharmacy_yr_avg_quantity_quartile = ntile(pharmacy_yr_avg_quantity, 4))

FULL <- FULL %>% left_join(PHARMACY, by = c("pharmacy_id"))

rm(PHARMACY)

rm(FULL_PREVIOUS_OPIOID)


################################################################################
################################## NDCcodes ####################################
################################################################################

NDC <- read.csv("../NDCcodes.csv")
NDC <- NDC %>% select(c(PRODUCTNDC, PRODUCTTYPENAME, DOSAGEFORMNAME))

FULL <- left_join(FULL, NDC, by = "PRODUCTNDC")
# TEST <- FULL[2774314:2774318, ]


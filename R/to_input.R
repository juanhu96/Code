### STEP 6
### Combine data and encode/convert to input form for riskSLIM
### Note that input is only for creating stumps

### INPUT: FULL_OPIOID_2018_FEATURE.csv
### OUTPUT: FULL_OPIOID_2018_INPUT.csv

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2018

FULL <- read.csv(paste("FULL_OPIOID_", year ,"_FEATURE.csv", sep=""))
# TEST <- FULL[1:20, ]

################################################################################
######################### ENCODE CATEGORICAL VARIABLES #########################
################################################################################

colnames(FULL)

# Gender
FULL <- FULL %>% mutate(patient_gender = ifelse(patient_gender == 'M', 0, 1))

# Drug and payment
FULL <- FULL %>% mutate(Codeine = ifelse(Codeine_MME > 0, 1, 0),
                        Hydrocodone = ifelse(Hydrocodone_MME > 0, 1, 0),
                        Oxycodone = ifelse(Oxycodone_MME > 0, 1, 0),
                        Morphine = ifelse(Morphine_MME > 0, 1, 0),
                        Hydromorphone = ifelse(Hydromorphone_MME > 0, 1, 0),
                        Methadone = ifelse(Methadone_MME > 0, 1, 0),
                        Fentanyl = ifelse(Fentanyl_MME > 0, 1, 0),
                        Oxymorphone = ifelse(Oxymorphone_MME > 0, 1, 0)) %>% 
  mutate(Medicaid = ifelse(payment == "Medicaid", 1, 0),
         CommercialIns = ifelse(payment == "CommercialIns", 1, 0),
         Medicare = ifelse(payment == "Medicare", 1, 0),
         CashCredit = ifelse(payment == "CashCredit", 1, 0),
         MilitaryIns = ifelse(payment == "MilitaryIns", 1, 0),
         WorkersComp = ifelse(payment == "WorkersComp", 1, 0),
         Other = ifelse(payment == "Other", 1, 0),
         IndianNation = ifelse(payment == "IndianNation", 1, 0))


################################################################################
############################# DROP FEATURES ####################################
################################################################################

colnames(FULL)

# drop irrelavent features or have been encoded (do not drop patient_id, patient_zip, date_filled, days_to_long_term!)
FULL_INPUT <- FULL %>% select(-c(patient_birth_year, prescriber_id, 
                                 prescriber_zip, pharmacy_id, pharmacy_zip, strength, # date_filled,
                                 MAINDOSE, drx_refill_number, drx_refill_authorized_number, 
                                 quantity_per_day, conversion, class, drug, payment, chronic,
                                 outliers, prescription_month, prescription_year, num_prescriptions, presc_until, 
                                 # num_prescriptions here is the total prescriptions (including future)
                                 prescription_id, overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                 num_alert, any_alert, overlap_lt, opioid_days, long_term, # days_to_long_term,
                                 city_name))

colnames(FULL_INPUT)

# 'median_household_income', 'family_poverty_pct', 'unemployment_pct'
FULL_INPUT <- FULL_INPUT %>% 
  mutate(zip_pop = as.numeric(gsub(",", "", zip_pop))) %>% 
  mutate(zip_pop_density_quartile = ntile(zip_pop_density, 4),
         median_household_income_quartile = ntile(median_household_income, 4),
         family_poverty_pct_quartile = ntile(family_poverty_pct, 4),
         unemployment_pct_quartile = ntile(unemployment_pct, 4),
         patient_zip_yr_num_prescriptions_per_pop = patient_zip_yr_num_prescriptions/zip_pop,
         patient_zip_yr_num_patients_per_pop = patient_zip_yr_num_patients/zip_pop) %>% 
  mutate(patient_zip_yr_num_prescriptions_per_pop_quartile = ntile(patient_zip_yr_num_prescriptions_per_pop, 4),
         patient_zip_yr_num_patients_per_pop_quartile = ntile(patient_zip_yr_num_patients_per_pop, 4))

colnames(FULL_INPUT)
TEST <- FULL_INPUT[0:20,]

write.csv(FULL_INPUT, paste("FULL_OPIOID_", year ,"_INPUT.csv", sep=""), row.names = FALSE)

################################################################################
####################### DOUBLE CHECK CLEANING ##################################
################################################################################

# Patient 69336149 has 60 prescriptions on the first date.
# Patient 68600209 has 28 prescriptions
# Patient 69566572 has 27 prescriptions

# These are example of outliers but not illicit behaviors
# (plus they did not become LT user)

################################################################################
##################### PRESCRIPTION ON FIRST DATE ###############################
################################################################################

# 0: 4416077
# 1: 562243
FULL_INPUT_FIRST <- FULL_INPUT %>% 
  group_by(patient_id) %>%
  filter(date_filled == min(date_filled)) %>%
  ungroup()

write.csv(FULL_INPUT_FIRST, paste("FULL_OPIOID_", year ,"_FIRST_INPUT.csv", sep=""), row.names = FALSE)


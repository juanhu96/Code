### Stratified sampling for flexible scoring table
library(caret)
library(dplyr)
library(tidyr)
library(arules)
library(fastDummies)

setwd("/export/storage_cures/CURES/Processed/")
year = 2018

FULL <- read.csv(paste("FULL_OPIOID_", year ,"_INPUT.csv", sep="")) # 10,025,197
FULL <- FULL %>% filter(!is.na(patient_HPIQuartile)) # 10,022,142

################################################################################
############################# SAMPLE BY PATIENTS ###############################
################################################################################

n_size = 5000
random_seed = 1234
set.seed(random_seed)

PATIENT <- FULL %>% 
  group_by(patient_id) %>% 
  summarise(
    long_term_user = ifelse(sum(long_term_180) > 0, 1, 0),
    patient_gender = first(patient_gender), 
    patient_HPIQuartile = first(patient_HPIQuartile)
  )

PATIENT <- PATIENT %>% mutate(composite_feature = paste(long_term_user, patient_gender, patient_HPIQuartile, sep = "_"))
# n_size <- min(PATIENT %>% count(composite_feature) %>% pull(n)) # 43012

# Perform stratified sampling based on the composite feature
# 5000 patients in each combo, 80000 in total
SAMPLED_PATIENT <- PATIENT %>%
  group_by(composite_feature) %>%
  sample_n(size = n_size)


# 278963 prescriptions from 80000 patients --> 264183 prescriptions 
SAMPLED_PRESC <- FULL %>% filter(patient_id %in% SAMPLED_PATIENT$patient_id) %>%
  dplyr::select(c(concurrent_MME, num_prescribers_past180, num_pharmacies_past180,
                  num_prior_prescriptions, concurrent_benzo,
                  dose_diff, concurrent_MME_diff, days_diff,
                  Codeine, Hydrocodone, Oxycodone, Morphine, HMFO,
                  Medicaid, Medicare, CashCredit,
                  ever_switch_drug, ever_switch_payment,
                  long_term_180,
                  patient_HPIQuartile, prescriber_HPIQuartile, pharmacy_HPIQuartile,
                  patient_zip_avg_days, patient_zip_avg_quantity, patient_zip_avg_MME, 
                  patient_zip_num_prescriptions_per_pop_quartile, patient_zip_num_patients_per_pop_quartile,
                  prescriber_yr_num_prescriptions_quartile, prescriber_yr_num_patients_quartile, prescriber_yr_num_pharmacies_quartile, 
                  prescriber_yr_avg_MME_quartile, prescriber_yr_avg_days_quartile, prescriber_yr_avg_quantity_quartile,
                  pharmacy_yr_num_prescriptions_quartile, pharmacy_yr_num_patients_quartile, 
                  pharmacy_yr_num_prescribers_quartile, pharmacy_yr_avg_MME_quartile, 
                  pharmacy_yr_avg_days_quartile, pharmacy_yr_avg_quantity_quartile,
                  age, patient_gender, zip_pop_density_quartile, median_household_income_quartile, family_poverty_pct_quartile, unemployment_pct_quartile)) %>%
  filter(!if_any(c(patient_HPIQuartile, prescriber_HPIQuartile, pharmacy_HPIQuartile,
                   patient_zip_avg_days, patient_zip_avg_quantity, patient_zip_avg_MME, 
                   patient_zip_num_prescriptions_per_pop_quartile, patient_zip_num_patients_per_pop_quartile,
                   prescriber_yr_num_prescriptions_quartile, prescriber_yr_num_patients_quartile, prescriber_yr_num_pharmacies_quartile, 
                   prescriber_yr_avg_MME_quartile, prescriber_yr_avg_days_quartile, prescriber_yr_avg_quantity_quartile,
                   pharmacy_yr_num_prescriptions_quartile, pharmacy_yr_num_patients_quartile, 
                   pharmacy_yr_num_prescribers_quartile, pharmacy_yr_avg_MME_quartile, 
                   pharmacy_yr_avg_days_quartile, pharmacy_yr_avg_quantity_quartile,
                   zip_pop_density_quartile, median_household_income_quartile, family_poverty_pct_quartile, unemployment_pct_quartile), is.na)) %>%
  mutate(any_prior_opioid = ifelse(num_prior_prescriptions > 0, 1, 0),
         num_prescribers = num_prescribers_past180,
         num_pharmacies = num_pharmacies_past180, 
         concurrent_benzo = ifelse(concurrent_benzo >= 1, 1, 0),
         dose_diff = ifelse(dose_diff > 0, 1, 0),
         concurrent_MME_diff = ifelse(concurrent_MME_diff > 0, 1, 0),
         days_diff = ifelse(days_diff > 0, 1, 0)) %>%
  dplyr::select(-c(num_prior_prescriptions, num_prescribers_past180, num_pharmacies_past180))

# dummies for quartiles
SAMPLED_PRESC <- dummy_cols(SAMPLED_PRESC, 
                            select_columns = c('patient_HPIQuartile', 'prescriber_HPIQuartile', 'pharmacy_HPIQuartile',
                                               'patient_zip_num_prescriptions_per_pop_quartile', 'patient_zip_num_patients_per_pop_quartile',
                                               'prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 'prescriber_yr_num_pharmacies_quartile', 
                                               'prescriber_yr_avg_MME_quartile', 'prescriber_yr_avg_days_quartile', 'prescriber_yr_avg_quantity_quartile',
                                               'pharmacy_yr_num_prescriptions_quartile', 'pharmacy_yr_num_patients_quartile', 
                                               'pharmacy_yr_num_prescribers_quartile', 'pharmacy_yr_avg_MME_quartile', 
                                               'pharmacy_yr_avg_days_quartile', 'pharmacy_yr_avg_quantity_quartile',
                                               'zip_pop_density_quartile', 'median_household_income_quartile', 'family_poverty_pct_quartile', 'unemployment_pct_quartile'),
                            remove_first_dummy = FALSE,
                            remove_selected_columns = TRUE)

write.csv(SAMPLED_PRESC, paste("Sample/SAMPLE_PATIENT_stratified_", n_size*2, "_random", random_seed, ".csv", sep = ""), row.names = FALSE)

# Count the number of 1's and 0's
# 67578 vs. 196605
Count_summary <- SAMPLED_PRESC %>% group_by(long_term_180) %>% summarise(Count = n())


colnames(SAMPLED_PRESC)



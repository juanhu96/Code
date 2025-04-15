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

# FULL <- read.csv(paste("FULL_OPIOID_", year ,"_FEATURE.csv", sep="")) # 10025192 prescriptions from 4847777 patients
# TEST <- FULL[1:20, ]

################################################################################
################################# FILTER #######################################
################################################################################

year = 2019
RAW <- read.csv(paste("../RX_", year, ".csv", sep="")) 

# 2018: 140566 prescriptions from 27935 patients
# 2019: 12474910 prescriptions from 1643910 patients
DUPLICATES <- RAW %>% group_by(across(-X)) %>% filter(n() > 1)

# 2018: 1396129 prescriptions from 393937 patients
# 2019: 1573266 prescriptions from 365692 patients
UNDERAGE <- RAW %>% filter(patient_birth_year > year - 18) 

# 2018: 421188 patients
# 2019: 1939144 patients
EXCLUDED_ID <- union(DUPLICATES$patient_id, UNDERAGE$patient_id) 

# 2018: 9763662 prescriptions from 4676803 patients
# 2019: 8127032 prescriptions from 3655084 patients
FULL <- FULL %>% filter(!patient_id %in% EXCLUDED_ID)


### DOUBLE CHECK CHRONIC
FULL_PREVIOUS <- read.csv(paste("../RX_", previous_year, ".csv", sep=""))
# 2017: 1787792 patients
# 2018: 1548327 patients
CHRONIC <- FULL_PREVIOUS %>%
  filter(class == 'Opioid') %>%
  mutate(prescription_month = month(as.POSIXlt(date_filled, format="%m/%d/%Y"))) %>%
  filter(prescription_month > 10) %>%
  select(patient_id) %>%
  distinct()
CHRONIC_vector <- unique(CHRONIC$patient_id)

# FINAL INPUT
# 2018: 6766011 (6766965) prescriptions from 4054269 patients
# 2019: 4893213 (4893385) prescriptions from 2908154 patients
FULL <- FULL %>% filter(!patient_id %in% CHRONIC_vector)

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

# drop irrelavent features or have been encoded (do not drop patient_id, patient_zip, date_filled, long_term, days_to_long_term!)
FULL_INPUT <- FULL %>% select(-c(patient_birth_year, prescriber_id, 
                                 prescriber_zip, pharmacy_id, pharmacy_zip, strength, # date_filled,
                                 MAINDOSE, drx_refill_number, drx_refill_authorized_number, 
                                 quantity_per_day, conversion, class, drug, payment, # chronic, # (only in 2018)
                                 outliers, prescription_month, prescription_year, num_prescriptions, presc_until, 
                                 # num_prescriptions here is the total prescriptions (including future)
                                 prescription_id, overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                 num_alert, any_alert, overlap_lt, opioid_days, # long_term, days_to_long_term,
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

# check columns (both 120 columns, now 127 columns)
FULL_INPUT_2018 <- read.csv(paste("FULL_OPIOID_2018_INPUT.csv", sep=""))
FULL_INPUT_2019 <- read.csv(paste("FULL_OPIOID_2019_INPUT.csv", sep=""))

colnames_full_input_2018 <- colnames(FULL_INPUT_2018)
colnames_full_input_2019 <- colnames(FULL_INPUT_2019)
diff_in_full_input_2018 <- setdiff(colnames_full_input_2018, colnames_full_input_2019)
diff_in_full_input_2019 <- setdiff(colnames_full_input_2019, colnames_full_input_2018)


min(FULL_INPUT$age)
max(FULL_INPUT$age)
max(FULL_INPUT$quantity)
max(FULL_INPUT$concurrent_MME)
max(FULL_INPUT$num_prescribers_past180)
max(FULL_INPUT$num_pharmacies_past180)
max(FULL_INPUT$num_prior_prescriptions)


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
# FULL_INPUT_FIRST <- FULL_INPUT %>% 
#   group_by(patient_id) %>%
#   filter(date_filled == min(date_filled)) %>%
#   ungroup()
# 
# write.csv(FULL_INPUT_FIRST, paste("FULL_OPIOID_", year ,"_FIRST_INPUT.csv", sep=""), row.names = FALSE)


################################################################################
################################################################################
################################################################################

FULL_INPUT_2018 <- FULL_INPUT_2018 %>% 
  mutate(long_acting = ifelse(DOSAGEFORMNAME %in% c("CAPSULE, EXTENDED RELEASE",
                                                    "PATCH, EXTENDED RELEASE",
                                                    "SUSPENSION, EXTENDED RELEASE", 
                                                    "TABLET, EXTENDED RELEASE",
                                                    "TABLET, FILM COATED, EXTENDED RELEASE"), 1, 0))

write.csv(FULL_INPUT_2018, "FULL_OPIOID_2018_INPUT.csv", row.names = FALSE)


FULL_INPUT_2019 <- FULL_INPUT_2019 %>% 
  mutate(long_acting = ifelse(DOSAGEFORMNAME %in% c("CAPSULE, EXTENDED RELEASE",
                                                    "PATCH, EXTENDED RELEASE",
                                                    "SUSPENSION, EXTENDED RELEASE", 
                                                    "TABLET, EXTENDED RELEASE",
                                                    "TABLET, FILM COATED, EXTENDED RELEASE"), 1, 0))

write.csv(FULL_INPUT_2019, "FULL_OPIOID_2019_INPUT.csv", row.names = FALSE)

################################################################################

LONGACT = FULL_INPUT_2019 %>% group_by(long_acting) %>% 
  summarize(TruePos = sum(long_term_180), n = n(), PropPos = TruePos/n) %>%
  rename(Group = long_acting) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Long acting drugs?")

ggplot(data=LONGACT) +
  geom_col(aes(x=Group, y=PropPos, fill=Group), width=0.5) +
  geom_text(aes(x=Group, y=-0.01, label=paste("n =", scales::comma(n))), size=3, color="black") +
  scale_fill_manual("Long acting opioids?", values=c("azure3", "azure4")) +
  scale_x_discrete("Long acting opioids?") +
  scale_y_continuous("Proportion that become long-term opioid users", labels=scales::percent, limits=c(-0.01,0.40)) +
  theme_bw() + theme(legend.position = "bottom")

ggsave("../Results/LongActingOpioids_Barplot.pdf", width = 6, height = 8)


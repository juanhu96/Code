### STEP 7
### SUMMARY STATISTICS OF THE FINAL TABLE

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2019

# 2018: 6766011 prescriptions from 4054269 patients
# 2019: 4893213 prescriptions from 2908154 patients
FULL_INPUT <- read.csv(paste("FULL_OPIOID_", year ,"_INPUT.csv", sep=""))
# TEST <- FULL_INPUT[0:20,]


FULL_INPUT_2018 <- read.csv("FULL_OPIOID_2018_INPUT.csv")
FULL_INPUT_2019 <- read.csv("FULL_OPIOID_2019_INPUT.csv")
FULL_INPUT <- rbind(FULL_INPUT_2018, FULL_INPUT_2019)

################################################################################
############################## PATIENT-LEVEL ###################################
################################################################################
# M: 0, F: 1
PATIENTS <- FULL_INPUT %>% 
  group_by(patient_id) %>%
  summarize(gender = first(patient_gender),
            age = first(age),
            num_prescriptions = n(),
            concurrent_benzo = ifelse(sum(concurrent_benzo) > 0, 1, 0),
            long_term_user = ifelse(sum(long_term_180) > 0, 1, 0),
            HPI_quartile = first(patient_HPIQuartile))

round(c(mean(PATIENTS$age, na.rm = TRUE), sd(PATIENTS$age, na.rm = TRUE),
        mean(PATIENTS$num_prescriptions, na.rm = TRUE), sd(PATIENTS$num_prescriptions, na.rm = TRUE)), 1)

PATIENTS_STAT <- PATIENTS %>% group_by(long_term_user) %>%
  summarize(num_patients = n(),
            male_count = sum(gender == 0),
            female_count = sum(gender == 1),
            avg_age = round(mean(age, na.rm = TRUE),1),
            sd_age = round(sd(age, na.rm = TRUE),1),
            avg_num_prescriptions = round(mean(num_prescriptions, na.rm = TRUE),1),
            sd_num_prescriptions = round(sd(num_prescriptions, na.rm = TRUE),1),
            concurrent_benzo_count = sum(concurrent_benzo),
            HPI_quartile_1_count = sum(HPI_quartile == 1, na.rm = TRUE),
            HPI_quartile_2_count = sum(HPI_quartile == 2, na.rm = TRUE),
            HPI_quartile_3_count = sum(HPI_quartile == 3, na.rm = TRUE),
            HPI_quartile_4_count = sum(HPI_quartile == 4, na.rm = TRUE))
PATIENTS_STAT <- t(PATIENTS_STAT)

################################################################################
############################ PRESCRIPTION-LEVEL ################################
################################################################################

LT <- FULL_INPUT %>% inner_join(PATIENTS %>% filter(long_term_user == 1), by = "patient_id")
NLT <- FULL_INPUT %>% inner_join(PATIENTS %>% filter(long_term_user == 0), by = "patient_id")

LT_STAT <- LT %>% summarize(avg_daily_mme = round(mean(daily_dose, na.rm = TRUE), 1),
                            sd_daily_mme = round(sd(daily_dose, na.rm = TRUE), 1),
                            avg_quantity = round(mean(quantity, na.rm = TRUE), 1),
                            sd_quantity = round(sd(quantity, na.rm = TRUE), 1),
                            avg_days_supply = round(mean(days_supply, na.rm = TRUE), 1),
                            sd_days_supply = round(sd(days_supply, na.rm = TRUE), 1),
                            Hydrocodone = sum(Hydrocodone == 1),
                            Oxycodone = sum(Oxycodone == 1),
                            Codeine = sum(Codeine == 1),
                            Morphine = sum(Morphine == 1),
                            Hydromorphone = sum(Hydromorphone == 1),
                            Methadone = sum(Methadone == 1),
                            Fentanyl = sum(Fentanyl == 1),
                            Oxymorphone = sum(Oxymorphone == 1),
                            CommercialIns = sum(CommercialIns == 1),
                            CashCredit = sum(CashCredit == 1),
                            Medicare = sum(Medicare == 1),
                            Medicaid = sum(Medicaid == 1),
                            MilitaryIns = sum(MilitaryIns == 1),
                            WorkersComp = sum(WorkersComp == 1),
                            Other = sum(Other == 1 | IndianNation == 1),
                            LongActing = sum(long_acting == 1))
LT_STAT <- t(LT_STAT)


NLT_STAT <- NLT %>% summarize(avg_daily_mme = round(mean(daily_dose, na.rm = TRUE), 1),
                            sd_daily_mme = round(sd(daily_dose, na.rm = TRUE), 1),
                            avg_quantity = round(mean(quantity, na.rm = TRUE), 1),
                            sd_quantity = round(sd(quantity, na.rm = TRUE), 1),
                            avg_days_supply = round(mean(days_supply, na.rm = TRUE), 1),
                            sd_days_supply = round(sd(days_supply, na.rm = TRUE), 1),
                            Hydrocodone = sum(Hydrocodone == 1),
                            Oxycodone = sum(Oxycodone == 1),
                            Codeine = sum(Codeine == 1),
                            Morphine = sum(Morphine == 1),
                            Hydromorphone = sum(Hydromorphone == 1),
                            Methadone = sum(Methadone == 1),
                            Fentanyl = sum(Fentanyl == 1),
                            Oxymorphone = sum(Oxymorphone == 1),
                            CommercialIns = sum(CommercialIns == 1),
                            CashCredit = sum(CashCredit == 1),
                            Medicare = sum(Medicare == 1),
                            Medicaid = sum(Medicaid == 1),
                            MilitaryIns = sum(MilitaryIns == 1),
                            WorkersComp = sum(WorkersComp == 1),
                            Other = sum(Other == 1 | IndianNation == 1),
                            LongActing = sum(long_acting == 1))

NLT_STAT <- t(NLT_STAT)

# ALL PRESCTIPIONS
round(c(mean(FULL_INPUT$daily_dose, na.rm = TRUE), sd(FULL_INPUT$daily_dose, na.rm = TRUE),
        mean(FULL_INPUT$quantity, na.rm = TRUE), sd(FULL_INPUT$quantity, na.rm = TRUE),
        mean(FULL_INPUT$days_supply, na.rm = TRUE), sd(FULL_INPUT$days_supply, na.rm = TRUE)), 1)


################################################################################
######################### NUM OF PRESCRIPTIONS #################################
################################################################################

summary_stats <- FULL_INPUT_2019 %>% 
  mutate(num_prescriptions = num_prior_prescriptions + 1) %>%
  group_by(num_prescriptions) %>%
  summarize(
    count_rows = n(), 
    count_long_term_180 = sum(long_term_180, na.rm = TRUE)
  )





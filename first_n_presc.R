### Use the first 1, 2, ..., n prescription for prediction

library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv") #8141616
FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")
################################################################################
################################################################################
################################################################################
### What would be the outcome variable then? 
### If a patient is ever going to be a long-term user?

## Not the first prescription, but the first prescription date
FULL_FIRST <- FULL_2018 %>% group_by(patient_id) %>% summarize(first_presc_date = date_filled[1],
                                                               long_term_ever = ifelse(sum(long_term) > 0, 1, 0)) # 4409867
FULL_2018 <- left_join(FULL_2018, FULL_FIRST, by = "patient_id") %>% filter(date_filled == first_presc_date) # 4509471, 88935 of them have multiple prescription on their first date

## For patient with multiple prescriptions on the first date, use the last one since it has the aggregate information
TEST <- FULL_2018 %>% group_by(patient_id) %>% filter(row_number()==n())

TEST <- TEST %>% ungroup()
write.csv(TEST, "../Data/FULL_2018_FIRST_LONGTERM.csv", row.names = FALSE)

TEST <- TEST %>% ungroup() %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                         prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                         conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                         DEASCHEDULE, MAINDOSE, payment, prescription_year, drug_payment, overlap, # alert1, alert2, alert3, alert4, alert5, alert6,
                                         long_term_180, days_to_long_term, num_alert, opioid_days, long_term, first_presc_date))

write.csv(TEST, "../Data/FULL_2018_FIRST_LONGTERM_INPUT.csv", row.names = FALSE)


################################################################################
################################################################################
################################################################################
## Second & third prescription date
TEMP <- FULL_2018 %>% group_by(patient_id, date_filled) %>% summarize(num_presc = n())
NEW_TEMP <- TEMP %>% group_by(patient_id) %>% summarize(first_presc_date = date_filled[1],
                                                        second_presc_date = date_filled[2],
                                                        third_presc_date = date_filled[3])

### Fill the second & third prescription date with a dummy date (e.g., 2020)
FULL_2018 <- left_join(FULL_2018, NEW_TEMP, by = "patient_id")
FULL_2018[c("second_presc_date", "third_presc_date")][is.na(FULL_2018[c("second_presc_date", "third_presc_date")])] <- "01/01/2020"

FULL_THIRD <- FULL_2018 %>% filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(third_presc_date, format = "%m/%d/%Y"))
FULL_SECOND <- FULL_2018 %>% filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(second_presc_date, format = "%m/%d/%Y"))

### Take patient's prescriptions up to their second/third prescription date
### Aggregate their information and export as input format
PATIENT_THIRD <- FULL_THIRD %>% group_by(patient_id) %>% summarize(age = age[1],
                                                                   patient_gender = patient_gender[1],
                                                                   num_presc = max(num_presc),
                                                                   quantity = mean(quantity),
                                                                   days_supply = mean(days_supply),
                                                                   quantity_per_day = mean(quantity_per_day),
                                                                   daily_dose = mean(daily_dose),
                                                                   total_dose = mean(total_dose),
                                                                   concurrent_MME = mean(concurrent_MME),
                                                                   concurrent_methadone_MME = mean(concurrent_methadone_MME),
                                                                   num_prescribers = max(num_prescribers),
                                                                   num_pharmacies = max(num_pharmacies),
                                                                   concurrent_benzo = max(concurrent_benzo),
                                                                   concurrent_benzo_same = max(concurrent_benzo_same),
                                                                   concurrent_benzo_diff = max(concurrent_benzo_diff),
                                                                   consecutive_days = max(consecutive_days),
                                                                   Codeine = sum(Codeine),
                                                                   Hydrocodone = sum(Hydrocodone),
                                                                   Oxycodone = sum(Oxycodone),
                                                                   Morphine = sum(Morphine),
                                                                   Hydromorphone = sum(Hydromorphone),
                                                                   Methadone = sum(Methadone),
                                                                   Fentanyl = sum(Fentanyl),
                                                                   Oxymorphone = sum(Oxymorphone),
                                                                   Medicaid = sum(Medicaid),
                                                                   CommercialIns = sum(CommercialIns),
                                                                   Medicare = sum(Medicare),
                                                                   CashCredit = sum(CashCredit),
                                                                   MilitaryIns = sum(MilitaryIns),
                                                                   WorkersComp = sum(WorkersComp),
                                                                   Other = sum(Other),
                                                                   IndianNation = sum(IndianNation),
                                                                   alert1 = sum(alert1),
                                                                   alert2 = sum(alert2),
                                                                   alert3 = sum(alert3),
                                                                   alert4 = sum(alert4),
                                                                   alert5 = sum(alert5),
                                                                   alert6 = sum(alert6),
                                                                   num_alert = sum(num_alert),
                                                                   opioid_days = max(opioid_days),
                                                                   num_presc = max(num_presc),
                                                                   dose_diff = max(dose_diff),
                                                                   MME_diff = max(MME_diff),
                                                                   days_diff = max(days_diff),
                                                                   switch_drug = max(switch_drug),
                                                                   switch_payment = max(switch_payment),
                                                                   Codeine_MME = max(Codeine_MME),
                                                                   Hydrocodone_MME = max(Hydrocodone_MME),
                                                                   Oxycodone_MME = max(Oxycodone_MME),
                                                                   Morphine_MME = max(Morphine_MME),
                                                                   Hydromorphone_MME = max(Hydromorphone_MME),
                                                                   Methadone_MME = max(Methadone_MME),
                                                                   Fentanyl_MME = max(Fentanyl_MME),
                                                                   Oxymorphone_MME = max(Oxymorphone_MME),
                                                                   Codeine_Medicaid = max(Codeine_Medicaid),
                                                                   Codeine_CommercialIns = max(Codeine_CommercialIns),
                                                                   Codeine_Medicare = max(Codeine_Medicare),
                                                                   Codeine_CashCredit = max(Codeine_CashCredit),
                                                                   Codeine_MilitaryIns = max(Codeine_MilitaryIns),
                                                                   Codeine_WorkersComp = max(Codeine_WorkersComp),
                                                                   Codeine_Other = max(Codeine_Other),
                                                                   Codeine_IndianNation = max(Codeine_IndianNation),
                                                                   Hydrocodone_Medicaid = max(Hydrocodone_Medicaid),
                                                                   Hydrocodone_CommercialIns = max(Hydrocodone_CommercialIns),
                                                                   Hydrocodone_Medicare = max(Hydrocodone_Medicare),
                                                                   Hydrocodone_CashCredit = max(Hydrocodone_CashCredit),
                                                                   Hydrocodone_MilitaryIns = max(Hydrocodone_MilitaryIns),
                                                                   Hydrocodone_WorkersComp = max(Hydrocodone_WorkersComp),
                                                                   Hydrocodone_Other = max(Hydrocodone_Other),
                                                                   Hydrocodone_IndianNation = max(Hydrocodone_IndianNation),
                                                                   Oxycodone_Medicaid = max(Oxycodone_Medicaid),
                                                                   Oxycodone_CommercialIns = max(Oxycodone_CommercialIns),
                                                                   Oxycodone_Medicare = max(Oxycodone_Medicare),
                                                                   Oxycodone_CashCredit = max(Oxycodone_CashCredit),
                                                                   Oxycodone_MilitaryIns = max(Oxycodone_MilitaryIns),
                                                                   Oxycodone_WorkersComp = max(Oxycodone_WorkersComp),
                                                                   Oxycodone_Other = max(Oxycodone_Other),
                                                                   Oxycodone_IndianNation = max(Oxycodone_IndianNation),
                                                                   Morphine_Medicaid = max(Morphine_Medicaid),
                                                                   Morphine_CommercialIns = max(Morphine_CommercialIns),
                                                                   Morphine_Medicare = max(Morphine_Medicare),
                                                                   Morphine_CashCredit = max(Morphine_CashCredit),
                                                                   Morphine_MilitaryIns = max(Morphine_MilitaryIns),
                                                                   Morphine_WorkersComp = max(Morphine_WorkersComp),
                                                                   Morphine_Other = max(Morphine_Other),
                                                                   Morphine_IndianNation = max(Morphine_IndianNation),
                                                                   Hydromorphone_Medicaid = max(Hydromorphone_Medicaid),
                                                                   Hydromorphone_CommercialIns = max(Hydromorphone_CommercialIns),
                                                                   Hydromorphone_Medicare = max(Hydromorphone_Medicare),
                                                                   Hydromorphone_CashCredit = max(Hydromorphone_CashCredit),
                                                                   Hydromorphone_MilitaryIns = max(Hydromorphone_MilitaryIns),
                                                                   Hydromorphone_WorkersComp = max(Hydromorphone_WorkersComp),
                                                                   Hydromorphone_Other = max(Hydromorphone_Other),
                                                                   Hydromorphone_IndianNation = max(Hydromorphone_IndianNation),
                                                                   Methadone_Medicaid = max(Methadone_Medicaid),
                                                                   Methadone_CommercialIns = max(Methadone_CommercialIns),
                                                                   Methadone_Medicare = max(Methadone_Medicare),
                                                                   Methadone_CashCredit = max(Methadone_CashCredit),
                                                                   Methadone_MilitaryIns = max(Methadone_MilitaryIns),
                                                                   Methadone_WorkersComp = max(Methadone_WorkersComp),
                                                                   Methadone_Other = max(Methadone_Other),
                                                                   Methadone_IndianNation = max(Methadone_IndianNation),
                                                                   Fentanyl_Medicaid = max(Fentanyl_Medicaid),
                                                                   Fentanyl_CommercialIns = max(Fentanyl_CommercialIns),
                                                                   Fentanyl_Medicare = max(Fentanyl_Medicare),
                                                                   Fentanyl_CashCredit = max(Fentanyl_CashCredit),
                                                                   Fentanyl_MilitaryIns = max(Fentanyl_MilitaryIns),
                                                                   Fentanyl_WorkersComp = max(Fentanyl_WorkersComp),
                                                                   Fentanyl_Other = max(Fentanyl_Other),
                                                                   Fentanyl_IndianNation = max(Fentanyl_IndianNation),
                                                                   Oxymorphone_Medicaid = max(Oxymorphone_Medicaid),
                                                                   Oxymorphone_CommercialIns = max(Oxymorphone_CommercialIns),
                                                                   Oxymorphone_Medicare = max(Oxymorphone_Medicare),
                                                                   Oxymorphone_CashCredit = max(Oxymorphone_CashCredit),
                                                                   Oxymorphone_MilitaryIns = max(Oxymorphone_MilitaryIns),
                                                                   Oxymorphone_WorkersComp = max(Oxymorphone_WorkersComp),
                                                                   Oxymorphone_Other = max(Oxymorphone_Other),
                                                                   Oxymorphone_IndianNation = max(Oxymorphone_IndianNation),
                                                                   long_term_ever = ifelse(sum(long_term) > 0, 1, 0))

write.csv(PATIENT_THIRD, "../Data/PATIENT_2018_THIRD_LONGTERM_INPUT.csv", row.names = FALSE)


PATIENT_SECOND <- FULL_SECOND %>% group_by(patient_id) %>% summarize(age = age[1],
                                                                     patient_gender = patient_gender[1],
                                                                     num_presc = max(num_presc),
                                                                     quantity = mean(quantity),
                                                                     days_supply = mean(days_supply),
                                                                     quantity_per_day = mean(quantity_per_day),
                                                                     daily_dose = mean(daily_dose),
                                                                     total_dose = mean(total_dose),
                                                                     concurrent_MME = mean(concurrent_MME),
                                                                     concurrent_methadone_MME = mean(concurrent_methadone_MME),
                                                                     num_prescribers = max(num_prescribers),
                                                                     num_pharmacies = max(num_pharmacies),
                                                                     concurrent_benzo = max(concurrent_benzo),
                                                                     concurrent_benzo_same = max(concurrent_benzo_same),
                                                                     concurrent_benzo_diff = max(concurrent_benzo_diff),
                                                                     consecutive_days = max(consecutive_days),
                                                                     Codeine = sum(Codeine),
                                                                     Hydrocodone = sum(Hydrocodone),
                                                                     Oxycodone = sum(Oxycodone),
                                                                     Morphine = sum(Morphine),
                                                                     Hydromorphone = sum(Hydromorphone),
                                                                     Methadone = sum(Methadone),
                                                                     Fentanyl = sum(Fentanyl),
                                                                     Oxymorphone = sum(Oxymorphone),
                                                                     Medicaid = sum(Medicaid),
                                                                     CommercialIns = sum(CommercialIns),
                                                                     Medicare = sum(Medicare),
                                                                     CashCredit = sum(CashCredit),
                                                                     MilitaryIns = sum(MilitaryIns),
                                                                     WorkersComp = sum(WorkersComp),
                                                                     Other = sum(Other),
                                                                     IndianNation = sum(IndianNation),
                                                                     alert1 = sum(alert1),
                                                                     alert2 = sum(alert2),
                                                                     alert3 = sum(alert3),
                                                                     alert4 = sum(alert4),
                                                                     alert5 = sum(alert5),
                                                                     alert6 = sum(alert6),
                                                                     num_alert = sum(num_alert),
                                                                     opioid_days = max(opioid_days),
                                                                     num_presc = max(num_presc),
                                                                     dose_diff = max(dose_diff),
                                                                     MME_diff = max(MME_diff),
                                                                     days_diff = max(days_diff),
                                                                     switch_drug = max(switch_drug),
                                                                     switch_payment = max(switch_payment),
                                                                     Codeine_MME = max(Codeine_MME),
                                                                     Hydrocodone_MME = max(Hydrocodone_MME),
                                                                     Oxycodone_MME = max(Oxycodone_MME),
                                                                     Morphine_MME = max(Morphine_MME),
                                                                     Hydromorphone_MME = max(Hydromorphone_MME),
                                                                     Methadone_MME = max(Methadone_MME),
                                                                     Fentanyl_MME = max(Fentanyl_MME),
                                                                     Oxymorphone_MME = max(Oxymorphone_MME),
                                                                     Codeine_Medicaid = max(Codeine_Medicaid),
                                                                     Codeine_CommercialIns = max(Codeine_CommercialIns),
                                                                     Codeine_Medicare = max(Codeine_Medicare),
                                                                     Codeine_CashCredit = max(Codeine_CashCredit),
                                                                     Codeine_MilitaryIns = max(Codeine_MilitaryIns),
                                                                     Codeine_WorkersComp = max(Codeine_WorkersComp),
                                                                     Codeine_Other = max(Codeine_Other),
                                                                     Codeine_IndianNation = max(Codeine_IndianNation),
                                                                     Hydrocodone_Medicaid = max(Hydrocodone_Medicaid),
                                                                     Hydrocodone_CommercialIns = max(Hydrocodone_CommercialIns),
                                                                     Hydrocodone_Medicare = max(Hydrocodone_Medicare),
                                                                     Hydrocodone_CashCredit = max(Hydrocodone_CashCredit),
                                                                     Hydrocodone_MilitaryIns = max(Hydrocodone_MilitaryIns),
                                                                     Hydrocodone_WorkersComp = max(Hydrocodone_WorkersComp),
                                                                     Hydrocodone_Other = max(Hydrocodone_Other),
                                                                     Hydrocodone_IndianNation = max(Hydrocodone_IndianNation),
                                                                     Oxycodone_Medicaid = max(Oxycodone_Medicaid),
                                                                     Oxycodone_CommercialIns = max(Oxycodone_CommercialIns),
                                                                     Oxycodone_Medicare = max(Oxycodone_Medicare),
                                                                     Oxycodone_CashCredit = max(Oxycodone_CashCredit),
                                                                     Oxycodone_MilitaryIns = max(Oxycodone_MilitaryIns),
                                                                     Oxycodone_WorkersComp = max(Oxycodone_WorkersComp),
                                                                     Oxycodone_Other = max(Oxycodone_Other),
                                                                     Oxycodone_IndianNation = max(Oxycodone_IndianNation),
                                                                     Morphine_Medicaid = max(Morphine_Medicaid),
                                                                     Morphine_CommercialIns = max(Morphine_CommercialIns),
                                                                     Morphine_Medicare = max(Morphine_Medicare),
                                                                     Morphine_CashCredit = max(Morphine_CashCredit),
                                                                     Morphine_MilitaryIns = max(Morphine_MilitaryIns),
                                                                     Morphine_WorkersComp = max(Morphine_WorkersComp),
                                                                     Morphine_Other = max(Morphine_Other),
                                                                     Morphine_IndianNation = max(Morphine_IndianNation),
                                                                     Hydromorphone_Medicaid = max(Hydromorphone_Medicaid),
                                                                     Hydromorphone_CommercialIns = max(Hydromorphone_CommercialIns),
                                                                     Hydromorphone_Medicare = max(Hydromorphone_Medicare),
                                                                     Hydromorphone_CashCredit = max(Hydromorphone_CashCredit),
                                                                     Hydromorphone_MilitaryIns = max(Hydromorphone_MilitaryIns),
                                                                     Hydromorphone_WorkersComp = max(Hydromorphone_WorkersComp),
                                                                     Hydromorphone_Other = max(Hydromorphone_Other),
                                                                     Hydromorphone_IndianNation = max(Hydromorphone_IndianNation),
                                                                     Methadone_Medicaid = max(Methadone_Medicaid),
                                                                     Methadone_CommercialIns = max(Methadone_CommercialIns),
                                                                     Methadone_Medicare = max(Methadone_Medicare),
                                                                     Methadone_CashCredit = max(Methadone_CashCredit),
                                                                     Methadone_MilitaryIns = max(Methadone_MilitaryIns),
                                                                     Methadone_WorkersComp = max(Methadone_WorkersComp),
                                                                     Methadone_Other = max(Methadone_Other),
                                                                     Methadone_IndianNation = max(Methadone_IndianNation),
                                                                     Fentanyl_Medicaid = max(Fentanyl_Medicaid),
                                                                     Fentanyl_CommercialIns = max(Fentanyl_CommercialIns),
                                                                     Fentanyl_Medicare = max(Fentanyl_Medicare),
                                                                     Fentanyl_CashCredit = max(Fentanyl_CashCredit),
                                                                     Fentanyl_MilitaryIns = max(Fentanyl_MilitaryIns),
                                                                     Fentanyl_WorkersComp = max(Fentanyl_WorkersComp),
                                                                     Fentanyl_Other = max(Fentanyl_Other),
                                                                     Fentanyl_IndianNation = max(Fentanyl_IndianNation),
                                                                     Oxymorphone_Medicaid = max(Oxymorphone_Medicaid),
                                                                     Oxymorphone_CommercialIns = max(Oxymorphone_CommercialIns),
                                                                     Oxymorphone_Medicare = max(Oxymorphone_Medicare),
                                                                     Oxymorphone_CashCredit = max(Oxymorphone_CashCredit),
                                                                     Oxymorphone_MilitaryIns = max(Oxymorphone_MilitaryIns),
                                                                     Oxymorphone_WorkersComp = max(Oxymorphone_WorkersComp),
                                                                     Oxymorphone_Other = max(Oxymorphone_Other),
                                                                     Oxymorphone_IndianNation = max(Oxymorphone_IndianNation),
                                                                     long_term_ever = ifelse(sum(long_term) > 0, 1, 0))
                                                                   
write.csv(PATIENT_SECOND, "../Data/PATIENT_2018_SECOND_LONGTERM_INPUT.csv", row.names = FALSE)

################################################################################
################################################################################

FULL_FIRST <- FULL_2018 %>% filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(first_presc_date, format = "%m/%d/%Y"))

PATIENT_FIRST <- FULL_FIRST %>% group_by(patient_id) %>% summarize(age = age[1],
                                                                   patient_gender = patient_gender[1],
                                                                   num_presc = max(num_presc),
                                                                   quantity = mean(quantity),
                                                                   days_supply = mean(days_supply),
                                                                   quantity_per_day = mean(quantity_per_day),
                                                                   daily_dose = mean(daily_dose),
                                                                   total_dose = mean(total_dose),
                                                                   concurrent_MME = mean(concurrent_MME),
                                                                   concurrent_methadone_MME = mean(concurrent_methadone_MME),
                                                                   num_prescribers = max(num_prescribers),
                                                                   num_pharmacies = max(num_pharmacies),
                                                                   concurrent_benzo = max(concurrent_benzo),
                                                                   concurrent_benzo_same = max(concurrent_benzo_same),
                                                                   concurrent_benzo_diff = max(concurrent_benzo_diff),
                                                                   consecutive_days = max(consecutive_days),
                                                                   Codeine = sum(Codeine),
                                                                   Hydrocodone = sum(Hydrocodone),
                                                                   Oxycodone = sum(Oxycodone),
                                                                   Morphine = sum(Morphine),
                                                                   Hydromorphone = sum(Hydromorphone),
                                                                   Methadone = sum(Methadone),
                                                                   Fentanyl = sum(Fentanyl),
                                                                   Oxymorphone = sum(Oxymorphone),
                                                                   Medicaid = sum(Medicaid),
                                                                   CommercialIns = sum(CommercialIns),
                                                                   Medicare = sum(Medicare),
                                                                   CashCredit = sum(CashCredit),
                                                                   MilitaryIns = sum(MilitaryIns),
                                                                   WorkersComp = sum(WorkersComp),
                                                                   Other = sum(Other),
                                                                   IndianNation = sum(IndianNation),
                                                                   alert1 = sum(alert1),
                                                                   alert2 = sum(alert2),
                                                                   alert3 = sum(alert3),
                                                                   alert4 = sum(alert4),
                                                                   alert5 = sum(alert5),
                                                                   alert6 = sum(alert6),
                                                                   num_alert = sum(num_alert),
                                                                   opioid_days = max(opioid_days),
                                                                   num_presc = max(num_presc),
                                                                   dose_diff = max(dose_diff),
                                                                   MME_diff = max(MME_diff),
                                                                   days_diff = max(days_diff),
                                                                   switch_drug = max(switch_drug),
                                                                   switch_payment = max(switch_payment),
                                                                   Codeine_MME = max(Codeine_MME),
                                                                   Hydrocodone_MME = max(Hydrocodone_MME),
                                                                   Oxycodone_MME = max(Oxycodone_MME),
                                                                   Morphine_MME = max(Morphine_MME),
                                                                   Hydromorphone_MME = max(Hydromorphone_MME),
                                                                   Methadone_MME = max(Methadone_MME),
                                                                   Fentanyl_MME = max(Fentanyl_MME),
                                                                   Oxymorphone_MME = max(Oxymorphone_MME),
                                                                   Codeine_Medicaid = max(Codeine_Medicaid),
                                                                   Codeine_CommercialIns = max(Codeine_CommercialIns),
                                                                   Codeine_Medicare = max(Codeine_Medicare),
                                                                   Codeine_CashCredit = max(Codeine_CashCredit),
                                                                   Codeine_MilitaryIns = max(Codeine_MilitaryIns),
                                                                   Codeine_WorkersComp = max(Codeine_WorkersComp),
                                                                   Codeine_Other = max(Codeine_Other),
                                                                   Codeine_IndianNation = max(Codeine_IndianNation),
                                                                   Hydrocodone_Medicaid = max(Hydrocodone_Medicaid),
                                                                   Hydrocodone_CommercialIns = max(Hydrocodone_CommercialIns),
                                                                   Hydrocodone_Medicare = max(Hydrocodone_Medicare),
                                                                   Hydrocodone_CashCredit = max(Hydrocodone_CashCredit),
                                                                   Hydrocodone_MilitaryIns = max(Hydrocodone_MilitaryIns),
                                                                   Hydrocodone_WorkersComp = max(Hydrocodone_WorkersComp),
                                                                   Hydrocodone_Other = max(Hydrocodone_Other),
                                                                   Hydrocodone_IndianNation = max(Hydrocodone_IndianNation),
                                                                   Oxycodone_Medicaid = max(Oxycodone_Medicaid),
                                                                   Oxycodone_CommercialIns = max(Oxycodone_CommercialIns),
                                                                   Oxycodone_Medicare = max(Oxycodone_Medicare),
                                                                   Oxycodone_CashCredit = max(Oxycodone_CashCredit),
                                                                   Oxycodone_MilitaryIns = max(Oxycodone_MilitaryIns),
                                                                   Oxycodone_WorkersComp = max(Oxycodone_WorkersComp),
                                                                   Oxycodone_Other = max(Oxycodone_Other),
                                                                   Oxycodone_IndianNation = max(Oxycodone_IndianNation),
                                                                   Morphine_Medicaid = max(Morphine_Medicaid),
                                                                   Morphine_CommercialIns = max(Morphine_CommercialIns),
                                                                   Morphine_Medicare = max(Morphine_Medicare),
                                                                   Morphine_CashCredit = max(Morphine_CashCredit),
                                                                   Morphine_MilitaryIns = max(Morphine_MilitaryIns),
                                                                   Morphine_WorkersComp = max(Morphine_WorkersComp),
                                                                   Morphine_Other = max(Morphine_Other),
                                                                   Morphine_IndianNation = max(Morphine_IndianNation),
                                                                   Hydromorphone_Medicaid = max(Hydromorphone_Medicaid),
                                                                   Hydromorphone_CommercialIns = max(Hydromorphone_CommercialIns),
                                                                   Hydromorphone_Medicare = max(Hydromorphone_Medicare),
                                                                   Hydromorphone_CashCredit = max(Hydromorphone_CashCredit),
                                                                   Hydromorphone_MilitaryIns = max(Hydromorphone_MilitaryIns),
                                                                   Hydromorphone_WorkersComp = max(Hydromorphone_WorkersComp),
                                                                   Hydromorphone_Other = max(Hydromorphone_Other),
                                                                   Hydromorphone_IndianNation = max(Hydromorphone_IndianNation),
                                                                   Methadone_Medicaid = max(Methadone_Medicaid),
                                                                   Methadone_CommercialIns = max(Methadone_CommercialIns),
                                                                   Methadone_Medicare = max(Methadone_Medicare),
                                                                   Methadone_CashCredit = max(Methadone_CashCredit),
                                                                   Methadone_MilitaryIns = max(Methadone_MilitaryIns),
                                                                   Methadone_WorkersComp = max(Methadone_WorkersComp),
                                                                   Methadone_Other = max(Methadone_Other),
                                                                   Methadone_IndianNation = max(Methadone_IndianNation),
                                                                   Fentanyl_Medicaid = max(Fentanyl_Medicaid),
                                                                   Fentanyl_CommercialIns = max(Fentanyl_CommercialIns),
                                                                   Fentanyl_Medicare = max(Fentanyl_Medicare),
                                                                   Fentanyl_CashCredit = max(Fentanyl_CashCredit),
                                                                   Fentanyl_MilitaryIns = max(Fentanyl_MilitaryIns),
                                                                   Fentanyl_WorkersComp = max(Fentanyl_WorkersComp),
                                                                   Fentanyl_Other = max(Fentanyl_Other),
                                                                   Fentanyl_IndianNation = max(Fentanyl_IndianNation),
                                                                   Oxymorphone_Medicaid = max(Oxymorphone_Medicaid),
                                                                   Oxymorphone_CommercialIns = max(Oxymorphone_CommercialIns),
                                                                   Oxymorphone_Medicare = max(Oxymorphone_Medicare),
                                                                   Oxymorphone_CashCredit = max(Oxymorphone_CashCredit),
                                                                   Oxymorphone_MilitaryIns = max(Oxymorphone_MilitaryIns),
                                                                   Oxymorphone_WorkersComp = max(Oxymorphone_WorkersComp),
                                                                   Oxymorphone_Other = max(Oxymorphone_Other),
                                                                   Oxymorphone_IndianNation = max(Oxymorphone_IndianNation),
                                                                   long_term_ever = ifelse(sum(long_term) > 0, 1, 0))
                                                                     
write.csv(PATIENT_FIRST, "../Data/PATIENT_2018_FIRST_LONGTERM_INPUT.csv", row.names = FALSE)

################################################################################
################################################################################
################################################################################

NEW <- NEW_TEMP %>% mutate(days_to_second = as.numeric(as.Date(second_presc_date, format = "%m/%d/%Y") - 
                                                         as.Date(first_presc_date, format = "%m/%d/%Y")),
                           days_to_third = as.numeric(as.Date(third_presc_date, format = "%m/%d/%Y") - 
                                                         as.Date(first_presc_date, format = "%m/%d/%Y")))


PATIENT <- FULL_2018 %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                            age = age[1],
                                                            first_presc_date = date_filled[1],
                                                            longterm = ifelse(sum(long_term) > 0, 1, 0), 
                                                            longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                            longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA))

PATIENT_LONGTERM <- PATIENT %>% filter(longterm == 1) 
PATIENT_LONGTERM <- PATIENT_LONGTERM %>% mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - 
                                                                                 as.Date(first_presc_date, format = "%m/%d/%Y")))

TEST <- NEW %>% filter(!is.na(days_to_second))
mean(TEST$days_to_second)

ggplot(TEST, aes(x=ifelse(days_to_second>365, 365, days_to_second))) + 
  geom_density(adjust = 1) +
  geom_vline(xintercept = mean(TEST$days_to_second), linetype = 'dashed', color = 'red') + 
  annotate("text", x = mean(TEST$days_to_second) + 50, y = 0.02, label = "Mean = 54.4") +
  ggtitle("Days until second prescription date") +
  xlab("Days") +
  ylab("Density") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/days_second.pdf", bg="white", width=8, height=6, dpi=300)


TEST <- NEW %>% filter(!is.na(days_to_third))
mean(TEST$days_to_third)

ggplot(TEST, aes(x=ifelse(days_to_third>365, 365, days_to_third))) + 
  geom_density(adjust = 1) +
  geom_vline(xintercept = mean(TEST$days_to_third), linetype = 'dashed', color = 'red') + 
  annotate("text", x = mean(TEST$days_to_third) + 50, y = 0.01, label = "Mean = 82.9") +
  ggtitle("Days until third prescription date") +
  xlab("Days") +
  ylab("Density") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/days_third.pdf", bg="white", width=8, height=6, dpi=300)


################################################################################
################################################################################
################################################################################

TEST <- read.csv("../Data/FULL_2019_LONGTERM.csv")
TEMP <- TEST[1:200,]

TEST <- read.csv("../Data/FULL_2018_STUMPS0.csv")
TEMP <- TEST[1:200,]
TEST <- read.csv("../Data/FULL_2018_LONGTERM_INPUT.csv")
TEMP <- TEST[1:200,]

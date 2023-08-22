### Identify if a patient will receive an alert within the next T days
### Feature engineering with demographics etc.

library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2018
T_period31 = 31
T_period62 = 62
T_period93 = 93

################################################################################

FULL_ALL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))

# FULL <- FULL_ALL %>% filter(patient_id < 41846152)
# FULL <- FULL_ALL %>% filter(patient_id >= 41846152 & patient_id < 54195156)
# FULL <- FULL_ALL %>% filter(patient_id >= 54195156 & patient_id < 68042918)
# FULL <- FULL_ALL %>% filter(patient_id >= 68042918)
rm(FULL_ALL)

################################################################################
###################### New consecutive days & alert ############################
################################################################################

compute_overlap <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  if(presc_index == 1){
    return (0)
  } else{
    # prescriptions before the current (exclude co-prescriptions)
    prev_presc_until <- tail(PATIENT[as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                       as.Date(presc_date, format = "%m/%d/%Y"), c("presc_until")], n=1)
    if(length(prev_presc_until) == 0){
      return (0)
    } else if(as.Date(prev_presc_until, format = "%m/%d/%Y") + 1 >= 
              as.Date(presc_date, format = "%m/%d/%Y")){
      # 6/14-7/14, 7/15-8/15 we still consider them overlap
      return (as.Date(prev_presc_until, format = "%m/%d/%Y") + 2 - as.Date(presc_date, format = "%m/%d/%Y"))
    } else{
      return (0)
    }
  }
}
FULL$overlap <- mcmapply(compute_overlap, FULL$patient_id, FULL$prescription_id, mc.cores=40)

compute_consecutive_days <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  days_current <- PATIENT_PRESC_OPIOIDS$days_supply
  
  # Modified: if no overlap, consecutive days should be the current days_supply
  if(PATIENT_PRESC_OPIOIDS$overlap == 0){
    return (days_current)
  } else{
    PATIENT_PREV_PRESC_OPIOIDS <- PATIENT[as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                            as.Date(presc_date, format = "%m/%d/%Y"),]
    # must exist as at least one index has overlap 0, i.e. the first prescription
    last_index <- tail(which(PATIENT_PREV_PRESC_OPIOIDS$overlap == 0), n=1)
    consecutive_day = as.Date(PATIENT_PRESC_OPIOIDS$presc_until, format = "%m/%d/%Y") - 
      as.Date(PATIENT[last_index, c("date_filled")], format = "%m/%d/%Y")
    return (consecutive_day)
  }
}
FULL$consecutive_days <- mcmapply(compute_consecutive_days, FULL$patient_id, FULL$prescription_id, mc.cores=40)

################################################################################
################################################################################
################################################################################

patient_alert <- function(pat_id, presc_id){
  
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  
  alert1 = 0
  alert2 = 0
  alert3 = 0
  alert4 = 0
  alert5 = 0
  alert6 = 0
  
  if(PATIENT_PRESC_OPIOIDS$concurrent_MME >= 90){alert1 = 1} 
  if(PATIENT_PRESC_OPIOIDS$concurrent_MME_methadone >= 40){alert2 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_prescribers >= 6){alert3 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_pharmacies >= 6){alert4 = 1}
  if(PATIENT_PRESC_OPIOIDS$consecutive_days >= 90){alert5 = 1}
  if(PATIENT_PRESC_OPIOIDS$concurrent_benzo > 0){alert6 = 1}
  
  return (c(alert1, alert2, alert3, alert4, alert5, alert6))
}

alert <- mcmapply(patient_alert, FULL$patient_id, FULL$prescription_id, mc.cores=40)

FULL$alert1 = alert[1, ]
FULL$alert2 = alert[2, ]
FULL$alert3 = alert[3, ]
FULL$alert4 = alert[4, ]
FULL$alert5 = alert[5, ]
FULL$alert6 = alert[6, ]

FULL$num_alert <- FULL$alert1 + FULL$alert2 + FULL$alert3 + FULL$alert4 + FULL$alert5 + FULL$alert6

################################################################################
############################# Alert within T days ##############################
################################################################################

### For each patient, look T days further to see if there's any alert ###
alert_T_days <- function(pat_id, presc_id, T_period){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  PATIENT_T_OPIOIDS <- PATIENT[which(as.Date(PATIENT$date_filled,format = "%m/%d/%Y") >= 
                                             as.Date(presc_date, format = "%m/%d/%Y") & 
                                             as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                             as.Date(presc_date, format = "%m/%d/%Y") + T_period &
                                       PATIENT$prescription_id != presc_id),]
  
  PATIENT_EXACT_T_OPIOID <- PATIENT[which(as.Date(PATIENT$date_filled,format = "%m/%d/%Y") >= 
                                            as.Date(presc_date, format = "%m/%d/%Y") + (T_period - 31) & 
                                            as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                            as.Date(presc_date, format = "%m/%d/%Y") + T_period &
                                            PATIENT$prescription_id != presc_id),]
  
  alert1_next_T = 0
  alert2_next_T = 0
  alert3_next_T = 0
  alert4_next_T = 0
  alert5_next_T = 0
  alert6_next_T = 0
  
  if(sum(PATIENT_T_OPIOIDS$alert1) > 0){alert1_next_T = 1} 
  if(sum(PATIENT_T_OPIOIDS$alert2) > 0){alert2_next_T = 1} 
  if(sum(PATIENT_T_OPIOIDS$alert3) > 0){alert3_next_T = 1} 
  if(sum(PATIENT_T_OPIOIDS$alert4) > 0){alert4_next_T = 1} 
  if(sum(PATIENT_T_OPIOIDS$alert5) > 0){alert5_next_T = 1} 
  if(sum(PATIENT_T_OPIOIDS$alert6) > 0){alert6_next_T = 1} 
  
  
  alert1_exact_T = 0
  alert2_exact_T = 0
  alert3_exact_T = 0
  alert4_exact_T = 0
  alert5_exact_T = 0
  alert6_exact_T = 0
  
  if(sum(PATIENT_EXACT_T_OPIOID$alert1) > 0){alert1_exact_T = 1} 
  if(sum(PATIENT_EXACT_T_OPIOID$alert2) > 0){alert2_exact_T = 1} 
  if(sum(PATIENT_EXACT_T_OPIOID$alert3) > 0){alert3_exact_T = 1} 
  if(sum(PATIENT_EXACT_T_OPIOID$alert4) > 0){alert4_exact_T = 1} 
  if(sum(PATIENT_EXACT_T_OPIOID$alert5) > 0){alert5_exact_T = 1} 
  if(sum(PATIENT_EXACT_T_OPIOID$alert6) > 0){alert6_exact_T = 1}
  
  
  return (c(alert1_next_T, alert2_next_T, alert3_next_T, alert4_next_T, alert5_next_T, alert6_next_T,
            alert1_exact_T, alert2_exact_T, alert3_exact_T, alert4_exact_T, alert5_exact_T, alert6_exact_T))
}

######## 31
alert <- mcmapply(alert_T_days, FULL$patient_id, FULL$prescription_id, T_period31, mc.cores=40)

FULL$alert1_next31 = alert[1, ]
FULL$alert2_next31 = alert[2, ]
FULL$alert3_next31 = alert[3, ]
FULL$alert4_next31 = alert[4, ]
FULL$alert5_next31 = alert[5, ]
FULL$alert6_next31 = alert[6, ]

FULL$alert1_exact31 = alert[7, ]
FULL$alert2_exact31 = alert[8, ]
FULL$alert3_exact31 = alert[9, ]
FULL$alert4_exact31 = alert[10, ]
FULL$alert5_exact31 = alert[11, ]
FULL$alert6_exact31 = alert[12, ]

rm(alert)

FULL <- FULL %>% mutate(alert_next31 = ifelse(alert1_next31 + alert2_next31 + alert3_next31 + 
                                                alert4_next31 + alert5_next31 + alert6_next31 > 0, 1, 0),
                        alert_exact31 = ifelse(alert1_exact31 + alert2_exact31 + alert3_exact31 + 
                                                alert4_exact31 + alert5_exact31 + alert6_exact31 > 0, 1, 0))


######## 62
alert <- mcmapply(alert_T_days, FULL$patient_id, FULL$prescription_id, T_period62, mc.cores=40)

FULL$alert1_next62 = alert[1, ]
FULL$alert2_next62 = alert[2, ]
FULL$alert3_next62 = alert[3, ]
FULL$alert4_next62 = alert[4, ]
FULL$alert5_next62 = alert[5, ]
FULL$alert6_next62 = alert[6, ]

FULL$alert1_exact62 = alert[7, ]
FULL$alert2_exact62 = alert[8, ]
FULL$alert3_exact62 = alert[9, ]
FULL$alert4_exact62 = alert[10, ]
FULL$alert5_exact62 = alert[11, ]
FULL$alert6_exact62 = alert[12, ]

rm(alert)

FULL <- FULL %>% mutate(alert_next62 = ifelse(alert1_next62 + alert2_next62 + alert3_next62 + 
                                                alert4_next62 + alert5_next62 + alert6_next62 > 0, 1, 0),
                        alert_exact62 = ifelse(alert1_exact62 + alert2_exact62 + alert3_exact62 + 
                                                 alert4_exact62 + alert5_exact62 + alert6_exact62 > 0, 1, 0))


######## 93
alert <- mcmapply(alert_T_days, FULL$patient_id, FULL$prescription_id, T_period93, mc.cores=40)

FULL$alert1_next93 = alert[1, ]
FULL$alert2_next93 = alert[2, ]
FULL$alert3_next93 = alert[3, ]
FULL$alert4_next93 = alert[4, ]
FULL$alert5_next93 = alert[5, ]
FULL$alert6_next93 = alert[6, ]

FULL$alert1_exact93 = alert[7, ]
FULL$alert2_exact93 = alert[8, ]
FULL$alert3_exact93 = alert[9, ]
FULL$alert4_exact93 = alert[10, ]
FULL$alert5_exact93 = alert[11, ]
FULL$alert6_exact93 = alert[12, ]

rm(alert)

FULL <- FULL %>% mutate(alert_next93 = ifelse(alert1_next93 + alert2_next93 + alert3_next93 + 
                                                alert4_next93 + alert5_next93 + alert6_next93 > 0, 1, 0),
                        alert_exact93 = ifelse(alert1_exact93 + alert2_exact93 + alert3_exact93 + 
                                                 alert4_exact93 + alert5_exact93 + alert6_exact93 > 0, 1, 0))

write.csv(FULL, paste("../Data/FULL_", year, "_ALERT_NEW4.csv", sep=""), row.names = FALSE)

################################################################################
### Merge four subdataset

FULL1 <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW1.csv", sep=""))
FULL2 <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW2.csv", sep=""))
FULL3 <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW3.csv", sep=""))
FULL4 <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW4.csv", sep=""))

FULL <- rbind(FULL1, FULL2)
FULL <- rbind(FULL, FULL3)
FULL <- rbind(FULL, FULL4)

write.csv(FULL, paste("../Data/FULL_", year, "_ALERT.csv", sep=""), row.names = FALSE)

################################################################################
################################################################################
################################################################################

### Patient that does not have alert on their first prescription
### Look at first prescription of each patient, and check if it is associated with an alert
FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))

### Before: 3,120,742 alerted prescriptions from 866,428 patients
### After: 997,274 alerted prescriptions from 251,295 patients

patient_table <- FULL %>% group_by(patient_id) %>% filter(row_number()==1) %>%
  mutate(alert_first_presc = num_alert) %>% select(c(patient_id, alert_first_presc))
FULL <- left_join(FULL, patient_table, by = 'patient_id') %>% filter(alert_first_presc == 0) %>% select(-c(alert_first_presc))

FULL_FIRST <- FULL %>% group_by(patient_id) %>% filter(row_number()==1) # 3,790,626 prescriptions from 3,790,626 patients
write.csv(FULL_FIRST, paste("../Data/FULL_", year, "_FIRST.csv", sep=""), row.names = FALSE)

################################################################################
################################################################################
################################################################################

### Patient that does not have alert on their first prescription
### Patient's prescription up to their first alert
FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))

### Patient with no alert on their first prescription
## Before: 9,601,432 prescriptions from 4,409,953
## After: 7,393,013 prescriptions from 3,790,626
patient_table <- FULL %>% group_by(patient_id) %>% summarize(alert_first_presc = ifelse(num_alert[1] > 0, 1, 0))
FULL <- left_join(FULL, patient_table, by = 'patient_id') %>% filter(alert_first_presc == 0) %>% select(-c(alert_first_presc))

### Patient's prescription up to their first alert
## If the patient never gets an alert, set a dummy date in 2020
## Note: we also exclude prescriptions that are on the same date of first alert ('<' instead of "<=")
## Before: 7,393,013 prescriptions from 3,790,626
## After: 6,103,698 prescriptions from 3,790,441
patient_table <- FULL %>% group_by(patient_id) %>% summarize(first_alert_date = ifelse(sum(num_alert) > 0, date_filled[num_alert > 0][1], "01/01/2020"))
FULL <- left_join(FULL, patient_table, by = 'patient_id') %>% 
  filter(as.Date(date_filled,format = "%m/%d/%Y") < as.Date(first_alert_date,format = "%m/%d/%Y")) %>%
  select(-c(first_alert_date))

# patient_summary <- FULL %>% group_by(patient_id) %>% summarize(num_presc = n())
# plot(patient_summary$num_presc, main="Average number of prescription up to first alert (number of prescription if never alert)")

################################################################################
############################### Interaction effect #############################
################################################################################

FULL <- read.csv(paste("../Data/FULL_", year, "_FIRST.csv", sep=""))

### Interaction between drug * quantity, drug * payment
FULL <- FULL %>% mutate(Codeine_MME = ifelse(drug == "Codeine", daily_dose, 0),
                        Hydrocodone_MME = ifelse(drug == "Hydrocodone", daily_dose, 0),
                        Oxycodone_MME = ifelse(drug == "Oxycodone", daily_dose, 0),
                        Morphine_MME = ifelse(drug == "Morphine", daily_dose, 0),
                        Hydromorphone_MME = ifelse(drug == "Hydromorphone", daily_dose, 0),
                        Methadone_MME = ifelse(drug == "Methadone", daily_dose, 0),
                        Fentanyl_MME = ifelse(drug == "Fentanyl", daily_dose, 0),
                        Oxymorphone_MME = ifelse(drug == "Oxymorphone", daily_dose, 0))

FULL <- FULL %>% mutate(drug_payment = paste(drug, payment, sep = "_")) %>%
  mutate(Codeine_Medicaid = ifelse(drug_payment == "Codeine_Medicaid", 1, 0),
         Codeine_CommercialIns = ifelse(drug_payment == "Codeine_CommercialIns", 1, 0),
         Codeine_Medicare = ifelse(drug_payment == "Codeine_Medicare", 1, 0),
         Codeine_CashCredit = ifelse(drug_payment == "Codeine_CashCredit", 1, 0),
         Codeine_MilitaryIns = ifelse(drug_payment == "Codeine_MilitaryIns", 1, 0),
         Codeine_WorkersComp = ifelse(drug_payment == "Codeine_WorkersComp", 1, 0),
         Codeine_Other = ifelse(drug_payment == "Codeine_Other", 1, 0),
         Codeine_IndianNation = ifelse(drug_payment == "Codeine_IndianNation", 1, 0),
         Hydrocodone_Medicaid = ifelse(drug_payment == "Hydrocodone_Medicaid", 1, 0),
         Hydrocodone_CommercialIns = ifelse(drug_payment == "Hydrocodone_CommercialIns", 1, 0),
         Hydrocodone_Medicare = ifelse(drug_payment == "Hydrocodone_Medicare", 1, 0),
         Hydrocodone_CashCredit = ifelse(drug_payment == "Hydrocodone_CashCredit", 1, 0),
         Hydrocodone_MilitaryIns = ifelse(drug_payment == "Hydrocodone_MilitaryIns", 1, 0),
         Hydrocodone_WorkersComp = ifelse(drug_payment == "Hydrocodone_WorkersComp", 1, 0),
         Hydrocodone_Other = ifelse(drug_payment == "Hydrocodone_Other", 1, 0),
         Hydrocodone_IndianNation = ifelse(drug_payment == "Hydrocodone_IndianNation", 1, 0),
         Oxycodone_Medicaid = ifelse(drug_payment == "Oxycodone_Medicaid", 1, 0),
         Oxycodone_CommercialIns = ifelse(drug_payment == "Oxycodone_CommercialIns", 1, 0),
         Oxycodone_Medicare = ifelse(drug_payment == "Oxycodone_Medicare", 1, 0),
         Oxycodone_CashCredit = ifelse(drug_payment == "Oxycodone_CashCredit", 1, 0),
         Oxycodone_MilitaryIns = ifelse(drug_payment == "Oxycodone_MilitaryIns", 1, 0),
         Oxycodone_WorkersComp = ifelse(drug_payment == "Oxycodone_WorkersComp", 1, 0),
         Oxycodone_Other = ifelse(drug_payment == "Oxycodone_Other", 1, 0),
         Oxycodone_IndianNation = ifelse(drug_payment == "Oxycodone_IndianNation", 1, 0),
         Morphine_Medicaid = ifelse(drug_payment == "Morphine_Medicaid", 1, 0),
         Morphine_CommercialIns = ifelse(drug_payment == "Morphine_CommercialIns", 1, 0),
         Morphine_Medicare = ifelse(drug_payment == "Morphine_Medicare", 1, 0),
         Morphine_CashCredit = ifelse(drug_payment == "Morphine_CashCredit", 1, 0),
         Morphine_MilitaryIns = ifelse(drug_payment == "Morphine_MilitaryIns", 1, 0),
         Morphine_WorkersComp = ifelse(drug_payment == "Morphine_WorkersComp", 1, 0),
         Morphine_Other = ifelse(drug_payment == "Morphine_Other", 1, 0),
         Morphine_IndianNation = ifelse(drug_payment == "Morphine_IndianNation", 1, 0),
         Hydromorphone_Medicaid = ifelse(drug_payment == "Hydromorphone_Medicaid", 1, 0),
         Hydromorphone_CommercialIns = ifelse(drug_payment == "Hydromorphone_CommercialIns", 1, 0),
         Hydromorphone_Medicare = ifelse(drug_payment == "Hydromorphone_Medicare", 1, 0),
         Hydromorphone_CashCredit = ifelse(drug_payment == "Hydromorphone_CashCredit", 1, 0),
         Hydromorphone_MilitaryIns = ifelse(drug_payment == "Hydromorphone_MilitaryIns", 1, 0),
         Hydromorphone_WorkersComp = ifelse(drug_payment == "Hydromorphone_WorkersComp", 1, 0),
         Hydromorphone_Other = ifelse(drug_payment == "Hydromorphone_Other", 1, 0),
         Hydromorphone_IndianNation = ifelse(drug_payment == "Hydromorphone_IndianNation", 1, 0),
         Methadone_Medicaid = ifelse(drug_payment == "Methadone_Medicaid", 1, 0),
         Methadone_CommercialIns = ifelse(drug_payment == "Methadone_CommercialIns", 1, 0),
         Methadone_Medicare = ifelse(drug_payment == "Methadone_Medicare", 1, 0),
         Methadone_CashCredit = ifelse(drug_payment == "Methadone_CashCredit", 1, 0),
         Methadone_MilitaryIns = ifelse(drug_payment == "Methadone_MilitaryIns", 1, 0),
         Methadone_WorkersComp = ifelse(drug_payment == "Methadone_WorkersComp", 1, 0),
         Methadone_Other = ifelse(drug_payment == "Methadone_Other", 1, 0),
         Methadone_IndianNation = ifelse(drug_payment == "Methadone_IndianNation", 1, 0),
         Fentanyl_Medicaid = ifelse(drug_payment == "Fentanyl_Medicaid", 1, 0),
         Fentanyl_CommercialIns = ifelse(drug_payment == "Fentanyl_CommercialIns", 1, 0),
         Fentanyl_Medicare = ifelse(drug_payment == "Fentanyl_Medicare", 1, 0),
         Fentanyl_CashCredit = ifelse(drug_payment == "Fentanyl_CashCredit", 1, 0),
         Fentanyl_MilitaryIns = ifelse(drug_payment == "Fentanyl_MilitaryIns", 1, 0),
         Fentanyl_WorkersComp = ifelse(drug_payment == "Fentanyl_WorkersComp", 1, 0),
         Fentanyl_Other = ifelse(drug_payment == "Fentanyl_Other", 1, 0),
         Fentanyl_IndianNation = ifelse(drug_payment == "Fentanyl_IndianNation", 1, 0),
         Oxymorphone_Medicaid = ifelse(drug_payment == "Oxymorphone_Medicaid", 1, 0),
         Oxymorphone_CommercialIns = ifelse(drug_payment == "Oxymorphone_CommercialIns", 1, 0),
         Oxymorphone_Medicare = ifelse(drug_payment == "Oxymorphone_Medicare", 1, 0),
         Oxymorphone_CashCredit = ifelse(drug_payment == "Oxymorphone_CashCredit", 1, 0),
         Oxymorphone_MilitaryIns = ifelse(drug_payment == "Oxymorphone_MilitaryIns", 1, 0),
         Oxymorphone_WorkersComp = ifelse(drug_payment == "Oxymorphone_WorkersComp", 1, 0),
         Oxymorphone_Other = ifelse(drug_payment == "Oxymorphone_Other", 1, 0),
         Oxymorphone_IndianNation = ifelse(drug_payment == "Oxymorphone_IndianNation", 1, 0))

# write.csv(FULL, paste("../Data/FULL_", year, "_FIRST_ALLFEATURE.csv", sep=""), row.names = FALSE)
write.csv(FULL, paste("../Data/FULL_", year, "_ROLLING_ALLFEATURE.csv", sep=""), row.names = FALSE)

################################################################################
################################################################################
################################################################################

FULL <- read.csv(paste("../Data/FULL_", year, "_FIRST_ALLFEATURE.csv", sep=""))
length(unique(FULL$patient_id))
ALERT31 <- FULL %>% filter(alert_exact31 == 1)
ALERT62 <- FULL %>% filter(alert_exact62 == 1)
ALERT93 <- FULL %>% filter(alert_exact93 == 1)

################################################################################
################################################################################
################################################################################

FULL <- read.csv(paste("../Data/FULL_", year, "_FIRST_ALLFEATURE.csv", sep=""))
FULL <- FULL %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                           conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                           DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_quarter, prescription_year, 
                           presence_MME, presence_MME_methadone, presence_num_prescribers, presence_num_pharmacies, 
                           overlap, presence_days, num_alert, long_term))
write.csv(FULL, paste("../Data/FULL_", year, "_INPUT_FIRST_ALLFEATURE.csv", sep=""), row.names = FALSE)

# test
TEST <- FULL %>% select(c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                           conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                           DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_quarter, prescription_year, 
                           presence_MME, presence_MME_methadone, presence_num_prescribers, presence_num_pharmacies, 
                           overlap, presence_days, num_alert, long_term, num_prescribers, num_pharmacies))

TEST <- TEST %>% filter(num_prescribers > 1 | num_pharmacies > 1)

FULL_ALL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))
TEST <- FULL_ALL %>% filter(patient_id == 15999989)
TEST <- TEST %>% select(c(prescription_id, patient_id, prescriber_id, pharmacy_id, days_supply, date_filled, presc_until, drug, daily_dose, num_prescribers, num_pharmacies, concurrent_MME))

################################################################################
################################################################################
################################################################################

FULL <- read.csv(paste("../Data/FULL_", year, "_ROLLING_ALLFEATURE.csv", sep=""))
FULL <- FULL %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                           conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                           DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_quarter, prescription_year, 
                           presence_MME, presence_MME_methadone, presence_num_prescribers, presence_num_pharmacies, 
                           overlap, presence_days, num_alert, long_term))
write.csv(FULL, paste("../Data/FULL_", year, "_INPUT_ROLLING_ALLFEATURE.csv", sep=""), row.names = FALSE)

################################################################################
################################################################################
################################################################################

# test <- FULL %>% filter(patient_id == 48)
# FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))
# FULL_31 <- FULL %>% filter(alert_next31 == 1) # 2055604
# FULL_62 <- FULL %>% filter(alert_next62 == 1) # 2312670
# FULL_93 <- FULL %>% filter(alert_next93 == 1) # 2370993
 
# FULL <- read.csv(paste("../Data/FULL_2018_INPUT.csv", sep=""))
# FULL_31 <- FULL %>% filter(alert_next31 == 1) # 2286405
# FULL_62 <- FULL %>% filter(alert_next62 == 1) # 2719718
# FULL_93 <- FULL %>% filter(alert_next93 == 1) # 2918841 (they are indeed nested)

# FULL_31exact <- FULL %>% filter(alert_exact31 == 1) # 2286405
# FULL_62exact <- FULL %>% filter(alert_exact62 == 1) # 1730583
# FULL_93exact <- FULL %>% filter(alert_exact93 == 1) # 1441210

# test <- FULL[1:200,] %>% select(c(patient_id, date_filled, any_alert, alert_next31, alert_next62, alert_next93))

# FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW1.csv", sep=""))
# test <- FULL %>% filter(patient_id == 48) %>% select(c(patient_id, date_filled, any_alert, alert_next31, alert_next62, alert_next93))
# test <- FULL %>% filter(patient_id == 439) %>% select(c(patient_id, date_filled, any_alert, alert_next31, alert_next62, alert_next93))

###############################################################################

## drug type
unique(FULL_ALL$drug)
summary(FULL_ALL$drug)
a = nrow(FULL_ALL[FULL_ALL$drug == 'Hydrocodone',]) / 9601432
b = nrow(FULL_ALL[FULL_ALL$drug == 'Oxycodone',]) / 9601432
c = nrow(FULL_ALL[FULL_ALL$drug == 'Hydromorphone',]) / 9601432
d = nrow(FULL_ALL[FULL_ALL$drug == 'Morphine',]) / 9601432
e = nrow(FULL_ALL[FULL_ALL$drug == 'Codeine',]) / 9601432
f = nrow(FULL_ALL[FULL_ALL$drug == 'Fentanyl',]) / 9601432
g = nrow(FULL_ALL[FULL_ALL$drug == 'Methadone',]) / 9601432
h = nrow(FULL_ALL[FULL_ALL$drug == 'Oxymorphone',]) / 9601432

## alert type
TEST <- FULL_ALL %>% filter(age > 80)
nrow(TEST)
nrow(TEST)/ 9601432


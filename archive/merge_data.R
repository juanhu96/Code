### Combine the feature engineered dataset together
### i.e., FULL_2018_ALERT_1, 2, 3, 4, FULL_2018_ALERT_SINGLE
### Perform some other engineering (encode categorical variables)

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
setwd("/mnt/phd/jihu/opioid/Code")
year = 2017
FULL_SINGLE <- read.csv(paste("../Data/FULL_ALERT_", year, "_SINGLE.csv", sep=""))
FULL_MULTIPLE <- read.csv(paste("../Data/FULL_ALERT_", year, "_ATLEASTTWO.csv", sep=""))

########################################################################
FULL <- rbind(FULL_SINGLE, FULL_MULTIPLE)
rm(FULL_SINGLE)
rm(FULL_MULTIPLE)

### Reindex
FULL$prescription_id = seq.int(nrow(FULL))

########################################################################
############################# LONG TERM  ###############################
########################################################################

### If a patient is long term when the current prescription ends
### At least 90 days of supply in the past 180 days
compute_long_term <- function(pat_id, presc_id){
  
  PATIENT <- FULL_2018_ALERT[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_until <- PATIENT_PRESC_OPIOIDS$presc_until
  period_start <- format(as.Date(presc_until, format = "%m/%d/%Y") - 180, "%m/%d/%Y")
  
  PATIENT_WITHIN <- PATIENT[PATIENT$presc_until > period_start & PATIENT$date_filled < presc_until, ]
  
  if(sum(PATIENT_WITHIN$days_supply) >= 90){
    return (1)
    }
  else{
    return (0)
    }
  }

FULL$long_term <- mcmapply(compute_long_term, FULL$patient_id,
                                      FULL$prescription_id, mc.cores=20)

# Summary
FULL_2018_W_ALERT <- FULL %>% filter(num_alert > 0)
FULL_2018_W_LONG <- FULL %>% filter(long_term > 0)
length(unique(FULL_2018_W_ALERT$patient_id))
length(unique(FULL_2018_W_LONG$patient_id))

########################################################################
########################################################################
########################################################################

FULL <- FULL %>% mutate(patient_gender = ifelse(patient_gender == "M", 1, 0),
                                              Codeine = ifelse(drug == "Codeine", 1, 0),
                                              Hydrocodone = ifelse(drug == "Hydrocodone", 1, 0),
                                              Oxycodone = ifelse(drug == "Oxycodone", 1, 0),
                                              Morphine = ifelse(drug == "Morphine", 1, 0),
                                              Hydromorphone = ifelse(drug == "Hydromorphone", 1, 0),
                                              Methadone = ifelse(drug == "Methadone", 1, 0),
                                              Fentanyl = ifelse(drug == "Fentanyl", 1, 0),
                                              Oxymorphone = ifelse(drug == "Oxymorphone", 1, 0),
                                              Medicaid = ifelse(payment == "Medicaid", 1, 0),
                                              CommercialIns = ifelse(payment == "CommercialIns", 1, 0),
                                              Medicare = ifelse(payment == "Medicare", 1, 0),
                                              CashCredit = ifelse(payment == "CashCredit", 1, 0),
                                              MilitaryIns = ifelse(payment == "MilitaryIns", 1, 0),
                                              WorkersComp = ifelse(payment == "WorkersComp", 1, 0),
                                              Other = ifelse(payment == "Other", 1, 0),
                                              IndianNation = ifelse(payment == "IndianNation", 1, 0)) %>% 
  mutate(presence_MME = concurrent_MME - daily_dose,
         presence_MME_methadone = ifelse(drug == 'Methadone', concurrent_MME_methadone - daily_dose, concurrent_MME_methadone),
         presence_days = ifelse(consecutive_days == 0, 0, consecutive_days - days_supply))

########################################################################
### Compute number of prescriber/pharmacies in last 6 months
## Excluding the current prescriber/pharmacies
compute_num_prescriber_pharmacy_presence <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  # in last 6 months
  PATIENT_CURRENT_OPIOIDS <- PATIENT[as.Date(PATIENT$date_filled,format = "%m/%d/%Y") >= 
                                       as.Date(presc_date, format = "%m/%d/%Y") - 180 & 
                                       as.Date(PATIENT$date_filled,format = "%m/%d/%Y") < 
                                       as.Date(presc_date, format = "%m/%d/%Y"),]
  
  num_prescribers = length(unique(PATIENT_CURRENT_OPIOIDS$prescriber_id))
  num_pharmacies = length(unique(PATIENT_CURRENT_OPIOIDS$pharmacy_id))
  
  return (c(num_prescribers, num_pharmacies))
}

num_prescriber_pharmacy_presence <- mcmapply(compute_num_prescriber_pharmacy_presence, 
                                             FULL$patient_id, FULL$prescription_id, mc.cores=20)
FULL$presence_num_prescribers = num_prescriber_pharmacy_presence[1, ]
FULL$presence_num_pharmacies = num_prescriber_pharmacy_presence[2, ]

########################################################################

FULL_2018_REORDER <- FULL %>% select(c(prescription_id, patient_id, patient_birth_year, age, patient_gender, patient_zip,
                                                  prescriber_id, prescriber_zip, pharmacy_id, pharmacy_zip, strength, quantity, days_supply,
                                                  date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
                                                  chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, MAINDOSE,
                                                  payment, prescription_month, prescription_quarter, prescription_year,
                                                  concurrent_MME, concurrent_MME_methadone, presence_MME, presence_MME_methadone, presence_days,
                                                  presence_num_prescribers, presence_num_pharmacies, num_prescribers, num_pharmacies,
                                                  concurrent_benzo, concurrent_benzo_same, concurrent_benzo_diff, overlap, consecutive_days,
                                                  Codeine, Hydrocodone, Oxycodone, Morphine, Hydromorphone, Methadone, Fentanyl, Oxymorphone,
                                                  Medicaid, CommercialIns, Medicare, CashCredit, MilitaryIns, WorkersComp, Other, IndianNation,
                                                  alert1, alert2, alert3, alert4, alert5, alert6, num_alert, long_term))
write.csv(FULL_2018_REORDER, "../Data/FULL.csv", row.names = FALSE)

FULL_ALERT <- read.csv("../Data/FULL.csv")
FULL_ALERT<- FULL_ALERT %>% mutate(any_alert = ifelse(num_alert > 0, 1, 0)) %>% 
  select(c(prescription_id, patient_id, patient_birth_year, age, patient_gender, patient_zip,
           prescriber_id, prescriber_zip, pharmacy_id, pharmacy_zip, strength, quantity, days_supply,
           date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
           chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, MAINDOSE,
           payment, prescription_month, prescription_quarter, prescription_year,
           concurrent_MME, concurrent_MME_methadone, presence_MME, presence_MME_methadone, presence_days,
           presence_num_prescribers, presence_num_pharmacies, num_prescribers, num_pharmacies,
           concurrent_benzo, concurrent_benzo_same, concurrent_benzo_diff, overlap, consecutive_days,
           Codeine, Hydrocodone, Oxycodone, Morphine, Hydromorphone, Methadone, Fentanyl, Oxymorphone,
           Medicaid, CommercialIns, Medicare, CashCredit, MilitaryIns, WorkersComp, Other, IndianNation,
           alert1, alert2, alert3, alert4, alert5, alert6, num_alert, any_alert, long_term))
write.csv(FULL_ALERT, "../Data/FULL.csv", row.names = FALSE)


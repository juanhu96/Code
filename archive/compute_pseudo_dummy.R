### Adjusted version of the dataset
### INPUT: SAMPLE_ALERT or 2018_ALERT etc. (output file from identify_alert)
### OUTPUT: flexible time window, 0 dummy observation, switch in drug/payment

library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2018
FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))

FULL$patient_id_numeric <- as.numeric(FULL$patient_id)
# FULL <- FULL %>% filter(patient_id_numeric < 42737461)
# FULL <- FULL %>% filter(patient_id_numeric >= 42737461 & patient_id_numeric < 59258072)
# FULL <- FULL %>% filter(patient_id_numeric >= 59258072 & patient_id_numeric < 68418211)
# FULL <- FULL %>% filter(patient_id_numeric >= 68418211)

########################################################################
############################# TIME WINDOW ##############################
########################################################################
TIME_WINDOW = 7
FULL$date_prescribed <- as.Date(FULL$date_filled, format = "%m/%d/%Y") - TIME_WINDOW

### NOTE: we have concurrent MME, consecutive days already computed
### Concurrent MME based on the new time window
compute_pseudo_MME <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date_filled <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_date_prescribed <- PATIENT_PRESC_OPIOIDS$date_prescribed
  
  PATIENT_CURRENT_OPIOIDS <- PATIENT[as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                       as.Date(presc_date_filled, format = "%m/%d/%Y") & 
                                       as.Date(PATIENT$presc_until,format = "%m/%d/%Y") >
                                       as.Date(presc_date_prescribed, format = "%m/%d/%Y"),] # Different
  
  PATIENT_CURRENT_OPIOIDS_METHADONE <- PATIENT_CURRENT_OPIOIDS[which(PATIENT_CURRENT_OPIOIDS$drug == 'Methadone'), ]
  
  concurrent_MME = sum(PATIENT_CURRENT_OPIOIDS$daily_dose)
  concurrent_MME_methadone = sum(PATIENT_CURRENT_OPIOIDS_METHADONE$daily_dose)
  
  return (c(concurrent_MME, concurrent_MME_methadone))
}

MME<- mcmapply(compute_pseudo_MME, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$concurrent_MME_pseudo = MME[1, ]
FULL$concurrent_MME_methadone_pseudo = MME[2, ]
rm(MME)

########################################################################
########################################################################

### Consecutive days based on the new time window (i.e. ignore gap within 7 days)
compute_pseudo_overlap <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date_filled <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_date_prescribed <- PATIENT_PRESC_OPIOIDS$date_prescribed
  real_overlap <- PATIENT_PRESC_OPIOIDS$overlap
  
  ## If real_overlap != 0, pseudo_overlap = real_overlap
  if(real_overlap != 0){
    return (real_overlap)
  } 
  
  ## If real_overlap == 0, look further 7 days
  presc_index <- which(PATIENT$prescription_id == presc_id)
  if(presc_index == 1){
    return (0)
  } else{
    # find the last presc_until before the current prescription
    prev_presc_until <- tail(PATIENT[which(as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                       as.Date(presc_date_filled, format = "%m/%d/%Y")), c("presc_until")], n=1)
    if(length(prev_presc_until) == 0){
      return (0)
    } else if(as.Date(prev_presc_until, format = "%m/%d/%Y") >= 
              as.Date(presc_date_prescribed, format = "%m/%d/%Y")){
      # Different: real_overlap == 0, but gap between two prescription < 7 days
      return (1)
    } else{
      return (0)
    }
  }
  
}
FULL$overlap_pseudo <- mcmapply(compute_pseudo_overlap, FULL$patient_id, FULL$prescription_id, mc.cores=40)

compute_pseudo_consecutive_days <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date_filled <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_date_prescribed <- PATIENT_PRESC_OPIOIDS$date_prescribed
  
  if(PATIENT_PRESC_OPIOIDS$overlap_pseudo == 0){
    return (0)
  } else{
    PATIENT_PREV_PRESC_OPIOIDS <- PATIENT[as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                            as.Date(presc_date_filled, format = "%m/%d/%Y"),]
    # must exist as at least one index has overlap 0, i.e. the first prescription
    last_index <- tail(which(PATIENT_PREV_PRESC_OPIOIDS$overlap_pseudo == 0), n=1)
    pseudo_consecutive_day = as.Date(PATIENT_PRESC_OPIOIDS$presc_until, format = "%m/%d/%Y") - 
      as.Date(PATIENT[last_index, c("date_filled")], format = "%m/%d/%Y")
    return (pseudo_consecutive_day)
  }
}
FULL$consecutive_days_pseudo <- mcmapply(compute_pseudo_consecutive_days, FULL$patient_id, FULL$prescription_id, mc.cores=40)

########################################################################
########################################################################
########################################################################

FULL_REORDER <- FULL %>% mutate(presence_MME_pseudo = concurrent_MME_pseudo - daily_dose,
                                presence_MME_methadone_pseudo = ifelse(drug == 'Methadone', 
                                                                       concurrent_MME_methadone_pseudo - daily_dose, 
                                                                       concurrent_MME_methadone_pseudo),
                                presence_days_pseudo = ifelse(consecutive_days_pseudo == 0, 0, consecutive_days_pseudo - days_supply)) %>% 
  select(c(prescription_id, patient_id, patient_birth_year, age, patient_gender, patient_zip,
           prescriber_id, prescriber_zip, pharmacy_id, pharmacy_zip, strength, quantity, days_supply,
           date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
           chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, MAINDOSE,
           payment, prescription_month, prescription_year,
           concurrent_MME, concurrent_MME_methadone, presence_MME, presence_MME_methadone,
           concurrent_MME_pseudo, concurrent_MME_methadone_pseudo,
           presence_MME_pseudo, presence_MME_methadone_pseudo,
           presence_num_prescribers, presence_num_pharmacies, num_prescribers, num_pharmacies,
           concurrent_benzo, concurrent_benzo_same, concurrent_benzo_diff,
           overlap, overlap_pseudo, consecutive_days, consecutive_days_pseudo, presence_days, presence_days_pseudo,
           Codeine, Hydrocodone, Oxycodone, Morphine, Hydromorphone, Methadone, Fentanyl, Oxymorphone,
           Medicaid, CommercialIns, Medicare, CashCredit, MilitaryIns, WorkersComp, Other, IndianNation,
           alert1, alert2, alert3, alert4, alert5, alert6, num_alert, long_term))

write.csv(FULL_REORDER, paste("../Data/FULL_", year, "_ALERT_NEW_4.csv", sep=""), row.names = FALSE)

########################################################################
############## Dummy prescriptions, switch effect ######################
########################################################################


library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2018

### Create dummy observation after alert ###
FULL <- read.csv(paste("../Data/FULL_", year, "_ALERT_NEW_4.csv", sep=""))
PATIENT_2019 <- read.csv("../Data/PATIENT_2019.csv")

## Old definition
FULL <- FULL %>% mutate(alert1_old = ifelse(concurrent_MME >= 100, 1, 0)) %>% 
  mutate(num_alert_old = alert1_old + alert2 + alert3 + alert4 + alert5 + alert6)

FULL <- FULL %>% mutate(alert = ifelse(num_alert > 0, 1, 0), alert_old = ifelse(num_alert_old > 0, 1, 0))
PATIENT_ALERT <- FULL %>% group_by(patient_id) %>% 
  summarize(num_prescriptions = n(),
            ever_alert1 = ifelse(sum(alert1) > 0, 1, 0),
            ever_alert1_old = ifelse(sum(alert1_old) > 0, 1, 0),
            ever_alert2 = ifelse(sum(alert2) > 0, 1, 0),
            ever_alert3 = ifelse(sum(alert3) > 0, 1, 0),
            ever_alert4 = ifelse(sum(alert4) > 0, 1, 0),
            ever_alert5 = ifelse(sum(alert5) > 0, 1, 0),
            ever_alert6 = ifelse(sum(alert6) > 0, 1, 0),
            ever_alert = ifelse(sum(alert) > 0, 1, 0),
            ever_alert_old = ifelse(sum(alert_old) > 0, 1, 0),
            alert1_date = ifelse(ever_alert1 == 1, date_filled[alert1 == 1][1], NA),
            alert1_date_old = ifelse(ever_alert1_old == 1, date_filled[alert1_old == 1][1], NA),
            alert2_date = ifelse(ever_alert2 == 1, date_filled[alert2 == 1][1], NA),
            alert3_date = ifelse(ever_alert3 == 1, date_filled[alert3 == 1][1], NA),
            alert4_date = ifelse(ever_alert4 == 1, date_filled[alert4 == 1][1], NA),
            alert5_date = ifelse(ever_alert5 == 1, date_filled[alert5 == 1][1], NA),
            alert6_date = ifelse(ever_alert6 == 1, date_filled[alert6 == 1][1], NA),
            alert_date = ifelse(ever_alert == 1, date_filled[alert == 1][1], NA),
            alert_date_old = ifelse(ever_alert_old == 1, date_filled[alert_old == 1][1], NA))
FULL <- left_join(FULL, PATIENT_ALERT, by = "patient_id")
rm(PATIENT_ALERT)

FULL <- FULL %>% mutate(prior_alert1 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert1_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert1_old = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert1_date_old,format = "%m/%d/%Y"), 1, 0),
                        prior_alert2 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert2_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert3 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert3_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert4 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert4_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert5 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert5_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert6 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert6_date,format = "%m/%d/%Y"), 1, 0),
                        prior_alert = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert_date,format = "%m/%d/%Y"), 1, 0)) %>%
  mutate(prior_alert1 = coalesce(prior_alert1, 0),
         prior_alert1_old = coalesce(prior_alert1_old, 0),
         prior_alert2 = coalesce(prior_alert2, 0),
         prior_alert3 = coalesce(prior_alert3, 0),
         prior_alert4 = coalesce(prior_alert4, 0),
         prior_alert5 = coalesce(prior_alert5, 0),
         prior_alert6 = coalesce(prior_alert6, 0),
         prior_alert = coalesce(prior_alert, 0)) %>% 
  mutate(active_alert1 = ifelse(presence_MME >= 90, 1, 0),
         active_alert1_old = ifelse(presence_MME >= 100, 1, 0),
         active_alert2 = ifelse(presence_MME_methadone >= 40, 1, 0),
         active_alert3 = ifelse(presence_num_prescribers >= 6, 1, 0),
         active_alert4 = ifelse(presence_num_pharmacies >= 6, 1, 0),
         active_alert5 = ifelse(presence_days >= 90, 1, 0), 
         active_alert1_pseudo = ifelse(presence_MME_pseudo >= 90, 1, 0),
         active_alert1_pseudo_old = ifelse(presence_MME_pseudo >= 100, 1, 0),
         active_alert2_pseudo = ifelse(presence_MME_methadone_pseudo >= 40, 1, 0),
         active_alert5_pseudo = ifelse(presence_days_pseudo >= 90, 1, 0))

############################## NEW version #####################################
## We only care about alert type 1 & 5 for now
# FULL <- FULL %>% dplyr::select(c(daily_dose, days_supply, quantity,
#                                  alert1, active_alert1, active_alert1_pseudo, prior_alert1, presence_MME, presence_MME_pseudo,
#                                  alert5, active_alert5, active_alert5_pseudo, prior_alert5, presence_days, presence_days_pseudo,
#                                  age, drug, payment, patient_gender, patient_id, prescriber_id, prescription_id)) %>%
#   mutate(alert = ifelse(alert1+alert5 > 0, 1, 0))

#### 1. Patient has more than one prescription
#### 2. Patient's last prescription triggers an alert
#### 3. Patient left the system (no records in 2019)

## Patient whose last prescription triggers an alert (overestimates the effect of alert)
# PATIENT_EVER_ALERT <- FULL %>% group_by(patient_id) %>%
#   summarize(total_prescriptions = n(), last_prescription_alert = ifelse(last(alert) == 1, 1, 0)) %>%
#   mutate(one_presc = ifelse(total_prescriptions < 2, 1, 0))

## We want to keep those that actually left (not in 2019)
# PATIENT_EVER_ALERT <- left_join(PATIENT_EVER_ALERT, PATIENT_2019, by = "patient_id") %>% filter(is.na(prescriptions)) 
# PATIENT_EVER_ALERT <- PATIENT_EVER_ALERT %>% filter(one_presc != 1)

## Get the last prescription of these patient, create dummy observations
# FULL_LAST_PRESCRIPTION <- left_join(FULL, PATIENT_EVER_ALERT, by = "patient_id") %>%
#   filter(last_prescription_alert == 1) %>% group_by(patient_id) %>% slice(n()) %>%
#   mutate(active_alert1 = alert1, active_alert1_pseudo = alert1, prior_alert1 = ifelse(prior_alert1 + alert1 > 0, 1, 0), presence_MME = daily_dose,
#          active_alert5 = alert5, active_alert5_pseudo = alert5, prior_alert5 = ifelse(prior_alert5 + alert5 > 0, 1, 0), presence_days = days_supply) %>%
#   mutate(quantity = 0, days_supply = 0, daily_dose = 0, alert = 1) %>%
#   dplyr::select(c(daily_dose, days_supply, quantity,
#                   alert1, active_alert1, active_alert1_pseudo, prior_alert1, presence_MME, presence_MME_pseudo,
#                   alert5, active_alert5, active_alert5_pseudo, prior_alert5, presence_days, presence_days_pseudo,
#                   age, drug, payment, patient_gender, patient_id, prescriber_id, prescription_id, alert))

############################## OLD version #####################################

FULL <- FULL %>% dplyr::select(c(daily_dose, days_supply, quantity,
                                 alert1_old, active_alert1_old, active_alert1_pseudo_old, prior_alert1_old, presence_MME, presence_MME_pseudo,
                                 alert5, active_alert5, active_alert5_pseudo, prior_alert5, presence_days, presence_days_pseudo,
                                 age, drug, payment, patient_gender, patient_id, prescriber_id, prescription_id)) %>%
  mutate(alert_old = ifelse(alert1_old+alert5 > 0, 1, 0))

## Patient whose last prescription triggers an alert (overestimates the effect of alert)
PATIENT_EVER_ALERT <- FULL %>% group_by(patient_id) %>%
  summarize(total_prescriptions = n(), last_prescription_alert_old = ifelse(last(alert_old) == 1, 1, 0)) %>%
  mutate(one_presc = ifelse(total_prescriptions < 2, 1, 0))

## We want to keep those that actually left (not in 2019)
PATIENT_EVER_ALERT <- left_join(PATIENT_EVER_ALERT, PATIENT_2019, by = "patient_id") %>% filter(is.na(prescriptions)) 
PATIENT_EVER_ALERT <- PATIENT_EVER_ALERT %>% filter(one_presc != 1)

FULL_LAST_PRESCRIPTION <- left_join(FULL, PATIENT_EVER_ALERT, by = "patient_id") %>%
  filter(last_prescription_alert_old == 1) %>% group_by(patient_id) %>% slice(n()) %>%
  mutate(active_alert1_old = alert1_old, active_alert1_pseudo_old = alert1_old, prior_alert1_old = ifelse(prior_alert1_old + alert1_old > 0, 1, 0), presence_MME = daily_dose,
         active_alert5 = alert5, active_alert5_pseudo = alert5, prior_alert5 = ifelse(prior_alert5 + alert5 > 0, 1, 0), presence_days = days_supply) %>%
  mutate(quantity = 0, days_supply = 0, daily_dose = 0, alert_old = 1) %>%
  dplyr::select(c(daily_dose, days_supply, quantity,
                  alert1_old, active_alert1_old, active_alert1_pseudo_old, prior_alert1_old, presence_MME, presence_MME_pseudo,
                  alert5, active_alert5, active_alert5_pseudo, prior_alert5, presence_days, presence_days_pseudo,
                  age, drug, payment, patient_gender, patient_id, prescriber_id, prescription_id, alert_old))

########################################################################

FULL_DUMMY <- rbind(FULL, FULL_LAST_PRESCRIPTION) %>% arrange(patient_id, prescription_id)
rm(PATIENT_EVER_ALERT)
rm(FULL_LAST_PRESCRIPTION)

### Switch in drug type
FULL_DUMMY = FULL_DUMMY %>% mutate(drug_switch = ifelse(drug != dplyr::lag(drug) & patient_id == lag(patient_id), 
                                                        paste(dplyr::lag(drug), drug, sep = "_"), "None"))
FULL_DUMMY[1,]$drug_switch = "None"

write.csv(FULL_DUMMY, paste("../Data/FULL_DUMMY_", year, "_4.csv", sep=""), row.names = FALSE)












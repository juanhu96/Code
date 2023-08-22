library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")

PATIENT <- FULL_2019 %>% group_by(patient_id) %>% summarize(first_presc_date = date_filled[1],
                                                            longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                            longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA),
                                                            first_longterm_presc = ifelse(sum(long_term) > 0, min(row_number()[long_term > 0]), NA),
                                                            first_longterm_presc_id = ifelse(sum(long_term) > 0, prescription_id[long_term > 0][1], NA))
FULL_2019 <- left_join(FULL_2019, PATIENT, by = "patient_id")

## We keep prescriptions filled before end date of the long term prescription
FULL_2019 <- FULL_2019 %>% mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - as.Date(date_filled, format = "%m/%d/%Y"))) %>%
  filter(days_to_long_term > 0 | is.na(days_to_long_term)) %>% # either never long-term or haven't
  mutate(long_term_180 = ifelse(days_to_long_term <= 180, 1, 0)) %>% # within 180 days
  mutate(long_term_180 = ifelse(is.na(long_term_180), 0, long_term_180)) # never is also 0

FULL_2019 <- FULL_2019 %>% select(-c(first_presc_date,
                                     longterm_filled_date, longterm_presc_date, 
                                     first_longterm_presc, first_longterm_presc_id))

nrow(TEST[TEST$long_term_180 == 1,]) # 3315206
TEMP <- TEST[1:200,]

PATIENT <- TEST %>% group_by(patient_id) %>% summarize(longterm = ifelse(sum(long_term) > 0, 1, 0))
TEMP <- PATIENT[1:100,]
nrow(PATIENT[PATIENT$longterm==1,]) # 536974

TEMP <- TEST[1:500, ] %>% select(c(patient_id, date_filled, days_supply, presc_until, overlap, consecutive_days, opioid_days, long_term, long_term_180))

########################################################################
########################################################################
########################################################################

FULL_ALL <- read.csv("../Data/FULL_2019_LONGTERM.csv")
FULL_ALL$prescription_id = seq.int(nrow(FULL_ALL))
PATIENT <- FULL_ALL %>% filter(patient_id == 439) %>% select(c(patient_id, date_filled, days_supply, 
                                                               presc_until, overlap, consecutive_days, 
                                                               opioid_days, daily_dose, concurrent_MME,
                                                               num_prescribers, num_pharmacies, 
                                                               long_term, long_term_180))

# FULL <- FULL_ALL %>% filter(patient_id < 41846152)
# FULL <- FULL_ALL %>% filter(patient_id >= 41846152 & patient_id < 54195156)
# FULL <- FULL_ALL %>% filter(patient_id >= 54195156 & patient_id < 68042918)
FULL <- FULL_ALL %>% filter(patient_id >= 68042918)

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

## Double check
# PATIENT <- FULL %>% filter(patient_id == 439) %>% select(c(prescription_id, patient_id, date_filled, days_supply, 
#                                                            presc_until, overlap, consecutive_days, 
#                                                            opioid_days, daily_dose, concurrent_MME,
#                                                            num_prescribers, num_pharmacies, 
#                                                           long_term, long_term_180))

# TEMP <- FULL[1:200,]

########################################################################
########################################################################
########################################################################

## Compute concurrent MME
compute_MME <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  PATIENT_CURRENT_OPIOIDS <- PATIENT[which(as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                             as.Date(presc_date, format = "%m/%d/%Y") & 
                                             as.Date(PATIENT$presc_until,format = "%m/%d/%Y") >
                                             as.Date(presc_date, format = "%m/%d/%Y")),]
  
  PATIENT_CURRENT_OPIOIDS_METHADONE <- PATIENT_CURRENT_OPIOIDS[which(PATIENT_CURRENT_OPIOIDS$drug == 'Methadone'), ]
  
  concurrent_MME = sum(PATIENT_CURRENT_OPIOIDS$daily_dose)
  concurrent_MME_methadone = sum(PATIENT_CURRENT_OPIOIDS_METHADONE$daily_dose)
  
  return (c(concurrent_MME, concurrent_MME_methadone))
}

MME <- mcmapply(compute_MME, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$concurrent_MME = MME[1, ]
FULL$concurrent_MME_methadone = MME[2, ]

########################################################################

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

write.csv(FULL, "../Data/FULL_2019_LONGTERM_4.csv", row.names = FALSE)

########################################################################
########################################################################
########################################################################

FULL_2019_1 <- read.csv("../Data/FULL_2019_LONGTERM_1.csv")
FULL_2019_2 <- read.csv("../Data/FULL_2019_LONGTERM_2.csv")
FULL_2019_3 <- read.csv("../Data/FULL_2019_LONGTERM_3.csv")
FULL_2019_4 <- read.csv("../Data/FULL_2019_LONGTERM_4.csv")

FULL_ALL <- rbind(FULL_2019_1, FULL_2019_2)
FULL_ALL <- rbind(FULL_ALL, FULL_2019_3)
FULL_ALL <- rbind(FULL_ALL, FULL_2019_4)
rm(FULL_2019_1)
rm(FULL_2019_2)
rm(FULL_2019_3)
rm(FULL_2019_4)

# FULL_OLD <- read.csv("../Data/FULL_2019_LONGTERM.csv")
write.csv(FULL_ALL, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)


FULL_ALL <- read.csv("../Data/FULL_2019_LONGTERM.csv")
FULL_ALL <- FULL_ALL %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                     prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                     conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                     DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_year,
                                     overlap, # alert1, alert2, alert3, alert4, alert5, alert6,
                                     days_to_long_term, previous_dose, previous_concurrent_MME, previous_days, num_alert, opioid_days, long_term))

FULL_ALL <- rename(FULL_ALL, concurrent_methadone_MME = concurrent_MME_methadone)
FULL_ALL <- rename(FULL_ALL, MME_diff = concurrent_MME_diff)

write.csv(FULL_ALL, "../Data/FULL_2019_INPUT_LONGTERM.csv", row.names = FALSE)

########################################################################

FULL_2018_1 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_1.csv")
FULL_2018_2 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_2.csv")
FULL_2018_3 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_3.csv")
FULL_2018_4 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_4.csv")

FULL_2018 <- rbind(FULL_2018_1, FULL_2018_2)
FULL_2018 <- rbind(FULL_2018, FULL_2018_3)
FULL_2018 <- rbind(FULL_2018, FULL_2018_4)
rm(FULL_2018_1)
rm(FULL_2018_2)
rm(FULL_2018_3)
rm(FULL_2018_4)


patient_2018 <- unique(FULL_2018$patient_id)
TEST <- FULL_ALL %>% filter(!patient_id %in% patient_2018)
length(unique(TEST$patient_id))



FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")
col_full <- colnames(FULL_2019)
FULL_2019_INPUT <- read.csv("../Data/FULL_2019_LONGTERM_INPUT.csv")
col_input <- colnames(FULL_2019_INPUT)

col_full[!(col_full %in% col_input)]
col_input[!(col_input %in% col_full)]

FULL_2019 <- read.csv("../Data/FULL_2019_STUMPS0.csv")


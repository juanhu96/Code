### Identify if a patient is eventually risky, and the date of risky
### For prescriptions from patient with at least two prescriptions
library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2019
FULL <- read.csv(paste("../Data/FULL_OPIOID_", year, "_ATLEASTTWO_2.csv", sep=""))
FULL$prescription_id = seq.int(nrow(FULL))

BENZO_TABLE <- read.csv(paste("../Data/FULL_BENZO_", year, ".csv", sep=""))

########################################################################
########### Determine if a patient is ever being alerted ###############
########################################################################

### Patient alerts if any of the following satisfies:
## More than 90 MME (daily dose) per day
## More than 40 MME (daily dose) methadone
## 6 or more perscribers last 6 months
## 6 or more pharmacies last 6 months
## On opioids more than 90 consecutive days
## On both benzos and opioids

FULL$presc_until <- as.Date(FULL$date_filled, format = "%m/%d/%Y") + FULL$days_supply
FULL$presc_until <- format(FULL$presc_until, "%m/%d/%Y")
FULL$age <- year(as.POSIXlt(FULL$date_filled, format="%m/%d/%Y")) - FULL$patient_birth_year

########################################################################
################ 1. More than 90 MME per day  ##########################
################ 2. More than 40 MME methadone per day  ################
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

# MME <- future_mapply(compute_MME, FULL$patient_id, FULL$prescription_id)
MME <- mcmapply(compute_MME, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$concurrent_MME = MME[1, ]
FULL$concurrent_MME_methadone = MME[2, ]

########################################################################
################ 3. 6 or more prescribers last 6 months  ###############
################ 4. 6 or more pharmacies last 6 months  ################
########################################################################

## Compute number of prescribers and pharmacies in last 6 months
compute_num_prescriber_pharmacy <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  # in last 6 months
  PATIENT_CURRENT_OPIOIDS <- PATIENT[which(as.Date(PATIENT$date_filled,format = "%m/%d/%Y") >= 
                                             as.Date(presc_date, format = "%m/%d/%Y") - 180 & 
                                             as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                             as.Date(presc_date, format = "%m/%d/%Y")),]
  
  num_prescribers = length(unique(PATIENT_CURRENT_OPIOIDS$prescriber_id))
  num_pharmacies = length(unique(PATIENT_CURRENT_OPIOIDS$pharmacy_id))
  
  return (c(num_prescribers, num_pharmacies))
}

# num_prescriber_pharmacy <- future_mapply(compute_num_prescriber_pharmacy, FULL$patient_id, FULL$prescription_id)
num_prescriber_pharmacy <- mcmapply(compute_num_prescriber_pharmacy, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$num_prescribers = num_prescriber_pharmacy[1, ]
FULL$num_pharmacies = num_prescriber_pharmacy[2, ]

########################################################################
########### 5. On opioids more than 90 consecutive days  ###############
########################################################################

# setwd("/mnt/phd/jihu/opioid/Code")
# year = 2018
# T_period = 30
# FULL_RAW <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))
# FULL_RAW <- FULL_RAW %>% arrange(patient_id, date_filled)

# FULL <- FULL_RAW %>% filter(patient_id < 44615422)
# FULL <- FULL_RAW %>% filter(patient_id >= 44615422 & patient_id < 52056873)
# FULL <- FULL_RAW %>% filter(patient_id >= 52056873 & patient_id < 68534830)
# FULL <- FULL_RAW %>% filter(patient_id >= 68534830)

## Days overlap with previous prescription (exclude co-prescriptions)
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

## Consecutive days, find the first prescription that starts overlap
# Days = end date of current prescription - start date of first consecutive prescription
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
# write.csv(FULL, paste("../Data/FULL_", year, "_4.csv", sep=""), row.names = FALSE)

########################################################################
################### 6. On both benzos and opioids  #####################
########################################################################

## Compute concurrent benzos
compute_concurrent_benzo <- function(pat_id, presc_id){
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_until <- PATIENT_PRESC_OPIOIDS$presc_until
  prescriber_id <- PATIENT_PRESC_OPIOIDS$prescriber_id
  
  PATIENT_PRESC_BENZOS <- BENZO_TABLE[which(BENZO_TABLE$patient_id == pat_id), ]
  
  if (nrow(PATIENT_PRESC_BENZOS) == 0){
    return (c(0, 0))
  } else{
    # same prescriber
    BENZO_BEFORE <- PATIENT_PRESC_BENZOS[as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") <= 
                                           as.Date(presc_date, format = "%m/%d/%Y") & 
                                           as.Date(PATIENT_PRESC_BENZOS$presc_until, format = "%m/%d/%Y") > 
                                           as.Date(presc_date, format = "%m/%d/%Y"), ]
    
    BENZO_AFTER <- PATIENT_PRESC_BENZOS[as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") > 
                                          as.Date(presc_date, format = "%m/%d/%Y") & 
                                          as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") <= 
                                          as.Date(presc_until, format = "%m/%d/%Y"), ]
    
    num_benzo_before <- nrow(BENZO_BEFORE)
    num_benzo_after <- nrow(BENZO_AFTER)
    num_benzo <- num_benzo_before + num_benzo_after
    
    num_benzo_before_same <- nrow(BENZO_BEFORE[BENZO_BEFORE$prescriber_id == prescriber_id, ])
    num_benzo_after_same <- nrow(BENZO_AFTER[BENZO_AFTER$prescriber_id == prescriber_id, ])
    num_benzo_same = num_benzo_before_same + num_benzo_after_same
      
    if(num_benzo > 0){
      return (c(num_benzo, num_benzo_same))
    } else {
      return (c(0, 0))
    }
  }
}

# concurrent_benzo_prescriber <- future_mapply(compute_concurrent_benzo, FULL$patient_id, FULL$prescription_id)
concurrent_benzo_prescriber <- mcmapply(compute_concurrent_benzo, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$concurrent_benzo <- concurrent_benzo_prescriber[1, ]
FULL$concurrent_benzo_same <- concurrent_benzo_prescriber[2, ]
FULL$concurrent_benzo_diff <- FULL$concurrent_benzo - FULL$concurrent_benzo_same

########################################################################
########################### PATIENT ALERT  #############################
########################################################################

### If more than one of the conditions satisfies, we pick the date of the first alarm
## More than 90 MME (daily dose) per day
## More than 40 MME (daily dose) methadone
## 6 or more perscribers last 6 months
## 6 or more pharmacies last 6 months
## On opioids more than 90 consecutive days
## On both benzos and opioids

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

########################################################################
########################################################################
########################################################################

### Original
FULL_REORDER <- select(FULL, prescription_id, patient_id, patient_birth_year, age, patient_gender, patient_zip,
                       prescriber_id, prescriber_zip, pharmacy_id, pharmacy_zip, strength, quantity, days_supply,
                       date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
                       chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, MAINDOSE,
                       payment, prescription_month, prescription_year,
                       concurrent_MME, concurrent_MME_methadone, num_prescribers, num_pharmacies, 
                       concurrent_benzo, concurrent_benzo_same, concurrent_benzo_diff, 
                       overlap, consecutive_days,
                       alert1, alert2, alert3, alert4, alert5, alert6, num_alert)

write.csv(FULL_REORDER, paste("../Data/FULL_ALERT_", year, "_ATLEASTTWO_2.csv", sep=""), row.names = FALSE)
rm(BENZO_TABLE)
rm(FULL_REORDER)




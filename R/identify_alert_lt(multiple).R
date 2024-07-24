### STEP 2
### Identify CURES alert (patient w/ 2+ prescriptions)

### STEP 3
### Identify longterm & longterm180 (patients w/ 2+ prescriptions)
### Keep prescriptions up to first long term

### INPUT: FULL_OPIOID_2018_ATLEASTTWO.csv
### OUTPUT: FULL_OPIOID_2018_ATLEASTTWO_TEMP.csv

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2019
case = "3"

FULL_ATLEASTTWO <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, ".csv", sep=""))
BENZO_TABLE <- read.csv(paste("FULL_BENZO_", year, ".csv", sep=""))

########################################################################
## Prescription until, Age

setDT(FULL_ATLEASTTWO)

FULL_ATLEASTTWO[, prescription_id := seq_len(.N)]
FULL_ATLEASTTWO[, date_filled := format(as.Date(date_filled, format="%m/%d/%Y"), "%m/%d/%Y")]
FULL_ATLEASTTWO[, age := year(as.Date(date_filled, format = "%m/%d/%Y")) - patient_birth_year]

FULL_ATLEASTTWO <- as.data.frame(FULL_ATLEASTTWO)

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

########################################################################
################ 1. More than 90 MME per day  ##########################
################ 2. More than 40 MME methadone per day  ################
########################################################################

## Compute concurrent MME at the date filled
compute_MME <- function(pat_id, presc_id){
  
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  PATIENT_CURRENT_OPIOIDS <- PATIENT[which(as.Date(PATIENT$date_filled,format = "%m/%d/%Y") <= 
                                       as.Date(presc_date, format = "%m/%d/%Y") & 
                                       as.Date(PATIENT$presc_until,format = "%m/%d/%Y") >
                                       as.Date(presc_date, format = "%m/%d/%Y")),]
  
  PATIENT_CURRENT_OPIOIDS_METHADONE <- PATIENT_CURRENT_OPIOIDS[which(PATIENT_CURRENT_OPIOIDS$drug == 'Methadone'), ]
  
  concurrent_MME = sum(PATIENT_CURRENT_OPIOIDS$daily_dose)
  concurrent_methadone_MME = sum(PATIENT_CURRENT_OPIOIDS_METHADONE$daily_dose)
  
  return (c(concurrent_MME, concurrent_methadone_MME))
}

MME <- mcmapply(compute_MME, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
FULL_ATLEASTTWO$concurrent_MME = MME[1, ]
FULL_ATLEASTTWO$concurrent_methadone_MME = MME[2, ]
# write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)
# rm(MME)
TEST <- FULL_ATLEASTTWO[1:20,]

########################################################################
################ 3. 6 or more prescribers last 6 months  ###############
################ 4. 6 or more pharmacies last 6 months  ################
########################################################################

## Compute number of prescribers and pharmacies in last 6 months
# - including current prescription
compute_num_prescriber_pharmacy <- function(pat_id, presc_id){
  
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
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

num_prescriber_pharmacy <- mcmapply(compute_num_prescriber_pharmacy, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
FULL_ATLEASTTWO$num_prescribers_past180 = num_prescriber_pharmacy[1, ]
FULL_ATLEASTTWO$num_pharmacies_past180 = num_prescriber_pharmacy[2, ]
write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)
# rm(num_prescriber_pharmacy)


########################################################################
########### 5. On opioids more than 90 consecutive days  ###############
########################################################################

## Days overlap with previous prescription (exclude co-prescriptions)
compute_overlap <- function(pat_id, presc_id){
  
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  if(presc_index == 1){
    return (0)
  } else{
    # prescriptions before the current (exclude co-prescriptions)
    prev_presc_until <- tail(PATIENT[as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                       as.Date(presc_date, format = "%m/%d/%Y"), c("presc_until")], n=1) # [[1]]
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

FULL_ATLEASTTWO$overlap <- mcmapply(compute_overlap, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
TEST <- FULL_ATLEASTTWO[1:20,]

## Consecutive days, find the first prescription that starts overlap
# Days = end date of current prescription - start date of first consecutive prescription
compute_consecutive_days <- function(pat_id, presc_id){
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  days_current <- PATIENT_PRESC_OPIOIDS$days_supply
  
  # Modified: if no overlap, consecutive days should be the current days_supply
  if(PATIENT_PRESC_OPIOIDS$overlap == 0){
    return (days_current)
  } else{
    PATIENT_PREV_PRESC_OPIOIDS <- PATIENT[as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <
                                            as.Date(presc_date, format = "%m/%d/%Y"),]
    # must exist as at least one index has overlap 0, i.e. the first prescription
    # last_index <- tail(which(PATIENT_PREV_PRESC_OPIOIDS$overlap == 0), n=1)[[1]]
    # consecutive_day = as.Date(PATIENT_PRESC_OPIOIDS$presc_until, format = "%m/%d/%Y") -
    #   as.Date(PATIENT[last_index, c("date_filled")][[1]], format = "%m/%d/%Y")
    last_index <- tail(which(PATIENT_PREV_PRESC_OPIOIDS$overlap == 0), n=1)
    consecutive_day = as.Date(PATIENT_PRESC_OPIOIDS$presc_until, format = "%m/%d/%Y") -
      as.Date(PATIENT[last_index, c("date_filled")], format = "%m/%d/%Y")
    return (consecutive_day)
  }
}

FULL_ATLEASTTWO$consecutive_days <- mcmapply(compute_consecutive_days, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)

########################################################################
################### 6. On both benzos and opioids  #####################
########################################################################

## Compute concurrent benzos
compute_concurrent_benzo <- function(pat_id, presc_id){
  
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
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
    
    num_benzo <- nrow(BENZO_BEFORE) + nrow(BENZO_AFTER)
    num_benzo_same = nrow(BENZO_BEFORE[BENZO_BEFORE$prescriber_id == prescriber_id, ]) + nrow(BENZO_AFTER[BENZO_AFTER$prescriber_id == prescriber_id, ])
      
    if(num_benzo > 0){
      return (c(num_benzo, num_benzo_same))
    } else {
      return (c(0, 0))
    }
  }
}

concurrent_benzo_prescriber <- mcmapply(compute_concurrent_benzo, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
FULL_ATLEASTTWO$concurrent_benzo <- concurrent_benzo_prescriber[1, ]
FULL_ATLEASTTWO$concurrent_benzo_same <- concurrent_benzo_prescriber[2, ]
FULL_ATLEASTTWO$concurrent_benzo_diff <- FULL_ATLEASTTWO$concurrent_benzo - FULL_ATLEASTTWO$concurrent_benzo_same

# write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)

# rm(concurrent_benzo_prescriber)
# rm(BENZO_TABLE)

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
  
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  
  alert1 = 0
  alert2 = 0
  alert3 = 0
  alert4 = 0
  alert5 = 0
  alert6 = 0
  
  if(PATIENT_PRESC_OPIOIDS$concurrent_MME >= 90){alert1 = 1} 
  if(PATIENT_PRESC_OPIOIDS$concurrent_methadone_MME >= 40){alert2 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_prescribers >= 6){alert3 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_pharmacies >= 6){alert4 = 1}
  if(PATIENT_PRESC_OPIOIDS$consecutive_days >= 90){alert5 = 1}
  if(PATIENT_PRESC_OPIOIDS$concurrent_benzo > 0){alert6 = 1}
    
  return (c(alert1, alert2, alert3, alert4, alert5, alert6))
}

alert <- mcmapply(patient_alert, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)

for (i in 1:6) {
  FULL_ATLEASTTWO[[paste0("alert", i)]] <- alert[i, ]
}

FULL_ATLEASTTWO$num_alert <- rowSums(FULL_ATLEASTTWO[, paste0("alert", 1:6)])
FULL_ATLEASTTWO$any_alert <- as.integer(FULL_ATLEASTTWO$num_alert > 0)

# write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)

# colnames(FULL_ATLEASTTWO)
# [29] "chronic"                      "num_presc"                    "max_dose"                     "outliers"                    
# [33] "prescription_month"           "prescription_year"            "num_prescriptions"            "presc_until"                 
# [37] "prescription_id"              "age"                          "concurrent_MME"               "concurrent_methadone_MME"    
# [41] "num_prescribers_past180"      "num_pharmacies_past180"       "overlap"                      "consecutive_days"            
# [45] "concurrent_benzo"             "concurrent_benzo_same"        "concurrent_benzo_diff"        "alert1"                      
# [49] "alert2"                       "alert3"                       "alert4"                       "alert5"                      
# [53] "alert6"                       "num_alert"                    "any_alert"  

# colnames(FULL_SINGLE)
# [29] "chronic"                      "num_presc"                    "max_dose"                     "outliers"                    
# [33] "prescription_month"           "prescription_year"            "num_prescriptions"            "presc_until"                 
# [37] "prescription_id"              "age"                          "concurrent_MME"               "concurrent_methadone_MME"    
# [41] "num_prescribers"              "num_pharmacies"               "overlap"                      "consecutive_days"            
# [45] "concurrent_benzo"             "concurrent_benzo_same"        "concurrent_benzo_diff"        "alert1"                      
# [49] "alert2"                       "alert3"                       "alert4"                       "alert5"                      
# [53] "alert6"                       "num_alert"                    "any_alert" 

########################################################################
############ Determine LT and drop prescriptions afterwards ############
########################################################################

### Long-term use: at least 90 day in 180 days period
### Outcome variable: at the end of current prescription,
### Will the patient become a long-term user?

### WARNING: for prescriptions in early 2018
### their 180 days window include prescription from 2017

FULL_ATLEASTTWO$period_start = format(as.Date(FULL_ATLEASTTWO$presc_until, format = "%m/%d/%Y") - 180, "%m/%d/%Y")
FULL_ATLEASTTWO <- FULL_ATLEASTTWO[order(FULL_ATLEASTTWO$patient_id, as.Date(FULL_ATLEASTTWO$date_filled, format = "%m/%d/%Y"), as.Date(FULL_ATLEASTTWO$presc_until, format = "%m/%d/%Y")),]

# overlap with the previous (index based) prescription
# this is different from overlap above
# e.g. three prescriptions, 03/01-03/20, 03/15-03/25, 03/15-03/25
# previous overlap computes the 2&3 prescription with the first, i.e., 5 days
# current computes the overlap with previous prescription, i.e., 5 and 0 days, so that we don't double count.
compute_overlap <- function(pat_id, presc_id){
  
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  if(presc_index == 1){
    return (0)
  } else{
    prev_presc_until <- PATIENT[presc_index - 1, c("presc_until")]
    if(as.Date(prev_presc_until, format = "%m/%d/%Y") >= as.Date(presc_date, format = "%m/%d/%Y")){
      return (as.numeric(as.Date(prev_presc_until, format = "%m/%d/%Y") - as.Date(presc_date, format = "%m/%d/%Y")))
    } else{
      return (0)
    }
  }
}

FULL_ATLEASTTWO$overlap_lt <- mcmapply(compute_overlap, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)

compute_opioid_days <- function(pat_id, presc_id){
  
  PATIENT <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_ATLEASTTWO[which(FULL_ATLEASTTWO$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_until <- PATIENT_PRESC_OPIOIDS$presc_until
  period_start <- PATIENT_PRESC_OPIOIDS$period_start
  days_supply <- PATIENT_PRESC_OPIOIDS$days_supply
  overlap_lt <- PATIENT_PRESC_OPIOIDS$overlap_lt
  
  # prescriptions before (in terms of time & index)
  # index for case where multiple prescription on same date
  PATIENT_PREV_PRESC <- PATIENT[which(as.Date(PATIENT$presc_until, format = "%m/%d/%Y") > 
                                        as.Date(period_start, format = "%m/%d/%Y") &
                                        as.Date(PATIENT$date_filled, format = "%m/%d/%Y") < # <=
                                        as.Date(presc_date, format = "%m/%d/%Y") &
                                        PATIENT$prescription_id < presc_id), ]
  
  if(nrow(PATIENT_PREV_PRESC) == 0){
    opioid_days = days_supply
  } else{
    
    first_presc_date <- PATIENT_PREV_PRESC[1, c("date_filled")]
    first_presc_until <- PATIENT_PREV_PRESC[1, c("presc_until")]
    first_days_supply <- PATIENT_PREV_PRESC[1, c("days_supply")]
    
    if(as.Date(first_presc_date, format = "%m/%d/%Y") <= as.Date(period_start, format = "%m/%d/%Y")){
      first_accumulate = as.numeric(as.Date(first_presc_until, format = "%m/%d/%Y") - as.Date(period_start, format = "%m/%d/%Y"))
    } else{
      first_accumulate = first_days_supply
    }
    
    if (nrow(PATIENT_PREV_PRESC) == 1){
      opioid_days = first_accumulate + days_supply - overlap_lt
    }
    else{
      # overlap_lt considers the co-prescription, but days supply not
      total_overlap_lt = sum(PATIENT_PREV_PRESC[2:nrow(PATIENT_PREV_PRESC), c("overlap_lt")])
      total_days_supply = sum(PATIENT_PREV_PRESC[2:nrow(PATIENT_PREV_PRESC), c("days_supply")])
      opioid_days = first_accumulate + total_days_supply + days_supply - total_overlap_lt - overlap_lt
    }
  }
  # Corner case: I can't think of a better way of doing this for now
  if(opioid_days > 180){
    opioid_days = 180
  }
  return (opioid_days)
}

FULL_ATLEASTTWO$opioid_days <- mcmapply(compute_opioid_days, FULL_ATLEASTTWO$patient_id, FULL_ATLEASTTWO$prescription_id, mc.cores=50)
FULL_ATLEASTTWO <- FULL_ATLEASTTWO %>% mutate(long_term = ifelse(opioid_days >= 90, 1, 0))
write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)

########################################################################

### Keep prescriptions up to long-term use, once it is long-term use, assume absorbing state
PATIENT <- FULL_ATLEASTTWO %>% 
  group_by(patient_id) %>% 
  summarize(first_presc_date = date_filled[1],
            longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
            longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA),
            first_longterm_presc = ifelse(sum(long_term) > 0, min(row_number()[long_term > 0]), NA),
            first_longterm_presc_id = ifelse(sum(long_term) > 0, prescription_id[long_term > 0][1], NA))

# PAT <- PATIENT[1:20,]

FULL_ATLEASTTWO <- left_join(FULL_ATLEASTTWO, PATIENT, by = "patient_id")

## Use presc_until of the long-term prescription to compute
# NA: patient never become long term
# >0: patient is going to become long term
# =0: patient is long term right after this prescription
# <0: patient is already long term
FULL_ATLEASTTWO <- FULL_ATLEASTTWO %>% 
  mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - as.Date(date_filled, format = "%m/%d/%Y"))) %>%
  filter(days_to_long_term > 0 | is.na(days_to_long_term)) %>% # either never long-term or haven't
  mutate(long_term_180 = ifelse(days_to_long_term <= 180, 1, 0)) %>% # within 180 days
  mutate(long_term_180 = ifelse(is.na(long_term_180), 0, long_term_180)) # never is also 0

FULL_ATLEASTTWO <- FULL_ATLEASTTWO %>% 
  select(-c(period_start, first_presc_date,
            longterm_filled_date, longterm_presc_date, 
            first_longterm_presc, first_longterm_presc_id))

# FULL_ATLEASTTWO <- rename(FULL_ATLEASTTWO, concurrent_methadone_MME = concurrent_MME_methadone)
write.csv(FULL_ATLEASTTWO, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", case, "_TEMP.csv", sep=""), row.names = FALSE)

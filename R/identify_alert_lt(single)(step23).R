### STEP 2
### Identify CURES alert (for patients w/ single prescriptions)

### STEP 3
### Identify longterm & longterm180 (for patients w/ single prescriptions)
### Keep prescriptions up to first long term

### INPUT: FULL_OPIOID_2018_ONE.csv
### OUTPUT: FULL_OPIOID_2018_ONE_TEMP.csv

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2019
  
FULL_SINGLE <- read.csv(paste("FULL_OPIOID_", year, "_ONE.csv", sep=""))
BENZO_TABLE <- read.csv(paste("FULL_BENZO_", year, ".csv", sep=""))

########################################################################
## Prescription until, Age

setDT(FULL_SINGLE)
setDT(BENZO_TABLE)

FULL_SINGLE[, prescription_id := seq_len(.N)]
FULL_SINGLE[, age := year(as.Date(date_filled, format = "%m/%d/%Y")) - patient_birth_year]

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

FULL_SINGLE[, concurrent_MME := daily_dose]
FULL_SINGLE[, concurrent_methadone_MME := ifelse(drug == 'Methadone', daily_dose, 0)]
FULL_SINGLE[, c("num_prescribers", "num_pharmacies") := list(1, 1)]
FULL_SINGLE[, overlap := 0]
FULL_SINGLE[, consecutive_days := days_supply]

########################################################################
### Compute concurrent benzos
compute_concurrent_benzo <- function(pat_id, presc_id){
  
  PATIENT_PRESC_BENZOS <- BENZO_TABLE[which(BENZO_TABLE$patient_id == pat_id), ]
  
  if (nrow(PATIENT_PRESC_BENZOS) == 0){
    return (c(0, 0))
  } else{
    
    PATIENT_PRESC_OPIOIDS <- FULL_SINGLE[which(FULL_SINGLE$prescription_id == presc_id),]
    presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
    presc_until <- PATIENT_PRESC_OPIOIDS$presc_until
    prescriber_id <- PATIENT_PRESC_OPIOIDS$prescriber_id
    
    
    BENZO_BEFORE <- PATIENT_PRESC_BENZOS[as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") <=
                                           as.Date(presc_date, format = "%m/%d/%Y") &
                                           as.Date(PATIENT_PRESC_BENZOS$presc_until, format = "%m/%d/%Y") >
                                           as.Date(presc_date, format = "%m/%d/%Y"), ]
    BENZO_AFTER <- PATIENT_PRESC_BENZOS[as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") >
                                          as.Date(presc_date, format = "%m/%d/%Y") &
                                          as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") <=
                                          as.Date(presc_until, format = "%m/%d/%Y"), ]
    num_benzo <- nrow(BENZO_BEFORE) + nrow(BENZO_AFTER)

    # same as the opioid prescriber
    num_benzo_same = nrow(BENZO_BEFORE[BENZO_BEFORE$prescriber_id == prescriber_id, ]) + nrow(BENZO_AFTER[BENZO_AFTER$prescriber_id == prescriber_id, ])

    if(num_benzo > 0){
      return (c(num_benzo, num_benzo_same))
    } else {
      return (c(0, 0))
    }
  }
}

FULL_SINGLE <- as.data.frame(FULL_SINGLE)

# TEST <- FULL_SINGLE[1:200,]
# concurrent_benzo_prescriber <- mcmapply(compute_concurrent_benzo, TEST$patient_id, TEST$prescription_id, mc.cores=40)
# TEST$concurrent_benzo <- concurrent_benzo_prescriber[1, ]
# TEST$concurrent_benzo_same <- concurrent_benzo_prescriber[2, ]

concurrent_benzo_prescriber <- mcmapply(compute_concurrent_benzo, FULL_SINGLE$patient_id, FULL_SINGLE$prescription_id, mc.cores=40)

FULL_SINGLE$concurrent_benzo <- concurrent_benzo_prescriber[1, ]
FULL_SINGLE$concurrent_benzo_same <- concurrent_benzo_prescriber[2, ]
FULL_SINGLE$concurrent_benzo_diff <- FULL_SINGLE$concurrent_benzo - FULL_SINGLE$concurrent_benzo_same

# write.csv(FULL_SINGLE, paste("FULL_OPIOID_", year, "_ONETEMP.csv", sep=""), row.names = FALSE)
rm(BENZO_TABLE)
rm(concurrent_benzo_prescriber)

########################################################################
### Determine alert
patient_alert <- function(pat_id, presc_id){
  
  PATIENT_PRESC_OPIOIDS <- FULL_SINGLE[which(FULL_SINGLE$prescription_id == presc_id),]
  
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

alert <- mcmapply(patient_alert, FULL_SINGLE$patient_id, FULL_SINGLE$prescription_id, mc.cores=40)

for (i in 1:6) {
  FULL_SINGLE[[paste0("alert", i)]] <- alert[i, ]
}

FULL_SINGLE$num_alert <- rowSums(FULL_SINGLE[, paste0("alert", 1:6)])
FULL_SINGLE$any_alert <- as.integer(FULL_SINGLE$num_alert > 0)

write.csv(FULL_SINGLE, paste("FULL_OPIOID_", year, "_ONE_TEMP.csv", sep=""), row.names = FALSE)
rm(alert)

########################################################################
### Determine LT and drop prescriptions afterwards

# - overlap with the previous (index based) prescription [different from the CURES overlap]
# - opioid days in 180 (different from the consecutive days)
setDT(FULL_SINGLE)
FULL_SINGLE[, overlap_lt := 0]
FULL_SINGLE[, opioid_days := days_supply]
FULL_SINGLE[, long_term := ifelse(opioid_days >= 90, 1, 0)]

FULL_SINGLE <- as.data.frame(FULL_SINGLE)

### Keep prescriptions up to long-term use, once it is long-term use, assume absorbing state
PATIENT <- FULL_SINGLE %>% 
  group_by(patient_id) %>% 
  summarize(first_presc_date = date_filled[1],
            longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
            longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA),
            first_longterm_presc = ifelse(sum(long_term) > 0, min(row_number()[long_term > 0]), NA),
            first_longterm_presc_id = ifelse(sum(long_term) > 0, prescription_id[long_term > 0][1], NA))

FULL_SINGLE <- left_join(FULL_SINGLE, PATIENT, by = "patient_id")

### Use presc_until of the long-term prescription to compute
# NA: patient never become long term
# >0: patient is going to become long term
# =0: patient is long term right after this prescription
# <0: patient is already long term
FULL_SINGLE <- FULL_SINGLE %>% 
  mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - as.Date(date_filled, format = "%m/%d/%Y"))) %>%
  filter(days_to_long_term > 0 | is.na(days_to_long_term)) %>% # either never long-term or haven't
  mutate(long_term_180 = ifelse(days_to_long_term <= 180, 1, 0)) %>% # within 180 days
  mutate(long_term_180 = ifelse(is.na(long_term_180), 0, long_term_180)) # never is also 0

FULL_SINGLE <- FULL_SINGLE %>% 
  select(-c(first_presc_date, longterm_filled_date, longterm_presc_date, 
            first_longterm_presc, first_longterm_presc_id))

write.csv(FULL_SINGLE, paste("FULL_OPIOID_", year, "_ONE_TEMP.csv", sep=""), row.names = FALSE)
rm(FULL_SINGLE)


########################################################################





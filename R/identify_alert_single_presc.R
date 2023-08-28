### Process the patients with one prescriptions only
library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2019

########################################################################

FULL_SINGLE <- read.csv(paste("../Data/FULL_OPIOID_", year, "_ONE.csv", sep=""))

FULL_SINGLE$prescription_id = seq.int(nrow(FULL_SINGLE))
FULL_SINGLE$presc_until <- as.Date(FULL_SINGLE$date_filled, format = "%m/%d/%Y") + FULL_SINGLE$days_supply
FULL_SINGLE$presc_until <- format(FULL_SINGLE$presc_until, "%m/%d/%Y")
FULL_SINGLE$age <- year(as.POSIXlt(FULL_SINGLE$date_filled, format="%m/%d/%Y")) - FULL_SINGLE$patient_birth_year

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

FULL_SINGLE <- FULL_SINGLE %>% mutate(concurrent_MME = daily_dose,
                                      concurrent_MME_methadone = ifelse(drug == 'Methadone', daily_dose, 0),
                                      num_prescribers = 1, num_pharmacies = 1,
                                      overlap = 0,
                                      consecutive_days = days_supply)

########################################################################
### Compute concurrent benzos
compute_concurrent_benzo <- function(pat_id, presc_id){
  PATIENT_PRESC_OPIOIDS <- FULL_SINGLE[which(FULL_SINGLE$prescription_id == presc_id),]
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

concurrent_benzo_prescriber <- mcmapply(compute_concurrent_benzo, FULL_SINGLE$patient_id, FULL_SINGLE$prescription_id, mc.cores=40)

FULL_SINGLE$concurrent_benzo <- concurrent_benzo_prescriber[1, ]
FULL_SINGLE$concurrent_benzo_same <- concurrent_benzo_prescriber[2, ]
FULL_SINGLE$concurrent_benzo_diff <- FULL_SINGLE$concurrent_benzo - FULL_SINGLE$concurrent_benzo_same

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
  if(PATIENT_PRESC_OPIOIDS$concurrent_MME_methadone >= 40){alert2 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_prescribers >= 6){alert3 = 1}
  if(PATIENT_PRESC_OPIOIDS$num_pharmacies >= 6){alert4 = 1}
  if(PATIENT_PRESC_OPIOIDS$consecutive_days >= 90){alert5 = 1}
  if(PATIENT_PRESC_OPIOIDS$concurrent_benzo > 0){alert6 = 1}
  
  return (c(alert1, alert2, alert3, alert4, alert5, alert6))
}

alert <- mcmapply(patient_alert, FULL_SINGLE$patient_id, FULL_SINGLE$prescription_id, mc.cores=40)

FULL_SINGLE$alert1 = alert[1, ]
FULL_SINGLE$alert2 = alert[2, ]
FULL_SINGLE$alert3 = alert[3, ]
FULL_SINGLE$alert4 = alert[4, ]
FULL_SINGLE$alert5 = alert[5, ]
FULL_SINGLE$alert6 = alert[6, ]

FULL_SINGLE$num_alert <- FULL_SINGLE$alert1 + FULL_SINGLE$alert2 + FULL_SINGLE$alert3 + FULL_SINGLE$alert4 + FULL_SINGLE$alert5 + FULL_SINGLE$alert6

########################################################################

FULL_2018_REORDER <- select(FULL_SINGLE, prescription_id, patient_id, patient_birth_year, age, patient_gender, patient_zip,
                            prescriber_id, prescriber_zip, pharmacy_id, pharmacy_zip, strength, quantity, days_supply,
                            date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
                            chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, MAINDOSE,
                            payment, prescription_month, prescription_year,
                            concurrent_MME, concurrent_MME_methadone, num_prescribers, num_pharmacies, 
                            concurrent_benzo, concurrent_benzo_same, concurrent_benzo_diff, 
                            overlap, consecutive_days,
                            alert1, alert2, alert3, alert4, alert5, alert6, num_alert)

write.csv(FULL_2018_REORDER, paste("../Data/FULL_ALERT_", year, "_SINGLE.csv", sep=""), row.names = FALSE)


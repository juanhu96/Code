### STEP 1
### Filter chronic/outlier/illicit prescriptions, export Opioid and Benzo prescriptions
### Split the data based on number of prescriptions

### INPUT: RX_2018.csv
### OUTPUT: FULL_OPIOID_2018_ATLEASTONE.csv, FULL_BENZO_2018.csv
### TEST <- FULL_OPIOID[1:20,]

library(dplyr)
library(lubridate)
library(data.table)
library(arules)
library(parallel)

setwd("/export/storage_cures/CURES/")
export_dir = "Processed/"
year = 2018
previous_year = year - 1

# Number of prescriptions 2017: 36182453, 2018: 33108449, 2019: 37175510
FULL_CURRENT <- read.csv(paste("RX_", year, ".csv", sep="")) 
FULL_PREVIOUS <- read.csv(paste("RX_", previous_year, ".csv", sep=""))

###################################
###### DROP CHRONIC, OUTLIERS
###################################

# Drop chronic users: 
# - those who filled an opioid prescription in the last 60 days of the prior year
# - 2019: 18204081 -> 8486712
chronic_patient_ids <- FULL_PREVIOUS %>%
  filter(class == 'Opioid') %>%
  mutate(prescription_month = month(as.POSIXlt(date_filled, format="%m/%d/%Y"))) %>%
  filter(prescription_month > 10) %>%
  select(patient_id) %>%
  distinct() %>%
  pull(patient_id)

setDT(FULL_CURRENT)

FULL_OPIOID <- FULL_CURRENT[class == 'Opioid'] 
FULL_OPIOID <- FULL_OPIOID[!patient_id %in% chronic_patient_ids] 


# Drop outliers:
# - prescriptions that exceed 1,000 daily MME
# - patients with more than 100 prior prescriptions)
# - 2019: 8486712 -> 8387944
OUTLIERS_PATIENT <- FULL_OPIOID[, .(num_presc = .N, max_dose = max(daily_dose)), by = patient_id][, outliers := as.integer(max_dose >= 1000 | num_presc >= 100)]
FULL_OPIOID <- merge(FULL_OPIOID, OUTLIERS_PATIENT, by = "patient_id", all.x = TRUE)
FULL_OPIOID[, outliers := ifelse(is.na(outliers), 0, outliers)]
FULL_OPIOID <- FULL_OPIOID[outliers == 0]


# Illicit users: 
# - filled prescriptions from three or more providers on their first date
# - 2019: 8387944 -> 8387430
setorder(FULL_OPIOID, patient_id, date_filled)

FULL_OPIOID[, date_filled := format(as.Date(date_filled, format="%m/%d/%Y"), "%m/%d/%Y")]
FULL_OPIOID[, prescription_month := month(as.Date(date_filled, format="%m/%d/%Y"))]
FULL_OPIOID[, prescription_year := year(as.Date(date_filled, format="%m/%d/%Y"))]
FULL_OPIOID[, presc_until := as.Date(date_filled, format="%m/%d/%Y") + days_supply]
FULL_OPIOID[, presc_until := format(presc_until, "%m/%d/%Y")]

FULL_OPIOID <- FULL_OPIOID[order(patient_id, date_filled)]
FIRST_PRESC_DATE <- FULL_OPIOID[, .(first_presc_date = min(date_filled)), by = patient_id]
FIRST_PRESC <- merge(FULL_OPIOID, FIRST_PRESC_DATE, by = "patient_id")
PRESCRIBERS_FIRST_PRESC_DATE <- FIRST_PRESC[date_filled == first_presc_date, .(unique_prescribers = uniqueN(prescriber_id)), by = patient_id]
ILLICIT_PATIENT <- PRESCRIBERS_FIRST_PRESC_DATE[unique_prescribers >= 3, patient_id] # 161 patients only
FULL_OPIOID <- FULL_OPIOID[!patient_id %in% ILLICIT_PATIENT]


# Underage patients: <18
# - 2019: 8387430 -> 8197802
FULL_OPIOID <- FULL_OPIOID[patient_birth_year <= year - 18]


# Duplicated rows, keep one each
# - 2019: 8197802 -> 6760290
duplicated_rows <- FULL_OPIOID[duplicated(FULL_OPIOID, by = setdiff(names(FULL_OPIOID), "X")) | 
                                 duplicated(FULL_OPIOID, by = setdiff(names(FULL_OPIOID), "X"), fromLast = TRUE)]
FULL_OPIOID <- FULL_OPIOID[!duplicated(FULL_OPIOID, by = setdiff(names(FULL_OPIOID), "X"))]


rm(chronic_patient_ids)
rm(duplicated_rows)
rm(OUTLIERS_PATIENT)
rm(FIRST_PRESC_DATE)
rm(FIRST_PRESC)
rm(PRESCRIBERS_FIRST_PRESC_DATE)
rm(ILLICIT_PATIENT)

###################################
### SPLIT DATA
###################################

PATIENT_NUM_PRESC <- FULL_OPIOID[, .(num_prescriptions = .N), by = patient_id]
FULL_NUM_PRESC <- merge(FULL_OPIOID, PATIENT_NUM_PRESC, by = "patient_id", all.x = TRUE)
FULL_NUM_PRESC_ONE <- FULL_NUM_PRESC[num_prescriptions == 1]
FULL_NUM_PRESC_ATLEASTTWO <- FULL_NUM_PRESC[num_prescriptions > 1]

# Filter based on patient_id_numeric ranges

# 2018
# FULL_NUM_PRESC_ATLEASTTWO_1 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id < 41846152]
# FULL_NUM_PRESC_ATLEASTTWO_2 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 41846152 & patient_id < 54195156]
# FULL_NUM_PRESC_ATLEASTTWO_3 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 54195156 & patient_id < 68042918]
# FULL_NUM_PRESC_ATLEASTTWO_4 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 68042918]

# 2019
FULL_NUM_PRESC_ATLEASTTWO_1 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id < 43247790]
FULL_NUM_PRESC_ATLEASTTWO_2 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 43247790 & patient_id < 59871298]
FULL_NUM_PRESC_ATLEASTTWO_3 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 59871298 & patient_id < 71429081]
FULL_NUM_PRESC_ATLEASTTWO_4 <- FULL_NUM_PRESC_ATLEASTTWO[patient_id >= 71429081]

write.csv(FULL_NUM_PRESC_ONE, paste(export_dir, "FULL_OPIOID_", year, "_ONE.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_1, paste(export_dir, "FULL_OPIOID_", year, "_ATLEASTTWO_1.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_2, paste(export_dir, "FULL_OPIOID_", year, "_ATLEASTTWO_2.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_3, paste(export_dir, "FULL_OPIOID_", year, "_ATLEASTTWO_3.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_4, paste(export_dir, "FULL_OPIOID_", year, "_ATLEASTTWO_4.csv", sep=""), row.names = FALSE)

rm(FULL_NUM_PRESC_ONE)
rm(FULL_NUM_PRESC_ATLEASTTWO_1)
rm(FULL_NUM_PRESC_ATLEASTTWO_2)
rm(FULL_NUM_PRESC_ATLEASTTWO_3)
rm(FULL_NUM_PRESC_ATLEASTTWO_4)

###################################
###### BENZO
###################################

FULL_BENZO <- FULL_CURRENT[class == 'Benzodiazepine']

setorder(FULL_BENZO, patient_id, date_filled)

FULL_BENZO[, date_filled := format(as.Date(date_filled, format="%m/%d/%Y"), "%m/%d/%Y")]
FULL_BENZO[, prescription_month := month(as.Date(date_filled, format="%m/%d/%Y"))]
FULL_BENZO[, prescription_year := year(as.Date(date_filled, format="%m/%d/%Y"))]
FULL_BENZO[, presc_until := as.Date(date_filled, format="%m/%d/%Y") + days_supply]
FULL_BENZO[, presc_until := format(presc_until, "%m/%d/%Y")]

write.csv(FULL_BENZO, paste(export_dir, "FULL_BENZO_", year, ".csv", sep=""), row.names = FALSE)


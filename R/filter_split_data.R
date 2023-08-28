### Filter chronic users, export Opioid and Benzo prescriptions
### Split the data based on number of prescriptions
library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2019
# 36182453 prescriptions in 2017
FULL <- read.csv(paste("../Data/RX_", year, ".csv", sep="")) 

CHRONIC_USERS <- read.csv("../Data/RX_2016.csv") #8692443
CHRONIC_USERS <- CHRONIC_USERS %>%
  mutate(prescription_month = month(as.POSIXlt(CHRONIC_USERS$date_filled, format="%m/%d/%Y"))) %>%
  group_by(patient_id) %>%
  summarize(chronic = ifelse(prescription_month[1]<4, 1, 0))

########################################################################
###### Opioid prescriptions for non-chronic users, split data ######
########################################################################

FULL_OPIOID <- FULL[which(FULL$class == 'Opioid'),]
FULL_OPIOID <- left_join(FULL_OPIOID, CHRONIC_USERS, by = "patient_id")
FULL_OPIOID <- FULL_OPIOID %>% mutate(chronic = coalesce(chronic, 0)) %>% filter(chronic == 0) #16397213

# Sort the rows by patient id and date
FULL_OPIOID <- FULL_OPIOID[order(FULL_OPIOID$patient_id, FULL_OPIOID$date_filled),]
FULL_OPIOID$prescription_month <- month(as.POSIXlt(FULL_OPIOID$date_filled, format="%m/%d/%Y"))
FULL_OPIOID$prescription_year <- year(as.POSIXlt(FULL_OPIOID$date_filled, format="%m/%d/%Y"))

# Number of prescriptions for each patient
PATIENT_NUM_PRESC <- FULL_OPIOID %>% group_by(patient_id) %>% summarize(num_prescriptions = n())
FULL_NUM_PRESC <- left_join(FULL_OPIOID, PATIENT_NUM_PRESC, by = "patient_id")

FULL_NUM_PRESC_ONE <- FULL_NUM_PRESC %>% filter(num_prescriptions == 1)
FULL_NUM_PRESC_ATLEASTTWO <- FULL_NUM_PRESC %>% filter(num_prescriptions > 1)

# Split the prescriptions into multiple fold
FULL_NUM_PRESC_ATLEASTTWO$patient_id_numeric <- as.numeric(FULL_NUM_PRESC_ATLEASTTWO$patient_id)
summary(FULL_NUM_PRESC_ATLEASTTWO$patient_id_numeric)

FULL_NUM_PRESC_ATLEASTTWO_1 <- FULL_NUM_PRESC_ATLEASTTWO %>% filter(patient_id_numeric < 41846152)
FULL_NUM_PRESC_ATLEASTTWO_2 <- FULL_NUM_PRESC_ATLEASTTWO %>% filter(patient_id_numeric >= 41846152 & patient_id_numeric < 54195156)
FULL_NUM_PRESC_ATLEASTTWO_3 <- FULL_NUM_PRESC_ATLEASTTWO %>% filter(patient_id_numeric >= 54195156 & patient_id_numeric < 68042918)
FULL_NUM_PRESC_ATLEASTTWO_4 <- FULL_NUM_PRESC_ATLEASTTWO %>% filter(patient_id_numeric >= 68042918)


write.csv(FULL_NUM_PRESC_ONE, paste("../Data/FULL_OPIOID_", year, "_ONE.csv", sep=""), row.names = FALSE)

write.csv(FULL_NUM_PRESC_ATLEASTTWO_1, paste("../Data/FULL_OPIOID_", year, "_ATLEASTTWO_1.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_2, paste("../Data/FULL_OPIOID_", year, "_ATLEASTTWO_2.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_3, paste("../Data/FULL_OPIOID_", year, "_ATLEASTTWO_3.csv", sep=""), row.names = FALSE)
write.csv(FULL_NUM_PRESC_ATLEASTTWO_4, paste("../Data/FULL_OPIOID_", year, "_ATLEASTTWO_4.csv", sep=""), row.names = FALSE)

########################################################################
###### Benzo prescriptions ######
########################################################################

FULL_BENZO <- FULL[FULL$class == 'Benzodiazepine',]

FULL_BENZO <- FULL_BENZO[order(FULL_BENZO$patient_id, FULL_BENZO$date_filled),]
FULL_BENZO$prescription_month <- month(as.POSIXlt(FULL_BENZO$date_filled, format="%m/%d/%Y"))
FULL_BENZO$prescription_year <- year(as.POSIXlt(FULL_BENZO$date_filled, format="%m/%d/%Y"))

FULL_BENZO$presc_until <- as.Date(FULL_BENZO$date_filled, format = "%m/%d/%Y") + FULL_BENZO$days_supply
FULL_BENZO$presc_until <- format(FULL_BENZO$presc_until, "%m/%d/%Y")

write.csv(FULL_BENZO, paste("../Data/FULL_BENZO_", year, ".csv", sep=""), row.names = FALSE)



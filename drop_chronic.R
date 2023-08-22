### Drop the chronic and non-opioid-naive patients based on previous year
### Filter illicit users

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
setwd("/mnt/phd/jihu/opioid/Code")

################################################################################
################################## Drop Chronic ################################
################################################################################

### 2019 based on 2018
FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv")
CHRONIC_2018 <- FULL_2018 %>% 
  mutate(prescription_month = month(as.POSIXlt(FULL_2018$date_filled, format="%m/%d/%Y"))) %>%
  filter(prescription_month == 11 | prescription_month == 12)

CHRONIC_PATIENT_2018 <- CHRONIC_2018 %>% group_by(patient_id) %>% summarize()
rm(CHRONIC_2018)

FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")
NEW_2019 <- FULL_2019[!(FULL_2019$patient_id %in% CHRONIC_PATIENT_2018$patient_id), ]
write.csv(NEW_2019, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)
rm(CHRONIC_PATIENT_2018)
rm(FULL_2019)
rm(NEW_2019)

# 2018 based on 2017
FULL_2017 <- read.csv("../Data/RX_2017.csv")
FULL_2017 <- FULL_2017[which(FULL_2017$class == 'Opioid'),]
# Patient who got prescription in the last two months of 2017
CHRONIC_2017 <- FULL_2017 %>% 
  mutate(prescription_month = month(as.POSIXlt(FULL_2017$date_filled, format="%m/%d/%Y"))) %>%
  filter(prescription_month == 11 | prescription_month == 12)

CHRONIC_PATIENT_2017 <- CHRONIC_2017 %>% group_by(patient_id) %>% summarize()
rm(FULL_2017)
rm(CHRONIC_2017)

# FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv")
NEW_2018 <- FULL_2018[!(FULL_2018$patient_id %in% CHRONIC_PATIENT_2017$patient_id), ]
write.csv(NEW_2018, "../Data/FULL_2018_LONGTERM_NEW.csv", row.names = FALSE)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

### Drop the ones who have prescription in the last 3 months of 2017
### Drop the illicit patients

################################################################################
# First by manually looking at those patient id
library(dplyr)
library(arules)

setwd("/mnt/phd/jihu/opioid/Code")
FULL_2018_FIRST <- read.csv(paste("../Data/FULL_2018_FIRST_ALLFEATURE.csv", sep=""))
TEST_FIRST <- FULL_2018_FIRST %>% select(c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                                           conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                           DEASCHEDULE, MAINDOSE, payment, overlap, presence_days, num_alert, long_term, num_prescribers, num_pharmacies))

## 86, given imbalance '1', this is still a lot
TEST_FIRST <- TEST_FIRST %>% filter(num_prescribers > 2 | num_pharmacies > 2) 
## 18, A lot of prescribers are OK, but a lot of pharmacies are weird (but maybe ok if from the same prescriber?)
TEST_FIRST <- TEST_FIRST %>% filter(num_pharmacies > 2) 
## 69, a lot of prescribers maybe be doctor-shoping
TEST_FIRST <- TEST_FIRST %>% filter(num_prescribers > 2) 

################################################################################
FULL_2017_ATLEASTTWO <- read.csv(paste("../Data/FULL_OPIOID_2017_ATLEASTTWO.csv", sep=""))
FULL_2017_ONE <- read.csv(paste("../Data/FULL_OPIOID_2017_ONE.csv", sep=""))

# 60699906 have 3 prescribers & 3 pharmacies, not in 2017
# 69236349 have 1 prescribers & 4 pharmacies, not in 2017
# 15999989 have 4 prescribers & 1 pharmacies, 1 in 02/08/2017
# 68581250 have 4 prescribers & 1 pharmacies, not in 2017
test1 <- FULL_2017_ONE %>% filter(patient_id == 68581250)
test2 <- FULL_2017_ATLEASTTWO %>% filter(patient_id == 68581250)

################################################################################
FULL_2018 <- read.csv(paste("../Data/FULL_2018_ALERT.csv", sep=""))
TEST <- FULL_2018 %>% select(c(patient_id, patient_birth_year, prescriber_id, pharmacy_id, strength, days_supply, date_filled, 
                               drug, daily_dose, payment, num_prescribers, num_pharmacies))

# look at all their prescription
test1 <- TEST %>% filter(patient_id == 60699906)
test2 <- TEST %>% filter(patient_id == 69236349)
test3 <- TEST %>% filter(patient_id == 15999989)
test4 <- TEST %>% filter(patient_id == 68581250)

################################################################################
############################### FILTER ILLCIT ##################################
################################################################################

library(dplyr)
library(arules)

setwd("/mnt/phd/jihu/opioid/Code")
FULL_2018_FIRST <- read.csv(paste("../Data/FULL_2018_FIRST_ALLFEATURE.csv", sep=""))
TEST_FIRST <- FULL_2018_FIRST %>% select(c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                                           conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                           DEASCHEDULE, MAINDOSE, payment, overlap, presence_days, num_alert, long_term, num_prescribers, num_pharmacies))
rm(FULL_2018_FIRST)

## 86, given imbalance '1', this is still a lot
TEST_FIRST <- TEST_FIRST %>% filter(num_prescribers > 2 | num_pharmacies > 2) %>%
  select(c(patient_id, patient_birth_year, date_filled, payment, num_prescribers, num_pharmacies))
Illicit_patient <- TEST_FIRST$patient_id

## Throw away these patient's prescriptions
FULL_2018 <- read.csv(paste("../Data/FULL_2018_ALERT.csv", sep=""))
TEST <- FULL_2018 %>% filter(!patient_id %in% Illicit_patient) # About 700 prescriptions from 100 patient are filtered
write.csv(TEST, "../Data/FULL_2018_ALERT_NOILLICIT.csv", row.names = FALSE)


## Do the same for rolling basis
FULL_2018 <- read.csv(paste("../Data/FULL_2018_ROLLING_ALLFEATURE.csv", sep="")) # 6,103,698
TEST <- FULL_2018 %>% filter(!patient_id %in% Illicit_patient) # 6,103,110
write.csv(TEST, "../Data/FULL_2018_ROLLING_ALLFEATURE_NOILLICIT.csv", row.names = FALSE)

################################################################################

### filter elicit for 2019 as well
FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")

PATIENT <- FULL_2019 %>% group_by(patient_id) %>% summarize(num_prescribers = num_prescribers[1],
                                                            num_pharmacies = num_pharmacies[1]) %>%
  filter(num_prescribers > 2 | num_pharmacies > 2)

Illicit_patient <- PATIENT$patient_id
TEST <- FULL_2019 %>% filter(!patient_id %in% Illicit_patient) 
write.csv(TEST, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)










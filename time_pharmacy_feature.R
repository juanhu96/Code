### Prescriptions/time to first alert for every patient
### Compute time-series & pharmacy-based features
### Plot histogram and breakdown by type

library(dplyr)
library(lubridate)
library(arules)
library(parallel)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2018
T_period31 = 31
T_period62 = 62
T_period93 = 93

FULL_ALL <- read.csv(paste("../Data/FULL_2018_ALERT_NOILLICIT.csv", sep=""))

################################################################################
################################################################################
################################################################################
### Prescription/time to first alert
PATIENT <- FULL_ALL %>% group_by(patient_id) %>% summarize(first_presc_date = date_filled[1],
                                                           first_alert_date = ifelse(sum(num_alert) > 0, date_filled[num_alert > 0][1], NA),
                                                           first_alert_presc = ifelse(sum(num_alert) > 0, min(row_number()[num_alert > 0]), NA),
                                                           first_alert_presc_id = ifelse(sum(num_alert) > 0, prescription_id[num_alert > 0][1], NA))

## Focus on patient who eventually received an alert
## 44099867 patient, 869795 of them ever received an alert
## 619327 of them got the alert on their first prescription
## 619512 of them got the alert on their first date
PATIENT <- PATIENT %>% filter(!is.na(first_alert_date))
PATIENT <- PATIENT %>% mutate(time_to_first_alert = as.numeric(as.Date(first_alert_date, format = "%m/%d/%Y") - as.Date(first_presc_date, format = "%m/%d/%Y")))
PATIENT_FIRST <- PATIENT %>% filter(first_alert_presc == 1)
PATIENT_FIRST_DATE <- PATIENT %>% filter(time_to_first_alert == 0)

# This patient have 513 prescriptions in 2018
# One prescription per day, but also gap every 40~50 days so that the alert is not triggered
# Until about 300 prescriptions
# Very old, 82 years old so I guess it's fine
MAX_PATIENT <- FULL_ALL %>% filter(patient_id == 60606292)

################################################################################
## Visualization: prescriptions to first alert
ggplot(PATIENT, aes(x=ifelse(first_alert_presc > 10, 10, first_alert_presc))) + 
  geom_bar() +
  ggtitle("Number of prescriptions until first alert") +
  xlab("Number of prescriptions") +
  ylab("Count") +
  xlim(0, 11) + 
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/num_presc_first_alert.pdf", bg="white", width=8, height=6, dpi=300)

ggplot(PATIENT %>% filter(first_alert_presc > 1), aes(x=ifelse(first_alert_presc > 10, 10, first_alert_presc))) + 
  geom_bar() +
  ggtitle("Number of prescriptions until first alert \n (Patient with alert on first prescription dropped)") +
  xlab("Number of prescriptions") +
  ylab("Count") +
  xlim(1, 10) + 
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/num_presc_first_alert(exclude_first).pdf", bg="white", width=8, height=6, dpi=300)

################################################################################
## Visualization: time to first alert
ggplot(PATIENT, aes(x=ifelse(time_to_first_alert > 90, 90, time_to_first_alert))) + 
  geom_histogram() +
  ggtitle("Days until first alert") +
  xlab("Days") +
  ylab("Count") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/days_first_alert.pdf", bg="white", width=8, height=6, dpi=300)

ggplot(PATIENT %>% filter(first_alert_presc > 1), aes(x=ifelse(time_to_first_alert > 300, 300, time_to_first_alert))) + 
  geom_density(adjust = 1) +
  ggtitle("Days until first alert \n (Patient with alert on first prescription dropped)") +
  xlab("Days") +
  ylab("Count") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/days_first_alert(exclude_first).pdf", bg="white", width=8, height=6, dpi=300)

################################################################################
########################### TIME-BASED FEATURES ################################
################################################################################

### Up to first alert, no illicit patients
FULL_ALL <- read.csv(paste("../Data/FULL_2018_ROLLING_ALLFEATURE_NOILLICIT.csv", sep=""))
# TEST <- FULL_ALL[1:400,]

## Number of prescription (xth prescription)
compute_num_presc <- function(pat_id, presc_id){
  # Bounday case: a patient have multiple prescriptions on the same day
  # Do we want number of prescription (day before) the current prescription
  # Or all prescription (even on same day) before the current prescription
  PATIENT <- FULL_ALL[which(FULL_ALL$patient_id == pat_id),]
  presc_index <- which(PATIENT$prescription_id == presc_id)
  return (presc_index)
}
FULL_ALL$num_presc <- mcmapply(compute_num_presc, FULL_ALL$patient_id, FULL_ALL$prescription_id, mc.cores=40)

## Increase in dosage (for single prescription, or concurrrent MME)
FULL_ALL <- FULL_ALL %>% mutate(previous_dose = ifelse(patient_id == lag(patient_id), lag(daily_dose), 0),
                                previous_concurrent_MME = ifelse(patient_id == lag(patient_id), lag(concurrent_MME), 0),
                                previous_days = ifelse(patient_id == lag(patient_id), lag(days_supply), 0),
                                dose_diff = ifelse(patient_id == lag(patient_id), daily_dose - previous_dose, 0),
                                concurrent_MME_diff = ifelse(patient_id == lag(patient_id), concurrent_MME - previous_concurrent_MME, 0),
                                days_diff = ifelse(patient_id == lag(patient_id), days_supply - previous_days, 0))

FULL_ALL[1,]$previous_dose = 0
FULL_ALL[1,]$previous_concurrent_MME = 0
FULL_ALL[1,]$previous_days = 0
FULL_ALL[1,]$dose_diff = 0
FULL_ALL[1,]$concurrent_MME_diff = 0
FULL_ALL[1,]$days_diff = 0

## Switch in drug type/payment
FULL_ALL <- FULL_ALL %>% mutate(switch_drug = ifelse(patient_id == lag(patient_id) & drug == lag(drug), 1, 0),
                                switch_payment = ifelse(patient_id == lag(patient_id) & payment == lag(payment), 1, 0))
FULL_ALL[1,]$switch_drug = 0
FULL_ALL[1,]$switch_payment = 0

write.csv(FULL_ALL, "../Data/FULL_2018_ROLLING_ALLFEATURE_NOILLICIT.csv", row.names = FALSE)

FULL <- FULL_ALL %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                               prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
                               conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                               DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_quarter, prescription_year, 
                               presence_MME, presence_MME_methadone, presence_num_prescribers, presence_num_pharmacies, 
                               overlap, presence_days, num_alert, long_term, previous_dose, previous_concurrent_MME, previous_days))

FULL <- rename(FULL, concurrent_methadone_MME = concurrent_MME_methadone)
FULL <- rename(FULL, MME_diff = concurrent_MME_diff)
write.csv(FULL, paste("../Data/FULL_", year, "_INPUT_ROLLING_ALLFEATURE_NOILLICIT_TIMESERIES.csv", sep=""), row.names = FALSE)

################################################################################
################################################################################
################################################################################
### Focus on patient who got alert on their first prescription, breakdown by type

## If prescription_id from FULL_ALL is in PATIENT_FIRST's first_alert_presc_id
## 619327 prescriptions from 619327 patient
PRESC_FIRST <- FULL_ALL[FULL_ALL$prescription_id %in% PATIENT_FIRST_DATE$first_alert_presc_id, ]

## Histogram by alert type
ALERT1 <- PRESC_FIRST %>% filter(alert1 == 1) # 294391
ALERT2 <- PRESC_FIRST %>% filter(alert2 == 1) # 13139
ALERT3 <- PRESC_FIRST %>% filter(alert3 == 1) # 0
ALERT4 <- PRESC_FIRST %>% filter(alert4 == 1) # 0
ALERT5 <- PRESC_FIRST %>% filter(alert5 == 1) # 5755
ALERT6 <- PRESC_FIRST %>% filter(alert6 == 1) # 404830

## Look into those who got alert 5 on first day
# patient id: 76784, 83633, 102331
patient1 <- FULL_ALL %>% filter(patient_id == 76784) # got a prescription with days_supply = 90
patient2 <- FULL_ALL %>% filter(patient_id == 83633)
patient3 <- FULL_ALL %>% filter(patient_id == 102331)





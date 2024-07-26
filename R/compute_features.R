### STEP 4
### Feature engineering (prescription-based)

### INPUT: FULL_OPIOID_2018_ONE_TEMP.csv, FULL_OPIOID_2018_ATLEASTTWO_TEMP.csv
### OUTPUT: FULL_OPIOID_2018_ONE_FEATURE.csv, FULL_OPIOID_2018_ATLEASTTWO_FEATURE.csv

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2019

################################################################################
### SINGLE FEATURES: CURES features/alerts, LT use, up to first LT
################################################################################

FULL <- read.csv(paste("FULL_OPIOID_", year, "_ONE_TEMP.csv", sep=""))
FULL <- FULL %>% 
  mutate(patient_zip = as.character(patient_zip)) %>% 
  rename(num_prescribers_past180 = num_prescribers, num_pharmacies_past180 = num_pharmacies) %>% 
  arrange(patient_id, date_filled, presc_until) %>% 
  select(-c(X, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, num_presc)) # unused features

OUTLIER_PRESC <- FULL %>% filter(quantity >= 1000 | concurrent_MME >= 1000 | concurrent_methadone_MME >= 1000 |
                                   num_prescribers_past180 > 10 | num_pharmacies_past180 > 10 | num_prescriptions >= 100 | age >= 100)
OUTLIER_PATIENT <- unique(OUTLIER_PRESC$patient_id) 
FULL <- FULL %>% filter(!patient_id %in% OUTLIER_PATIENT)

rm(OUTLIER_PRESC)
rm(OUTLIER_PATIENT)

setDT(FULL)
setorder(FULL, patient_id, date_filled, presc_until)

FULL[, prescription_id := seq_len(.N)]
FULL[, num_prior_prescriptions := 0]
FULL[, num_prior_prescriptions_past180 := 0]
FULL[, num_prior_prescriptions_past90 := 0]
FULL[, num_prior_prescriptions_past30 := 0]

FULL[, switch_drug := 0]
FULL[, switch_payment := 0]
FULL[, ever_switch_drug := 0]
FULL[, ever_switch_payment := 0]

FULL[, dose_diff := 0]
FULL[, concurrent_MME_diff := 0]
FULL[, quantity_diff := 0]
FULL[, days_diff := 0]

FULL[, avgMME_past180 := daily_dose]
FULL[, avgDays_past180 := days_supply]
FULL[, avgMME_past90 := daily_dose]
FULL[, avgDays_past90 := days_supply]
FULL[, avgMME_past30 := daily_dose]
FULL[, avgDays_past30 := days_supply]

FULL[, HMFO := as.integer(drug %in% c("Hydromorphone", "Methadone", "Fentanyl", "Oxymorphone"))]
FULL[, c("Codeine_MME", "Hydrocodone_MME", "Oxycodone_MME", "Morphine_MME", 
         "Hydromorphone_MME", "Methadone_MME", "Fentanyl_MME", "Oxymorphone_MME") :=
       .(ifelse(drug == "Codeine", daily_dose, 0),
         ifelse(drug == "Hydrocodone", daily_dose, 0),
         ifelse(drug == "Oxycodone", daily_dose, 0),
         ifelse(drug == "Morphine", daily_dose, 0),
         ifelse(drug == "Hydromorphone", daily_dose, 0),
         ifelse(drug == "Methadone", daily_dose, 0),
         ifelse(drug == "Fentanyl", daily_dose, 0),
         ifelse(drug == "Oxymorphone", daily_dose, 0))]

FULL[, gap := 360]
FULL <- as.data.frame(FULL)
# TEST <- FULL[1:30,]

# compute_num_prior_benzo defined below
BENZO_TABLE <- read.csv(paste("FULL_BENZO_", year, ".csv", sep=""))
num_prior_prescriptions_benzo <- mcmapply(compute_num_prior_benzo, FULL$patient_id, FULL$prescription_id, mc.cores=50)
FULL$num_prior_prescriptions_benzo_past180 = num_prior_prescriptions_benzo[1, ]
FULL$num_prior_prescriptions_benzo_past90 = num_prior_prescriptions_benzo[2, ]
FULL$num_prior_prescriptions_benzo_past30 = num_prior_prescriptions_benzo[3, ]

write.csv(FULL, paste("FULL_OPIOID_", year ,"_ONE_FEATURE.csv", sep=""), row.names = FALSE)

rm(num_prior_prescriptions_benzo)

################################################################################
### MULTIPLE FEATURES: CURES features/alerts, LT use, up to first LT
################################################################################

batch = 3
FULL <- read.csv(paste("FULL_OPIOID_", year, "_ATLEASTTWO_", batch ,"_TEMP.csv", sep=""))
FULL <- as.data.frame(FULL)
FULL <- FULL %>% mutate(patient_zip = as.character(patient_zip)) %>%
  arrange(patient_id, date_filled, presc_until) %>% 
  select(-c(X, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME, DEASCHEDULE, num_presc))

# drop more outliers (based on CURES features)
# e.g. patient 51282 has 30 prescriptions (all Fentanyl & CommercialIns), 7 up to LT
OUTLIER_PRESC <- FULL %>% filter(quantity >= 1000 | concurrent_MME >= 1000 | concurrent_methadone_MME >= 1000 |
                                   num_prescribers_past180 > 10 | num_pharmacies_past180 > 10 | num_prescriptions >= 100 | age >= 100)
OUTLIER_PATIENT <- unique(OUTLIER_PRESC$patient_id) 
FULL <- FULL %>% filter(!patient_id %in% OUTLIER_PATIENT)

rm(OUTLIER_PRESC)
rm(OUTLIER_PATIENT)

################################################################################
### PRESCRIPTION-BASED FEATURES
################################################################################

### NUMBER OF PRIOR PRESCRIPTIONS (OVERALL AND IN PAST 180)
FULL <- FULL %>% 
  group_by(patient_id) %>%
  mutate(num_prior_prescriptions = row_number() - 1)

# TEST <- FULL[1:40,]

compute_num_prior_presc <- function(pat_id, presc_id) {

  PATIENT <- FULL[which(FULL$patient_id == pat_id), ]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id), ]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  count_prior_prescriptions <- function(patient_data, presc_date, presc_id, days_window) {
    start_date <- as.Date(presc_date, format = "%m/%d/%Y") - days(days_window)
    end_date <- as.Date(presc_date, format = "%m/%d/%Y")
    
    PATIENT_PREV_PRESC <- patient_data %>%
      filter(as.Date(date_filled, format = "%m/%d/%Y") >= start_date &
               as.Date(date_filled, format = "%m/%d/%Y") <= end_date &
               prescription_id < presc_id)
    
    return(nrow(PATIENT_PREV_PRESC))
  }
  
  num_prescriptions_180 <- count_prior_prescriptions(PATIENT, presc_date, presc_id, 180)
  num_prescriptions_90 <- count_prior_prescriptions(PATIENT, presc_date, presc_id, 90)
  num_prescriptions_30 <- count_prior_prescriptions(PATIENT, presc_date, presc_id, 30)
  
  return(c(num_prescriptions_180, num_prescriptions_90, num_prescriptions_30))
}

num_prior_prescriptions <- mcmapply(compute_num_prior_presc, FULL$patient_id, FULL$prescription_id, mc.cores=50)
FULL$num_prior_prescriptions_past180 = num_prior_prescriptions[1, ]
FULL$num_prior_prescriptions_past90 = num_prior_prescriptions[2, ]
FULL$num_prior_prescriptions_past30 = num_prior_prescriptions[3, ]
write.csv(FULL, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", batch ,"_FEATURE.csv", sep=""), row.names = FALSE)

rm(num_prior_prescriptions)


### SWITCH IN DRUG/PAYMENT
FULL <- as.data.frame(FULL) # avoid weird error: DLL requires the use of native symbols
FULL <- FULL %>% 
  ungroup() %>% 
  mutate(switch_drug = ifelse(patient_id == lag(patient_id) & drug != lag(drug), 1, 0),
         switch_payment = ifelse(patient_id == lag(patient_id) & payment != lag(payment), 1, 0))
FULL[1,]$switch_drug = 0
FULL[1,]$switch_payment = 0


### EVER SWITCH
PATIENT <- FULL %>% 
  group_by(patient_id) %>% 
  summarize(first_switch_drug = ifelse(sum(switch_drug) > 0, date_filled[switch_drug > 0][1], "01/01/2021"),
            first_switch_payment = ifelse(sum(switch_payment) > 0, date_filled[switch_payment > 0][1], "01/01/2021"))
FULL <- left_join(FULL, PATIENT, by = 'patient_id')
FULL <- FULL %>% 
  mutate(ever_switch_drug = ifelse(as.Date(date_filled, format = "%m/%d/%Y") >= as.Date(first_switch_drug, format = "%m/%d/%Y"), 1, 0),
         ever_switch_payment = ifelse(as.Date(date_filled, format = "%m/%d/%Y") >= as.Date(first_switch_payment, format = "%m/%d/%Y"), 1, 0)) %>%
  select(-c(first_switch_drug, first_switch_payment))


### CHANGE IN DOSAGE/MME/QUANTITY/DAYS SUPPLY 
# (TRICKY HERE, AS DIFFERENT TYPE OF DRUGS HAVE DIFFERENT MME)
FULL <- FULL %>% mutate(previous_dose = ifelse(patient_id == lag(patient_id), lag(daily_dose), 0),
                        previous_concurrent_MME = ifelse(patient_id == lag(patient_id), lag(concurrent_MME), 0),
                        previous_quantity = ifelse(patient_id == lag(patient_id), lag(quantity), 0),
                        previous_days = ifelse(patient_id == lag(patient_id), lag(days_supply), 0),
                        dose_diff = ifelse(patient_id == lag(patient_id), daily_dose - previous_dose, 0),
                        concurrent_MME_diff = ifelse(patient_id == lag(patient_id), concurrent_MME - previous_concurrent_MME, 0),
                        quantity_diff =  ifelse(patient_id == lag(patient_id), quantity - previous_quantity, 0),
                        days_diff = ifelse(patient_id == lag(patient_id), days_supply - previous_days, 0))

FULL[1,]$previous_dose = 0
FULL[1,]$previous_concurrent_MME = 0
FULL[1,]$previous_quantity = 0
FULL[1,]$previous_days = 0
FULL[1,]$dose_diff = 0
FULL[1,]$concurrent_MME_diff = 0
FULL[1,]$quantity_diff = 0
FULL[1,]$days_diff = 0

FULL <- FULL %>%
  select(-c(previous_dose, previous_concurrent_MME, previous_quantity, previous_days))


### AVG MME/DAYS IN PAST 180/90/30 DAYS
compute_avg <- function(pat_id, presc_id) {

  PATIENT <- FULL[which(FULL$patient_id == pat_id), ]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id), ]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  calculate_averages <- function(patient_data, presc_date, presc_id, days_window) {
    PATIENT_PREV_PRESC <- patient_data %>%
      filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(presc_date, format = "%m/%d/%Y"),
             as.Date(date_filled, format = "%m/%d/%Y") >= as.Date(presc_date, format = "%m/%d/%Y") - days(days_window),
             prescription_id <= presc_id)
    
    avg_dailyMME <- mean(PATIENT_PREV_PRESC$daily_dose, na.rm = TRUE)
    avg_days <- mean(PATIENT_PREV_PRESC$days_supply, na.rm = TRUE)
    
    return(c(avg_dailyMME, avg_days))
  }
  
  # Calculate averages for different time windows
  avg <- list(
    calculate_averages(PATIENT, presc_date, presc_id, 180),
    calculate_averages(PATIENT, presc_date, presc_id, 90),
    calculate_averages(PATIENT, presc_date, presc_id, 30)
  )
  
  # Flatten the list into a vector and return
  return(unlist(avg))
}

avg <- mcmapply(compute_avg, FULL$patient_id, FULL$prescription_id, mc.cores=50)
FULL$avgMME_past180 = avg[1, ]
FULL$avgDays_past180 = avg[2, ]
FULL$avgMME_past90 = avg[3, ]
FULL$avgDays_past90 = avg[4, ]
FULL$avgMME_past30 = avg[5, ]
FULL$avgDays_past30 = avg[6, ]

# TEST <- FULL[1:40,]
write.csv(FULL, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", batch ,"_FEATURE.csv", sep=""), row.names = FALSE)

rm(avg)


### HMFO
FULL <- FULL %>% mutate(HMFO = ifelse(drug == "Hydromorphone" | drug == "Methadone" | drug == "Fentanyl" | drug == "Oxymorphone", 1, 0))


### INTERACTION: DRUG X MME
FULL <- FULL %>% mutate(Codeine_MME = ifelse(drug == "Codeine", daily_dose, 0),
                        Hydrocodone_MME = ifelse(drug == "Hydrocodone", daily_dose, 0),
                        Oxycodone_MME = ifelse(drug == "Oxycodone", daily_dose, 0),
                        Morphine_MME = ifelse(drug == "Morphine", daily_dose, 0),
                        Hydromorphone_MME = ifelse(drug == "Hydromorphone", daily_dose, 0),
                        Methadone_MME = ifelse(drug == "Methadone", daily_dose, 0),
                        Fentanyl_MME = ifelse(drug == "Fentanyl", daily_dose, 0),
                        Oxymorphone_MME = ifelse(drug == "Oxymorphone", daily_dose, 0))

### INTERACTION: DRUG X PAYMENT
# FULL <- FULL %>% mutate(drug_payment = paste(drug, payment, sep = "_")) %>%
#   mutate(Codeine_Medicaid = ifelse(drug_payment == "Codeine_Medicaid", 1, 0),
#          Codeine_CommercialIns = ifelse(drug_payment == "Codeine_CommercialIns", 1, 0),
#          Codeine_Medicare = ifelse(drug_payment == "Codeine_Medicare", 1, 0),
#          Codeine_CashCredit = ifelse(drug_payment == "Codeine_CashCredit", 1, 0),
#          Codeine_MilitaryIns = ifelse(drug_payment == "Codeine_MilitaryIns", 1, 0),
#          Codeine_WorkersComp = ifelse(drug_payment == "Codeine_WorkersComp", 1, 0),
#          Codeine_Other = ifelse(drug_payment == "Codeine_Other", 1, 0),
#          Codeine_IndianNation = ifelse(drug_payment == "Codeine_IndianNation", 1, 0),
#          Hydrocodone_Medicaid = ifelse(drug_payment == "Hydrocodone_Medicaid", 1, 0),
#          Hydrocodone_CommercialIns = ifelse(drug_payment == "Hydrocodone_CommercialIns", 1, 0),
#          Hydrocodone_Medicare = ifelse(drug_payment == "Hydrocodone_Medicare", 1, 0),
#          Hydrocodone_CashCredit = ifelse(drug_payment == "Hydrocodone_CashCredit", 1, 0),
#          Hydrocodone_MilitaryIns = ifelse(drug_payment == "Hydrocodone_MilitaryIns", 1, 0),
#          Hydrocodone_WorkersComp = ifelse(drug_payment == "Hydrocodone_WorkersComp", 1, 0),
#          Hydrocodone_Other = ifelse(drug_payment == "Hydrocodone_Other", 1, 0),
#          Hydrocodone_IndianNation = ifelse(drug_payment == "Hydrocodone_IndianNation", 1, 0),
#          Oxycodone_Medicaid = ifelse(drug_payment == "Oxycodone_Medicaid", 1, 0),
#          Oxycodone_CommercialIns = ifelse(drug_payment == "Oxycodone_CommercialIns", 1, 0),
#          Oxycodone_Medicare = ifelse(drug_payment == "Oxycodone_Medicare", 1, 0),
#          Oxycodone_CashCredit = ifelse(drug_payment == "Oxycodone_CashCredit", 1, 0),
#          Oxycodone_MilitaryIns = ifelse(drug_payment == "Oxycodone_MilitaryIns", 1, 0),
#          Oxycodone_WorkersComp = ifelse(drug_payment == "Oxycodone_WorkersComp", 1, 0),
#          Oxycodone_Other = ifelse(drug_payment == "Oxycodone_Other", 1, 0),
#          Oxycodone_IndianNation = ifelse(drug_payment == "Oxycodone_IndianNation", 1, 0),
#          Morphine_Medicaid = ifelse(drug_payment == "Morphine_Medicaid", 1, 0),
#          Morphine_CommercialIns = ifelse(drug_payment == "Morphine_CommercialIns", 1, 0),
#          Morphine_Medicare = ifelse(drug_payment == "Morphine_Medicare", 1, 0),
#          Morphine_CashCredit = ifelse(drug_payment == "Morphine_CashCredit", 1, 0),
#          Morphine_MilitaryIns = ifelse(drug_payment == "Morphine_MilitaryIns", 1, 0),
#          Morphine_WorkersComp = ifelse(drug_payment == "Morphine_WorkersComp", 1, 0),
#          Morphine_Other = ifelse(drug_payment == "Morphine_Other", 1, 0),
#          Morphine_IndianNation = ifelse(drug_payment == "Morphine_IndianNation", 1, 0),
#          Hydromorphone_Medicaid = ifelse(drug_payment == "Hydromorphone_Medicaid", 1, 0),
#          Hydromorphone_CommercialIns = ifelse(drug_payment == "Hydromorphone_CommercialIns", 1, 0),
#          Hydromorphone_Medicare = ifelse(drug_payment == "Hydromorphone_Medicare", 1, 0),
#          Hydromorphone_CashCredit = ifelse(drug_payment == "Hydromorphone_CashCredit", 1, 0),
#          Hydromorphone_MilitaryIns = ifelse(drug_payment == "Hydromorphone_MilitaryIns", 1, 0),
#          Hydromorphone_WorkersComp = ifelse(drug_payment == "Hydromorphone_WorkersComp", 1, 0),
#          Hydromorphone_Other = ifelse(drug_payment == "Hydromorphone_Other", 1, 0),
#          Hydromorphone_IndianNation = ifelse(drug_payment == "Hydromorphone_IndianNation", 1, 0),
#          Methadone_Medicaid = ifelse(drug_payment == "Methadone_Medicaid", 1, 0),
#          Methadone_CommercialIns = ifelse(drug_payment == "Methadone_CommercialIns", 1, 0),
#          Methadone_Medicare = ifelse(drug_payment == "Methadone_Medicare", 1, 0),
#          Methadone_CashCredit = ifelse(drug_payment == "Methadone_CashCredit", 1, 0),
#          Methadone_MilitaryIns = ifelse(drug_payment == "Methadone_MilitaryIns", 1, 0),
#          Methadone_WorkersComp = ifelse(drug_payment == "Methadone_WorkersComp", 1, 0),
#          Methadone_Other = ifelse(drug_payment == "Methadone_Other", 1, 0),
#          Methadone_IndianNation = ifelse(drug_payment == "Methadone_IndianNation", 1, 0),
#          Fentanyl_Medicaid = ifelse(drug_payment == "Fentanyl_Medicaid", 1, 0),
#          Fentanyl_CommercialIns = ifelse(drug_payment == "Fentanyl_CommercialIns", 1, 0),
#          Fentanyl_Medicare = ifelse(drug_payment == "Fentanyl_Medicare", 1, 0),
#          Fentanyl_CashCredit = ifelse(drug_payment == "Fentanyl_CashCredit", 1, 0),
#          Fentanyl_MilitaryIns = ifelse(drug_payment == "Fentanyl_MilitaryIns", 1, 0),
#          Fentanyl_WorkersComp = ifelse(drug_payment == "Fentanyl_WorkersComp", 1, 0),
#          Fentanyl_Other = ifelse(drug_payment == "Fentanyl_Other", 1, 0),
#          Fentanyl_IndianNation = ifelse(drug_payment == "Fentanyl_IndianNation", 1, 0),
#          Oxymorphone_Medicaid = ifelse(drug_payment == "Oxymorphone_Medicaid", 1, 0),
#          Oxymorphone_CommercialIns = ifelse(drug_payment == "Oxymorphone_CommercialIns", 1, 0),
#          Oxymorphone_Medicare = ifelse(drug_payment == "Oxymorphone_Medicare", 1, 0),
#          Oxymorphone_CashCredit = ifelse(drug_payment == "Oxymorphone_CashCredit", 1, 0),
#          Oxymorphone_MilitaryIns = ifelse(drug_payment == "Oxymorphone_MilitaryIns", 1, 0),
#          Oxymorphone_WorkersComp = ifelse(drug_payment == "Oxymorphone_WorkersComp", 1, 0),
#          Oxymorphone_Other = ifelse(drug_payment == "Oxymorphone_Other", 1, 0),
#          Oxymorphone_IndianNation = ifelse(drug_payment == "Oxymorphone_IndianNation", 1, 0))


### GAP BETWEEN REFILLS
# ANY OVERLAP PRESCRIPTIONS ARE CONSIDERED AS 0
# GAP FOR FIRST PRESCRIPTIONS AS 360
FULL <- FULL %>%
  group_by(patient_id) %>%
  mutate(gap = ifelse(overlap > 0, 0,
                      ifelse(is.na(lag(patient_id)), 360, 
                             as.Date(date_filled, format = "%m/%d/%Y") - lag(as.Date(presc_until, format = "%m/%d/%Y"), default = first(as.Date(date_filled, format = "%m/%d/%Y"))))))


### PRIOR PRESCRIPTIONS OF BENZO, PAST 180/90/30
BENZO_TABLE <- read.csv(paste("FULL_BENZO_", year, ".csv", sep=""))

compute_num_prior_benzo <- function(pat_id, presc_id){
  
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  PATIENT_PRESC_BENZOS <- BENZO_TABLE[which(BENZO_TABLE$patient_id == pat_id), ]
  
  if (nrow(PATIENT_PRESC_BENZOS) == 0){
    return (c(0, 0, 0))
  } else{

    count_prior_prescriptions <- function(patient_data, presc_date, days_window) {
      
      start_date <- as.Date(presc_date, format = "%m/%d/%Y") - days(days_window)
      end_date <- as.Date(presc_date, format = "%m/%d/%Y")
      
      PATIENT_PRIOR_PRESC_BENZOS <- patient_data %>%
        filter(as.Date(date_filled, format = "%m/%d/%Y") >= start_date &
                 as.Date(date_filled, format = "%m/%d/%Y") <= end_date)
      
      return(nrow(PATIENT_PRIOR_PRESC_BENZOS))
    }
    
    num_benzo_prescriptions_180 <- count_prior_prescriptions(PATIENT_PRESC_BENZOS, presc_date, 180)
    num_benzo_prescriptions_90 <- count_prior_prescriptions(PATIENT_PRESC_BENZOS, presc_date, 90)
    num_benzo_prescriptions_30 <- count_prior_prescriptions(PATIENT_PRESC_BENZOS, presc_date, 30)
    
    return(c(num_benzo_prescriptions_180, num_benzo_prescriptions_90, num_benzo_prescriptions_30))
    
  }
  
}

num_prior_prescriptions_benzo <- mcmapply(compute_num_prior_benzo, FULL$patient_id, FULL$prescription_id, mc.cores=50)
FULL$num_prior_prescriptions_benzo_past180 = num_prior_prescriptions_benzo[1, ]
FULL$num_prior_prescriptions_benzo_past90 = num_prior_prescriptions_benzo[2, ]
FULL$num_prior_prescriptions_benzo_past30 = num_prior_prescriptions_benzo[3, ]
# TEST <- FULL[1:60,]

write.csv(FULL, paste("FULL_OPIOID_", year, "_ATLEASTTWO_", batch ,"_FEATURE.csv", sep=""), row.names = FALSE)

rm(num_prior_prescriptions_benzo)


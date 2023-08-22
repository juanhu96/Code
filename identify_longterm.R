### Use long-term use as the alternative outcome variable

library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

################################################################################
########## New outcome variable: long-term use in the next 180 days ############
################################################################################

## All prescriptions in 2018 (without illicit patients)
# FULL_ALL <- read.csv("../Data/FULL_2018_ALERT_NOILLICIT.csv")

# FULL_ALL <- FULL_ALL %>% select(-c(prescription_month, prescription_quarter, 
#                                    alert1_next31, alert2_next31, alert3_next31, alert4_next31, alert5_next31, alert6_next31,
#                                    alert1_next62, alert2_next62, alert3_next62, alert4_next62, alert5_next62, alert6_next62,
#                                    alert1_next93, alert2_next93, alert3_next93, alert4_next93, alert5_next93, alert6_next93,
#                                    alert1_exact31, alert2_exact31, alert3_exact31, alert4_exact31, alert5_exact31, alert6_exact31,
#                                    alert1_exact62, alert2_exact62, alert3_exact62, alert4_exact62, alert5_exact62, alert6_exact62,
#                                    alert1_exact93, alert2_exact93, alert3_exact93, alert4_exact93, alert5_exact93, alert6_exact93))

# FULL_2018 <- FULL_ALL %>% filter(patient_id < 41846152)
# FULL_2018 <- FULL_ALL %>% filter(patient_id >= 41846152 & patient_id < 54195156)
# FULL_2018 <- FULL_ALL %>% filter(patient_id >= 54195156 & patient_id < 68042918)
# FULL_2018 <- FULL_ALL %>% filter(patient_id >= 68042918)

FULL_2018 <- read.csv("../Data/FULL_ALERT_2019_SINGLE.csv")
# FULL_2018 <- read.csv("../Data/FULL_ALERT_2019_ATLEASTTWO_4.csv")

################################################################################
################################################################################
################################################################################
### Long-term use: at least 90 day in 180 days period
### Outcome variable: at the end of current prescription,
### Will the patient become a long-term user

FULL_2018$period_start = format(as.Date(FULL_2018$presc_until, format = "%m/%d/%Y") - 180, "%m/%d/%Y")
FULL_2018 <- FULL_2018[order(FULL_2018$patient_id, as.Date(FULL_2018$date_filled, format = "%m/%d/%Y"), as.Date(FULL_2018$presc_until, format = "%m/%d/%Y")),]

# overlap with the previous (index based) prescription
compute_overlap <- function(pat_id, presc_id){
  PATIENT <- FULL_2018[which(FULL_2018$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_2018[which(FULL_2018$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  if(presc_index == 1){
    return (0)
  } else{
    prev_presc_until <- PATIENT[presc_index-1,c("presc_until")]
    if(as.Date(prev_presc_until, format = "%m/%d/%Y") >= as.Date(presc_date, format = "%m/%d/%Y")){
      return (as.numeric(as.Date(prev_presc_until, format = "%m/%d/%Y") - as.Date(presc_date, format = "%m/%d/%Y")))
    } else{
      return (0)
    }
  }
}

FULL_2018$overlap <- mcmapply(compute_overlap, FULL_2018$patient_id, FULL_2018$prescription_id, mc.cores=50)

########################################################################
compute_opioid_days <- function(pat_id, presc_id){
  PATIENT <- FULL_2018[which(FULL_2018$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL_2018[which(FULL_2018$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_until <- PATIENT_PRESC_OPIOIDS$presc_until
  period_start <- PATIENT_PRESC_OPIOIDS$period_start
  days_supply <- PATIENT_PRESC_OPIOIDS$days_supply
  overlap <- PATIENT_PRESC_OPIOIDS$overlap
  
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
      opioid_days = first_accumulate + days_supply - overlap
    }
    else{
      # overlap considers the co-prescription, but days supply not
      total_overlap = sum(PATIENT_PREV_PRESC[2:nrow(PATIENT_PREV_PRESC), c("overlap")])
      total_days_supply = sum(PATIENT_PREV_PRESC[2:nrow(PATIENT_PREV_PRESC), c("days_supply")])
      opioid_days = first_accumulate + total_days_supply + days_supply - total_overlap - overlap
    }
  }
  # Corner case: I can't think of a better way of doing this for now
  if(opioid_days > 180){
    opioid_days = 180
  }
  return (opioid_days)
}

FULL_2018$opioid_days <- mcmapply(compute_opioid_days, FULL_2018$patient_id, FULL_2018$prescription_id, mc.cores=50)
FULL_2018 <- FULL_2018 %>% mutate(long_term = ifelse(opioid_days >= 90, 1, 0))

# TEST <- FULL_2018 %>% select(c(prescription_id, patient_id, date_filled, days_supply, presc_until, overlap, opioid_days, long_term))
# TEST$opioid_days <- mcmapply(compute_opioid_days, TEST$patient_id, TEST$prescription_id, mc.cores=20)
# TEST <- TEST %>% mutate(long_term = ifelse(opioid_days >= 90, 1, 0))

################################################################################
################################################################################
################################################################################
### Keep prescriptions up to long-term use, once it is long-term use, assume absorbing state
PATIENT <- FULL_2018 %>% group_by(patient_id) %>% summarize(first_presc_date = date_filled[1],
                                                          longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                          longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA),
                                                          first_longterm_presc = ifelse(sum(long_term) > 0, min(row_number()[long_term > 0]), NA),
                                                          first_longterm_presc_id = ifelse(sum(long_term) > 0, prescription_id[long_term > 0][1], NA))

FULL_2018 <- left_join(FULL_2018, PATIENT, by = "patient_id")

## Use presc_until of the long-term prescription to compute
# NA: patient never become long term
# >0: patient is going to become long term
# =0: patient is long term right after this prescription
# <0: patient is already long term
FULL_2018 <- FULL_2018 %>% mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - as.Date(date_filled, format = "%m/%d/%Y"))) %>%
  filter(days_to_long_term > 0 | is.na(days_to_long_term)) %>% # either never long-term or haven't
  mutate(long_term_180 = ifelse(days_to_long_term <= 180, 1, 0)) %>% # within 180 days
  mutate(long_term_180 = ifelse(is.na(long_term_180), 0, long_term_180)) # never is also 0

FULL_2018 <- FULL_2018 %>% select(-c(period_start, first_presc_date,
                                     longterm_filled_date, longterm_presc_date, 
                                     first_longterm_presc, first_longterm_presc_id))


# write.csv(FULL_2018, "../Data/FULL_2018_LONGTERM_NOILLICIT.csv", row.names = FALSE)

################################################################################
########################### TIME-BASED FEATURES ################################
################################################################################

# FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT.csv")

## Number of prescription (xth prescription)
compute_num_presc <- function(pat_id, presc_id){
  # Bounday case: a patient have multiple prescriptions on the same day
  # Do we want number of prescription (day before) the current prescription
  # Or all prescription (even on same day) before the current prescription
  PATIENT <- FULL_2018[which(FULL_2018$patient_id == pat_id),]
  presc_index <- which(PATIENT$prescription_id == presc_id)
  return (presc_index)
}
FULL_2018$num_presc <- mcmapply(compute_num_presc, FULL_2018$patient_id, FULL_2018$prescription_id, mc.cores=40)

## Increase in dosage (for single prescription, or concurrrent MME)
FULL_2018 <- FULL_2018 %>% mutate(previous_dose = ifelse(patient_id == lag(patient_id), lag(daily_dose), 0),
                                  previous_concurrent_MME = ifelse(patient_id == lag(patient_id), lag(concurrent_MME), 0),
                                  previous_days = ifelse(patient_id == lag(patient_id), lag(days_supply), 0),
                                  dose_diff = ifelse(patient_id == lag(patient_id), daily_dose - previous_dose, 0),
                                  concurrent_MME_diff = ifelse(patient_id == lag(patient_id), concurrent_MME - previous_concurrent_MME, 0),
                                  days_diff = ifelse(patient_id == lag(patient_id), days_supply - previous_days, 0))

FULL_2018[1,]$previous_dose = 0
FULL_2018[1,]$previous_concurrent_MME = 0
FULL_2018[1,]$previous_days = 0
FULL_2018[1,]$dose_diff = 0
FULL_2018[1,]$concurrent_MME_diff = 0
FULL_2018[1,]$days_diff = 0

## Switch in drug type/payment
FULL_2018 <- FULL_2018 %>% mutate(switch_drug = ifelse(patient_id == lag(patient_id) & drug == lag(drug), 1, 0),
                                  switch_payment = ifelse(patient_id == lag(patient_id) & payment == lag(payment), 1, 0))
FULL_2018[1,]$switch_drug = 0
FULL_2018[1,]$switch_payment = 0

# write.csv(FULL_2018, "../Data/FULL_2018_LONGTERM_NOILLICIT.csv", row.names = FALSE)

################################################################################
################################ INTERACTION ###################################
################################################################################

FULL_2018 <- FULL_2018 %>% mutate(Codeine_MME = ifelse(drug == "Codeine", daily_dose, 0),
                                  Hydrocodone_MME = ifelse(drug == "Hydrocodone", daily_dose, 0),
                                  Oxycodone_MME = ifelse(drug == "Oxycodone", daily_dose, 0),
                                  Morphine_MME = ifelse(drug == "Morphine", daily_dose, 0),
                                  Hydromorphone_MME = ifelse(drug == "Hydromorphone", daily_dose, 0),
                                  Methadone_MME = ifelse(drug == "Methadone", daily_dose, 0),
                                  Fentanyl_MME = ifelse(drug == "Fentanyl", daily_dose, 0),
                                  Oxymorphone_MME = ifelse(drug == "Oxymorphone", daily_dose, 0))

FULL_2018 <- FULL_2018 %>% mutate(drug_payment = paste(drug, payment, sep = "_")) %>%
  mutate(Codeine_Medicaid = ifelse(drug_payment == "Codeine_Medicaid", 1, 0),
         Codeine_CommercialIns = ifelse(drug_payment == "Codeine_CommercialIns", 1, 0),
         Codeine_Medicare = ifelse(drug_payment == "Codeine_Medicare", 1, 0),
         Codeine_CashCredit = ifelse(drug_payment == "Codeine_CashCredit", 1, 0),
         Codeine_MilitaryIns = ifelse(drug_payment == "Codeine_MilitaryIns", 1, 0),
         Codeine_WorkersComp = ifelse(drug_payment == "Codeine_WorkersComp", 1, 0),
         Codeine_Other = ifelse(drug_payment == "Codeine_Other", 1, 0),
         Codeine_IndianNation = ifelse(drug_payment == "Codeine_IndianNation", 1, 0),
         Hydrocodone_Medicaid = ifelse(drug_payment == "Hydrocodone_Medicaid", 1, 0),
         Hydrocodone_CommercialIns = ifelse(drug_payment == "Hydrocodone_CommercialIns", 1, 0),
         Hydrocodone_Medicare = ifelse(drug_payment == "Hydrocodone_Medicare", 1, 0),
         Hydrocodone_CashCredit = ifelse(drug_payment == "Hydrocodone_CashCredit", 1, 0),
         Hydrocodone_MilitaryIns = ifelse(drug_payment == "Hydrocodone_MilitaryIns", 1, 0),
         Hydrocodone_WorkersComp = ifelse(drug_payment == "Hydrocodone_WorkersComp", 1, 0),
         Hydrocodone_Other = ifelse(drug_payment == "Hydrocodone_Other", 1, 0),
         Hydrocodone_IndianNation = ifelse(drug_payment == "Hydrocodone_IndianNation", 1, 0),
         Oxycodone_Medicaid = ifelse(drug_payment == "Oxycodone_Medicaid", 1, 0),
         Oxycodone_CommercialIns = ifelse(drug_payment == "Oxycodone_CommercialIns", 1, 0),
         Oxycodone_Medicare = ifelse(drug_payment == "Oxycodone_Medicare", 1, 0),
         Oxycodone_CashCredit = ifelse(drug_payment == "Oxycodone_CashCredit", 1, 0),
         Oxycodone_MilitaryIns = ifelse(drug_payment == "Oxycodone_MilitaryIns", 1, 0),
         Oxycodone_WorkersComp = ifelse(drug_payment == "Oxycodone_WorkersComp", 1, 0),
         Oxycodone_Other = ifelse(drug_payment == "Oxycodone_Other", 1, 0),
         Oxycodone_IndianNation = ifelse(drug_payment == "Oxycodone_IndianNation", 1, 0),
         Morphine_Medicaid = ifelse(drug_payment == "Morphine_Medicaid", 1, 0),
         Morphine_CommercialIns = ifelse(drug_payment == "Morphine_CommercialIns", 1, 0),
         Morphine_Medicare = ifelse(drug_payment == "Morphine_Medicare", 1, 0),
         Morphine_CashCredit = ifelse(drug_payment == "Morphine_CashCredit", 1, 0),
         Morphine_MilitaryIns = ifelse(drug_payment == "Morphine_MilitaryIns", 1, 0),
         Morphine_WorkersComp = ifelse(drug_payment == "Morphine_WorkersComp", 1, 0),
         Morphine_Other = ifelse(drug_payment == "Morphine_Other", 1, 0),
         Morphine_IndianNation = ifelse(drug_payment == "Morphine_IndianNation", 1, 0),
         Hydromorphone_Medicaid = ifelse(drug_payment == "Hydromorphone_Medicaid", 1, 0),
         Hydromorphone_CommercialIns = ifelse(drug_payment == "Hydromorphone_CommercialIns", 1, 0),
         Hydromorphone_Medicare = ifelse(drug_payment == "Hydromorphone_Medicare", 1, 0),
         Hydromorphone_CashCredit = ifelse(drug_payment == "Hydromorphone_CashCredit", 1, 0),
         Hydromorphone_MilitaryIns = ifelse(drug_payment == "Hydromorphone_MilitaryIns", 1, 0),
         Hydromorphone_WorkersComp = ifelse(drug_payment == "Hydromorphone_WorkersComp", 1, 0),
         Hydromorphone_Other = ifelse(drug_payment == "Hydromorphone_Other", 1, 0),
         Hydromorphone_IndianNation = ifelse(drug_payment == "Hydromorphone_IndianNation", 1, 0),
         Methadone_Medicaid = ifelse(drug_payment == "Methadone_Medicaid", 1, 0),
         Methadone_CommercialIns = ifelse(drug_payment == "Methadone_CommercialIns", 1, 0),
         Methadone_Medicare = ifelse(drug_payment == "Methadone_Medicare", 1, 0),
         Methadone_CashCredit = ifelse(drug_payment == "Methadone_CashCredit", 1, 0),
         Methadone_MilitaryIns = ifelse(drug_payment == "Methadone_MilitaryIns", 1, 0),
         Methadone_WorkersComp = ifelse(drug_payment == "Methadone_WorkersComp", 1, 0),
         Methadone_Other = ifelse(drug_payment == "Methadone_Other", 1, 0),
         Methadone_IndianNation = ifelse(drug_payment == "Methadone_IndianNation", 1, 0),
         Fentanyl_Medicaid = ifelse(drug_payment == "Fentanyl_Medicaid", 1, 0),
         Fentanyl_CommercialIns = ifelse(drug_payment == "Fentanyl_CommercialIns", 1, 0),
         Fentanyl_Medicare = ifelse(drug_payment == "Fentanyl_Medicare", 1, 0),
         Fentanyl_CashCredit = ifelse(drug_payment == "Fentanyl_CashCredit", 1, 0),
         Fentanyl_MilitaryIns = ifelse(drug_payment == "Fentanyl_MilitaryIns", 1, 0),
         Fentanyl_WorkersComp = ifelse(drug_payment == "Fentanyl_WorkersComp", 1, 0),
         Fentanyl_Other = ifelse(drug_payment == "Fentanyl_Other", 1, 0),
         Fentanyl_IndianNation = ifelse(drug_payment == "Fentanyl_IndianNation", 1, 0),
         Oxymorphone_Medicaid = ifelse(drug_payment == "Oxymorphone_Medicaid", 1, 0),
         Oxymorphone_CommercialIns = ifelse(drug_payment == "Oxymorphone_CommercialIns", 1, 0),
         Oxymorphone_Medicare = ifelse(drug_payment == "Oxymorphone_Medicare", 1, 0),
         Oxymorphone_CashCredit = ifelse(drug_payment == "Oxymorphone_CashCredit", 1, 0),
         Oxymorphone_MilitaryIns = ifelse(drug_payment == "Oxymorphone_MilitaryIns", 1, 0),
         Oxymorphone_WorkersComp = ifelse(drug_payment == "Oxymorphone_WorkersComp", 1, 0),
         Oxymorphone_Other = ifelse(drug_payment == "Oxymorphone_Other", 1, 0),
         Oxymorphone_IndianNation = ifelse(drug_payment == "Oxymorphone_IndianNation", 1, 0))

# write.csv(FULL_2018, "../Data/FULL_2018_LONGTERM_NOILLICIT_4.csv", row.names = FALSE)
write.csv(FULL_2018, "../Data/FULL_2019_LONGTERM_5.csv", row.names = FALSE)

# TEST <- FULL_2018 %>% filter(patient_id == 708) %>% select(c(patient_id, date_filled, presc_until, days_supply, overlap, opioid_days, long_term, long_term_180))

################################################################################
################################### INPUT ######################################
################################################################################

## For input
# FULL_2018_1 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_1.csv")
# FULL_2018_2 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_2.csv")
# FULL_2018_3 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_3.csv")
# FULL_2018_4 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT_4.csv")
# 
# FULL_2018 <- rbind(FULL_2018_1, FULL_2018_2)
# FULL_2018 <- rbind(FULL_2018, FULL_2018_3)
# FULL_2018 <- rbind(FULL_2018, FULL_2018_4)
# rm(FULL_2018_1)
# rm(FULL_2018_2)
# rm(FULL_2018_3)
# rm(FULL_2018_4)


FULL_2018_1 <- read.csv("../Data/FULL_2019_LONGTERM_1.csv")
FULL_2018_2 <- read.csv("../Data/FULL_2019_LONGTERM_2.csv")
FULL_2018_3 <- read.csv("../Data/FULL_2019_LONGTERM_3.csv")
FULL_2018_4 <- read.csv("../Data/FULL_2019_LONGTERM_4.csv")
FULL_2018_5 <- read.csv("../Data/FULL_2019_LONGTERM_5.csv")

FULL_2018 <- rbind(FULL_2018_1, FULL_2018_2)
FULL_2018 <- rbind(FULL_2018, FULL_2018_3)
FULL_2018 <- rbind(FULL_2018, FULL_2018_4)
FULL_2018 <- rbind(FULL_2018, FULL_2018_5)
rm(FULL_2018_1)
rm(FULL_2018_2)
rm(FULL_2018_3)
rm(FULL_2018_4)
rm(FULL_2018_5)
FULL_2018 <- FULL_2018[order(FULL_2018$patient_id, FULL_2018$date_filled),]
write.csv(FULL_2018, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)

LONGTERMPRESC <- FULL_2018 %>% filter(long_term_180 == 1)

# FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_NOILLICIT.csv")
# FULL_2018 <- FULL_2018 %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
#                                      prescriber_zip, pharmacy_id, pharmacy_zip, strength, days_supply, date_filled, presc_until, 
#                                      conversion, class, drug, daily_dose, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
#                                      DEASCHEDULE, MAINDOSE, payment, prescription_year,
#                                      presence_MME, presence_MME_methadone, presence_num_prescribers, presence_num_pharmacies, 
#                                      overlap, presence_days, # alert1, alert2, alert3, alert4, alert5, alert6,
#                                      alert_next31, alert_exact31, alert_next62, alert_exact62, alert_next93, alert_exact93,
#                                      days_to_long_term, previous_dose, previous_concurrent_MME, previous_days, num_alert, long_term))

FULL_2018 <- FULL_2018 %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                     prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                     conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                     DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_year,
                                     overlap, # alert1, alert2, alert3, alert4, alert5, alert6,
                                     days_to_long_term, previous_dose, previous_concurrent_MME, previous_days, num_alert, opioid_days, long_term))

FULL_2018 <- rename(FULL_2018, concurrent_methadone_MME = concurrent_MME_methadone)
FULL_2018 <- rename(FULL_2018, MME_diff = concurrent_MME_diff)

# write.csv(FULL_2018, "../Data/FULL_2018_INPUT_LONGTERM_NOILLICIT.csv", row.names = FALSE)
write.csv(FULL_2018, "../Data/FULL_2019_INPUT_LONGTERM.csv", row.names = FALSE)


### Proportion of long_term users, days to long_term
PATIENT <- FULL_2018 %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                            age = age[1],
                                                            first_presc_date = date_filled[1],
                                                            longterm = ifelse(sum(long_term) > 0, 1, 0), 
                                                            longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                            longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA))
write.csv(PATIENT, "../Data/PATIENT_2018.csv", row.names = FALSE)
# 261954 
PATIENT_LONGTERM <- PATIENT %>% filter(longterm == 1) 
PATIENT_LONGTERM <- PATIENT_LONGTERM %>% mutate(days_to_long_term = as.numeric(as.Date(longterm_presc_date, format = "%m/%d/%Y") - 
                                                  as.Date(first_presc_date, format = "%m/%d/%Y")))

TEST <- FULL_2018 %>% filter(patient_id == 64134737)

# Note the patient is not a long term user at date_filled, but it will become during [date_filled, date_end]
ggplot(PATIENT_LONGTERM, aes(x=ifelse(days_to_long_term>365, 365, days_to_long_term))) + 
  geom_density(adjust = 1) +
  ggtitle("Days until long term use") +
  xlab("Days") +
  ylab("Density") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 14))
ggsave("../Result/days_longterm.pdf", bg="white", width=8, height=6, dpi=300)


################################################################################
## Summary
TEST <- PATIENT %>% filter(patient_gender == 0)
TEST <- FULL_2018 %>% filter(patient_gender == 0)

# Prescriptions
TEST <- FULL_2018 %>% filter(age <= 20)
TEST <- FULL_2018 %>% filter(age > 20 & age <= 40)
TEST <- FULL_2018 %>% filter(age > 40 & age <= 60)
TEST <- FULL_2018 %>% filter(age > 60 & age <= 80)
TEST <- FULL_2018 %>% filter(age > 80)

# Patient
TEST <- PATIENT %>% filter(age <= 20)
TEST <- PATIENT %>% filter(age > 20 & age <= 40)
TEST <- PATIENT %>% filter(age > 40 & age <= 60)
TEST <- PATIENT %>% filter(age > 60 & age <= 80)
TEST <- PATIENT %>% filter(age > 80)

# Long term user
TEST <- PATIENT_LONGTERM %>% filter(patient_gender == 0)

TEST <- PATIENT_LONGTERM %>% filter(age <= 20)
TEST <- PATIENT_LONGTERM %>% filter(age > 20 & age <= 40)
TEST <- PATIENT_LONGTERM %>% filter(age > 40 & age <= 60)
TEST <- PATIENT_LONGTERM %>% filter(age > 60 & age <= 80)
TEST <- PATIENT_LONGTERM %>% filter(age > 80)

################################################################################
################################# Barplot ######################################
################################################################################


# FULL_2018$payment <- factor(FULL_2018$payment, levels = c("CommercialIns", "CashCredit", "Medicare", "Medicaid", 
#                                                           "Other", "WorkersComp", "MilitaryIns", "IndianNation" ))
ggplot(FULL_2018, aes(drug)) + 
  geom_bar(aes(fill = payment)) +
  xlab("Drug type") +
  ylab("Count") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 10)) + 
  labs(colour = "Payment type") 
ggsave("../Result/drug_payment.pdf", bg="white", width=12, height=6, dpi=300)

# One prescription could have multiple alerts so it's hard to make barplot
ALERT <- FULL_2018 %>% filter(alert1 == 1)
ALERT <- FULL_2018 %>% filter(alert6 == 1)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
############################# Ever Switch ######################################
################################################################################

# FULL <- read.csv("../Data/FULL_2018_LONGTERM.csv")
FULL <- read.csv("../Data/FULL_2019_LONGTERM.csv")

# Recompute switch_drug and switch_payment (b/c I did it wrong)
FULL <- FULL %>% mutate(switch_drug = ifelse(patient_id == lag(patient_id) & drug != lag(drug), 1, 0),
                        switch_payment = ifelse(patient_id == lag(patient_id) & payment != lag(payment), 1, 0))
FULL[1,]$switch_drug = 0
FULL[1,]$switch_payment = 0

# FULL <- FULL %>% select(-c(first_switch_drug, first_switch_payment)) # for 2019 only

# Ever switch
PATIENT <- FULL %>% group_by(patient_id) %>% summarize(first_switch_drug = ifelse(sum(switch_drug) > 0, date_filled[switch_drug > 0][1], "01/01/2020"),
                                                       first_switch_payment = ifelse(sum(switch_payment) > 0, date_filled[switch_payment > 0][1], "01/01/2020"))
FULL <- left_join(FULL, PATIENT, by = 'patient_id')

FULL <- FULL %>% mutate(ever_switch_drug = ifelse(as.Date(date_filled, format = "%m/%d/%Y") >= as.Date(first_switch_drug, format = "%m/%d/%Y"), 1, 0),
                        ever_switch_payment = ifelse(as.Date(date_filled, format = "%m/%d/%Y") >= as.Date(first_switch_payment, format = "%m/%d/%Y"), 1, 0))

# write.csv(FULL, "../Data/FULL_2018_LONGTERM.csv", row.names = FALSE)
write.csv(FULL, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)

################################################################################
########################## Avg MME/days, HMFO ##################################
################################################################################

# FULL <- read.csv("../Data/FULL_2018_LONGTERM.csv")
FULL <- read.csv("../Data/FULL_2019_LONGTERM.csv")

compute_avg <- function(pat_id, presc_id){
  PATIENT <- FULL[which(FULL$patient_id == pat_id),]
  PATIENT_PRESC_OPIOIDS <- FULL[which(FULL$prescription_id == presc_id),]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  presc_index <- which(PATIENT$prescription_id == presc_id)
  
  
  # prescriptions before (in terms of time & index)
  # index for case where multiple prescription on same date
  PATIENT_PREV_PRESC <- PATIENT[which(as.Date(PATIENT$date_filled, format = "%m/%d/%Y") <=
                                        as.Date(presc_date, format = "%m/%d/%Y") &
                                        PATIENT$prescription_id <= presc_id), ]
  
  avg_dailyMME <- mean(PATIENT_PREV_PRESC$daily_dose)
  avg_days <- mean(PATIENT_PREV_PRESC$days_supply)
  return (c(avg_dailyMME, avg_days))
}


avg <- mcmapply(compute_avg, FULL$patient_id, FULL$prescription_id, mc.cores=40)
FULL$avgMME = avg[1, ]
FULL$avgDays = avg[2, ]

FULL <- FULL %>% mutate(HMFO = ifelse(drug == "Hydromorphone" | drug == "Methadone" | drug == "Fentanyl" | drug == "Oxymorphone", 1, 0))
# write.csv(FULL, "../Data/FULL_2018_LONGTERM.csv", row.names = FALSE)
write.csv(FULL, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE)




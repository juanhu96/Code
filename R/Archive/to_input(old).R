### Transfer dataset into the input format
### Note that input is only for creating stumps (for flexible & full)

library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

################################################################################
################################################################################
################################################################################

FULL_2018_INPUT <- read.csv("../Data/FULL_2018_LONGTERM_INPUT.csv") # previous input, will be overwritten
FULL_2018_NEW <- read.csv("../Data/FULL_2018_LONGTERM.csv")

FULL_2018_NEW <- FULL_2018_NEW %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                             prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                             conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                             DEASCHEDULE, MAINDOSE, payment, prescription_year,
                                             overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                             days_to_long_term, num_alert, opioid_days, long_term))

cols_only_in_df1 <- setdiff(names(FULL_2018_NEW), names(FULL_2018_INPUT)) # Only in new
cols_only_in_df2 <- setdiff(names(FULL_2018_INPUT), names(FULL_2018_NEW)) # Only in previous input

write.csv(FULL_2018_NEW, "../Data/FULL_2018_LONGTERM_INPUT.csv", row.names = FALSE)

################################################################################
################################################################################
################################################################################

FULL_2019_INPUT <- read.csv("../Data/FULL_2019_LONGTERM_INPUT.csv") # previous input, will be overwritten
FULL_2019_NEW <- read.csv("../Data/FULL_2019_LONGTERM.csv")

FULL_2019_NEW <- FULL_2019_NEW %>% mutate(Codeine = ifelse(Codeine_MME > 0, 1, 0),
                                          Hydrocodone = ifelse(Hydrocodone_MME > 0, 1, 0),
                                          Oxycodone = ifelse(Oxycodone_MME > 0, 1, 0),
                                          Morphine = ifelse(Morphine_MME > 0, 1, 0),
                                          Hydromorphone = ifelse(Hydromorphone_MME > 0, 1, 0),
                                          Methadone = ifelse(Methadone_MME > 0, 1, 0),
                                          Fentanyl = ifelse(Fentanyl_MME > 0, 1, 0),
                                          Oxymorphone = ifelse(Oxymorphone_MME > 0, 1, 0)) %>% 
  mutate(Medicaid = ifelse(payment == "Medicaid", 1, 0),
         CommercialIns = ifelse(payment == "CommercialIns", 1, 0),
         Medicare = ifelse(payment == "Medicare", 1, 0),
         CashCredit = ifelse(payment == "CashCredit", 1, 0),
         MilitaryIns = ifelse(payment == "MilitaryIns", 1, 0),
         WorkersComp = ifelse(payment == "WorkersComp", 1, 0),
         Other = ifelse(payment == "Other", 1, 0),
         IndianNation = ifelse(payment == "IndianNation", 1, 0))

FULL_2019_NEW <- FULL_2019_NEW %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                             prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                             conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                             DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_year,
                                             overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                             days_to_long_term, previous_dose, previous_concurrent_MME, previous_days, 
                                             num_alert, opioid_days, long_term))

cols_only_in_df1 <- setdiff(names(FULL_2019_NEW), names(FULL_2019_INPUT)) # Only in new
cols_only_in_df2 <- setdiff(names(FULL_2019_INPUT), names(FULL_2019_NEW)) # Only in previous input

write.csv(FULL_2019_NEW, "../Data/FULL_2019_LONGTERM_INPUT.csv", row.names = FALSE)

################################################################################
################################################################################
################################################################################
### Double check
FULL_2018_INPUT <- read.csv("../Data/FULL_2018_INPUT_LONGTERM.csv")
FULL_2019_INPUT <- read.csv("../Data/FULL_2019_INPUT_LONGTERM.csv")

cols_only_in_df1 <- setdiff(names(FULL_2018_INPUT), names(FULL_2019_INPUT)) # Only in 2018
cols_only_in_df2 <- setdiff(names(FULL_2019_INPUT), names(FULL_2018_INPUT)) # Only in 2019

colnames(FULL_2019_INPUT)

################################################################################
################################################################################
################################################################################

FULL_2018_INPUT <- read.csv("../Data/FULL_2018_LONGTERM_INPUT.csv")
FULL_2019_INPUT <- read.csv("../Data/FULL_2019_LONGTERM_INPUT.csv")

################################################################################
################################################################################
################################################################################

FULL_2018_STUMPS <- read.csv("../Data/FULL_2018_STUMPS0.csv")
colnames(FULL_2018_STUMPS)
colnames(FULL_2018_INPUT)

'opioid_days' %in% colnames(FULL_2018_STUMPS)


################################################################################
################################################################################
################################################################################

FULL_2018_NEW <- read.csv("../Data/FULL_2018_LONGTERM.csv")

prob_list = c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0)

quantity <- quantile(FULL_2018_NEW$quantity, probs = prob_list)
daily_dose <- quantile(FULL_2018_NEW$daily_dose, probs = prob_list)
days_supply <- quantile(FULL_2018_NEW$days_supply, probs = prob_list)
concurrent_MME <- quantile(FULL_2018_NEW$concurrent_MME, probs = prob_list)
concurrent_methadone_MME <- quantile(FULL_2018_NEW$concurrent_methadone_MME, probs = prob_list)
num_prescribers <- quantile(FULL_2018_NEW$num_prescribers, probs = prob_list)
num_pharmacies <- quantile(FULL_2018_NEW$num_pharmacies, probs = prob_list)
concurrent_benzo <- quantile(FULL_2018_NEW$concurrent_benzo, probs = prob_list)
consecutive_days <- quantile(FULL_2018_NEW$consecutive_days, probs = prob_list)
num_presc <- quantile(FULL_2018_NEW$num_presc, probs = prob_list)
dose_diff <- quantile(FULL_2018_NEW$dose_diff, probs = prob_list)
MME_diff <- quantile(FULL_2018_NEW$MME_diff, probs = prob_list)
days_diff <- quantile(FULL_2018_NEW$days_diff, probs = prob_list)
age <- quantile(FULL_2018_NEW$age, probs = prob_list)

quantiles <- rbind(quantity, daily_dose)
quantiles <- rbind(quantiles, days_supply)
quantiles <- rbind(quantiles, concurrent_MME)
quantiles <- rbind(quantiles, concurrent_methadone_MME)
quantiles <- rbind(quantiles, num_prescribers)
quantiles <- rbind(quantiles, num_pharmacies)
quantiles <- rbind(quantiles, concurrent_benzo)
quantiles <- rbind(quantiles, consecutive_days)
quantiles <- rbind(quantiles, num_presc)
quantiles <- rbind(quantiles, dose_diff)
quantiles <- rbind(quantiles, MME_diff)
quantiles <- rbind(quantiles, days_diff)
quantiles <- rbind(quantiles, age)

quantiles_df <- as.data.frame(quantiles)

write.csv(quantiles_df, "../Data/Quantiles.csv", row.names = TRUE)

################################################################################
################################################################################
################################################################################
### Compute differences in quantity and drop outliers
FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv")
FULL_2018 <- FULL_2018 %>% mutate(previous_quantity = ifelse(patient_id == lag(patient_id), lag(quantity), 0),
                                  quantity_diff =  ifelse(patient_id == lag(patient_id), quantity - previous_quantity, 0))

FULL_2018[1,]$previous_quantity = 0
FULL_2018[1,]$quantity_diff = 0
FULL_2018 <- FULL_2018 %>% select(-c(previous_quantity))
# TEMP <- FULL_2018[1:20,]

### Collect outlier patients
# 46876 prescriptions
OUTLIER_PRESC <- FULL_2018 %>% filter(quantity >= 1000 | concurrent_MME >= 1000 | concurrent_methadone_MME >= 1000 |
                                        num_prescribers > 10 | num_pharmacies > 10 | num_presc >= 100 | age >= 100) 
# 24012 patients
OUTLIER_PATIENT <- unique(OUTLIER_PRESC$patient_id) 
FULL_2018_FILTERED <- FULL_2018 %>% filter(!patient_id %in% OUTLIER_PATIENT)
write.csv(FULL_2018_FILTERED, "../Data/FULL_2018_LONGTERM.csv", row.names = FALSE) # 146

### Create input
FULL_2018_INPUT <- FULL_2018_FILTERED %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id,
                                                    prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                                    conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                                    DEASCHEDULE, MAINDOSE, payment, prescription_year,
                                                    overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                                    days_to_long_term, num_alert, opioid_days, long_term))
write.csv(FULL_2018_INPUT, "../Data/FULL_2018_INPUT_LONGTERM.csv", row.names = FALSE) # 112

################################################################################
### For 2019
FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")

FULL_2019 <- rename(FULL_2019, concurrent_methadone_MME = concurrent_MME_methadone)
FULL_2019 <- rename(FULL_2019, MME_diff = concurrent_MME_diff)
FULL_2019 <- FULL_2019 %>% mutate(Codeine = ifelse(Codeine_MME > 0, 1, 0),
                                  Hydrocodone = ifelse(Hydrocodone_MME > 0, 1, 0),
                                  Oxycodone = ifelse(Oxycodone_MME > 0, 1, 0),
                                  Morphine = ifelse(Morphine_MME > 0, 1, 0),
                                  Hydromorphone = ifelse(Hydromorphone_MME > 0, 1, 0),
                                  Methadone = ifelse(Methadone_MME > 0, 1, 0),
                                  Fentanyl = ifelse(Fentanyl_MME > 0, 1, 0),
                                  Oxymorphone = ifelse(Oxymorphone_MME > 0, 1, 0)) %>% 
  mutate(Medicaid = ifelse(payment == "Medicaid", 1, 0),
         CommercialIns = ifelse(payment == "CommercialIns", 1, 0),
         Medicare = ifelse(payment == "Medicare", 1, 0),
         CashCredit = ifelse(payment == "CashCredit", 1, 0),
         MilitaryIns = ifelse(payment == "MilitaryIns", 1, 0),
         WorkersComp = ifelse(payment == "WorkersComp", 1, 0),
         Other = ifelse(payment == "Other", 1, 0),
         IndianNation = ifelse(payment == "IndianNation", 1, 0))

FULL_2019 <- FULL_2019 %>% mutate(previous_quantity = ifelse(patient_id == lag(patient_id), lag(quantity), 0),
                                  quantity_diff =  ifelse(patient_id == lag(patient_id), quantity - previous_quantity, 0))

FULL_2019[1,]$previous_quantity = 0
FULL_2019[1,]$quantity_diff = 0
FULL_2019 <- FULL_2019 %>% select(-c(previous_quantity))

### Collect outlier patients
# 102903 prescriptions
OUTLIER_PRESC <- FULL_2019 %>% filter(quantity >= 1000 | concurrent_MME >= 1000 | concurrent_methadone_MME >= 1000 |
                                        num_prescribers > 10 | num_pharmacies > 10 | num_presc >= 100 | age >= 100)
# 25175 patients
OUTLIER_PATIENT <- unique(OUTLIER_PRESC$patient_id) 
FULL_2019_FILTERED <- FULL_2019 %>% filter(!patient_id %in% OUTLIER_PATIENT)
write.csv(FULL_2019_FILTERED, "../Data/FULL_2019_LONGTERM.csv", row.names = FALSE) # 150

### Create input
FULL_2019_INPUT <- FULL_2019_FILTERED %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id, 
                                                    prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                                    conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                                    DEASCHEDULE, MAINDOSE, payment, prescription_month, prescription_year,
                                                    overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                                    days_to_long_term, previous_dose, previous_concurrent_MME, previous_days, 
                                                    num_alert, opioid_days, long_term))
write.csv(FULL_2019_INPUT, "../Data/FULL_2019_INPUT_LONGTERM.csv", row.names = FALSE) # 112

################################################################################

FULL_2018_STUMPS <- read.csv("../Data/FULL_2018_STUMPS_UPTOFIRST0.csv") # 6,060,797 from 3735323 patients

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

### Keep the prescriptions up to first 1 only
FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv") # 6,060,797 from 3735323 patients
# TEMP <- FULL_2018[1:200,]
PATIENT <- FULL_2018 %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                            age = age[1],
                                                            first_presc_date = date_filled[1],
                                                            num_prescriptions = n(),
                                                            longterm = ifelse(sum(long_term) > 0, 1, 0), 
                                                            long_term_180_date = ifelse(sum(long_term_180) > 0, date_filled[long_term_180 > 0][1], "01/01/2020"))


# TEMP <- left_join(TEMP, PATIENT[ , c("patient_id", "long_term_180_date")], by = "patient_id") 
FULL_2018 <- left_join(FULL_2018, PATIENT[ , c("patient_id", "long_term_180_date")], by = "patient_id")
FULL_2018 <- FULL_2018 %>% filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(long_term_180_date, format = "%m/%d/%Y")) # 5,637,115 from 3735323 patients
write.csv(FULL_2018, "../Data/FULL_2018_LONGTERM_UPTOFIRST.csv", row.names = FALSE)

FULL_2018_INPUT <- FULL_2018 %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id,
                                           prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                           conversion, class, drug, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                           DEASCHEDULE, MAINDOSE, payment, prescription_year,
                                           overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                           days_to_long_term, num_alert, opioid_days, long_term, long_term_180_date,
                                           first_switch_drug, first_switch_payment))
write.csv(FULL_2018_INPUT, "../Data/FULL_2018_LONGTERM_INPUT_UPTOFIRST.csv", row.names = FALSE) # 112 -> 114 -> 116 now


# prob_list = c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0)
# avgMME <- quantile(FULL_2018$avgMME, probs = prob_list)
# avgDays <- quantile(FULL_2018$avgDays, probs = prob_list)
# quantiles <- rbind(avgMME, avgDays)
# quantiles_df <- as.data.frame(quantiles)

################################################################################

FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM.csv")
PATIENT <- FULL_2019 %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                            age = age[1],
                                                            first_presc_date = date_filled[1],
                                                            num_prescriptions = n(),
                                                            longterm = ifelse(sum(long_term) > 0, 1, 0), 
                                                            long_term_180_date = ifelse(sum(long_term_180) > 0, date_filled[long_term_180 > 0][1], "01/01/2020"))

FULL_2019 <- left_join(FULL_2019, PATIENT[ , c("patient_id", "long_term_180_date")], by = "patient_id")
FULL_2019 <- FULL_2019 %>% filter(as.Date(date_filled, format = "%m/%d/%Y") <= as.Date(long_term_180_date, format = "%m/%d/%Y"))
write.csv(FULL_2019, "../Data/FULL_2019_LONGTERM_UPTOFIRST.csv", row.names = FALSE)

################################################################################

# Merge HPI to data
FULL_2018_UPTOFIRST <- read.csv("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv")
FULL_2019_UPTOFIRST <- read.csv("../Data/FULL_2019_LONGTERM_UPTOFIRST.csv")

# TEST <- FULL_2018_UPTOFIRST[1:10,]
FULL_2018_UPTOFIRST <- FULL_2018_UPTOFIRST %>% mutate(patient_zip = as.character(patient_zip),
                                                      prescriber_zip = as.character(prescriber_zip),
                                                      pharmacy_zip = as.character(pharmacy_zip))

FULL_2019_UPTOFIRST <- FULL_2019_UPTOFIRST %>% mutate(patient_zip = as.character(patient_zip),
                                                      prescriber_zip = as.character(prescriber_zip),
                                                      pharmacy_zip = as.character(pharmacy_zip))

HPI  <- read.csv("../Data/CA/HPI.csv")
HPI <- HPI %>% mutate(Zip = as.character(Zip))
HPI_patient <- HPI %>% rename(patient_zip = Zip, patient_HPIQuartile = HPIQuartile) %>% select(-c(HPI))
HPI_prescriber <- HPI %>% rename(prescriber_zip = Zip, prescriber_HPIQuartile = HPIQuartile) %>% select(-c(HPI))
HPI_pharmacy <- HPI %>% rename(pharmacy_zip = Zip, pharmacy_HPIQuartile = HPIQuartile) %>% select(-c(HPI))

FULL_2018_UPTOFIRST <- FULL_2018_UPTOFIRST %>%
  left_join(HPI_patient, by = "patient_zip") %>%
  left_join(HPI_prescriber, by = "prescriber_zip") %>%
  left_join(HPI_pharmacy, by = "pharmacy_zip")

FULL_2018_UPTOFIRST <- FULL_2018_UPTOFIRST %>%
  mutate(patient_HPIQuartile_1 = ifelse(patient_HPIQuartile == 1, 1, 0),
         patient_HPIQuartile_2 = ifelse(patient_HPIQuartile == 2, 1, 0),
         patient_HPIQuartile_3 = ifelse(patient_HPIQuartile == 3, 1, 0),
         patient_HPIQuartile_4 = ifelse(patient_HPIQuartile == 4, 1, 0),
         prescriber_HPIQuartile_1 = ifelse(prescriber_HPIQuartile == 1, 1, 0),
         prescriber_HPIQuartile_2 = ifelse(prescriber_HPIQuartile == 2, 1, 0),
         prescriber_HPIQuartile_3 = ifelse(prescriber_HPIQuartile == 3, 1, 0),
         prescriber_HPIQuartile_4 = ifelse(prescriber_HPIQuartile == 4, 1, 0),
         pharmacy_HPIQuartile_1 = ifelse(pharmacy_HPIQuartile == 1, 1, 0),
         pharmacy_HPIQuartile_2 = ifelse(pharmacy_HPIQuartile == 2, 1, 0),
         pharmacy_HPIQuartile_3 = ifelse(pharmacy_HPIQuartile == 3, 1, 0),
         pharmacy_HPIQuartile_4 = ifelse(pharmacy_HPIQuartile == 4, 1, 0))

FULL_2019_UPTOFIRST <- FULL_2019_UPTOFIRST %>%
  left_join(HPI_patient, by = "patient_zip") %>%
  left_join(HPI_prescriber, by = "prescriber_zip") %>%
  left_join(HPI_pharmacy, by = "pharmacy_zip")

FULL_2019_UPTOFIRST <- FULL_2019_UPTOFIRST %>%
  mutate(patient_HPIQuartile_1 = ifelse(patient_HPIQuartile == 1, 1, 0),
         patient_HPIQuartile_2 = ifelse(patient_HPIQuartile == 2, 1, 0),
         patient_HPIQuartile_3 = ifelse(patient_HPIQuartile == 3, 1, 0),
         patient_HPIQuartile_4 = ifelse(patient_HPIQuartile == 4, 1, 0),
         prescriber_HPIQuartile_1 = ifelse(prescriber_HPIQuartile == 1, 1, 0),
         prescriber_HPIQuartile_2 = ifelse(prescriber_HPIQuartile == 2, 1, 0),
         prescriber_HPIQuartile_3 = ifelse(prescriber_HPIQuartile == 3, 1, 0),
         prescriber_HPIQuartile_4 = ifelse(prescriber_HPIQuartile == 4, 1, 0),
         pharmacy_HPIQuartile_1 = ifelse(pharmacy_HPIQuartile == 1, 1, 0),
         pharmacy_HPIQuartile_2 = ifelse(pharmacy_HPIQuartile == 2, 1, 0),
         pharmacy_HPIQuartile_3 = ifelse(pharmacy_HPIQuartile == 3, 1, 0),
         pharmacy_HPIQuartile_4 = ifelse(pharmacy_HPIQuartile == 4, 1, 0))

write.csv(FULL_2018_UPTOFIRST, "../Data/FULL_2018_LONGTERM_UPTOFIRST.csv", row.names = FALSE)
write.csv(FULL_2019_UPTOFIRST, "../Data/FULL_2019_LONGTERM_UPTOFIRST.csv", row.names = FALSE)

################################################################################

# Descriptive analysis of HPI

# ALL
summary_HPIQuartile <- FULL_2018_UPTOFIRST %>%
  # group_by(patient_HPIQuartile) %>%
  # group_by(prescriber_HPIQuartile) %>%
  group_by(pharmacy_HPIQuartile) %>%
  summarise(
    num_prescriptions = n(),
    unique_patients = n_distinct(patient_id),
    unique_pharmacies = n_distinct(pharmacy_id),
    unique_prescribers = n_distinct(prescriber_id)
  )

summary_long <- summary_HPIQuartile %>%
  pivot_longer(cols = c(num_prescriptions, unique_patients),
               names_to = "Type",
               values_to = "Number")
ggplot(summary_long, aes(x = pharmacy_HPIQuartile, y = Number/1000000, fill = Type, pattern = Type)) +
  geom_bar(stat = "identity", position = position_dodge(), alpha = 1) +
  scale_fill_manual(values = c("num_prescriptions" = "#2774AE", "unique_patients" = "#003B5C")) +
  # labs(x = "Patient HPI Quartile", y = "Million") +
  # labs(x = "Prescriber HPI Quartile", y = "Million") +
  labs(x = "Pharmacy HPI Quartile", y = "Million") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.spacing = unit(2, "lines"),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        legend.key.width = unit(1.2, "cm"),
        legend.key.height = unit(1, "cm"),
        plot.subtitle = element_text(hjust = 1, vjust = 1)
  )

# ggsave("../Figs/Patient_HPI.pdf", width = 6, height = 5, dpi = 300)
# ggsave("../Figs/Prescriber_HPI.pdf", width = 6, height = 5, dpi = 300)
ggsave("../Figs/Pharmacy_HPI.pdf", width = 6, height = 5, dpi = 300)



# LONGTERM
PATIENT <- FULL_2018_UPTOFIRST %>% group_by(patient_id) %>% summarize(longterm = ifelse(sum(long_term_180) > 0, 1, 0)) # use long_term_180 instead of long_term
FULL_2018_UPTOFIRST <- left_join(FULL_2018_UPTOFIRST, PATIENT[ , c("patient_id", "longterm")], by = "patient_id")
LONGTERM <- FULL_2018_UPTOFIRST %>% filter(longterm == 1)

summary_HPIQuartile <- LONGTERM %>%
  # group_by(patient_HPIQuartile) %>%
  # group_by(prescriber_HPIQuartile) %>%
  group_by(pharmacy_HPIQuartile) %>%
  summarise(
    num_prescriptions = n(),
    unique_patients = n_distinct(patient_id),
    unique_pharmacies = n_distinct(pharmacy_id),
    unique_prescribers = n_distinct(prescriber_id)
  )

summary_long <- summary_HPIQuartile %>%
  pivot_longer(cols = c(num_prescriptions, unique_patients),
               names_to = "Type",
               values_to = "Number")
ggplot(summary_long, aes(x = pharmacy_HPIQuartile, y = Number/1000000, fill = Type, pattern = Type)) +
  geom_bar(stat = "identity", position = position_dodge(), alpha = 1) +
  scale_fill_manual(values = c("num_prescriptions" = "#2774AE", "unique_patients" = "#003B5C")) +
  # labs(x = "Patient HPI Quartile", y = "Million") +
  # labs(x = "Prescriber HPI Quartile", y = "Million") +
  labs(x = "Pharmacy HPI Quartile", y = "Million") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.spacing = unit(2, "lines"),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        legend.key.width = unit(1.2, "cm"),
        legend.key.height = unit(1, "cm"),
        plot.subtitle = element_text(hjust = 1, vjust = 1)
  )

# ggsave("../Figs/Patient_HPI.pdf", width = 6, height = 5, dpi = 300)
# ggsave("../Figs/Prescriber_HPI.pdf", width = 6, height = 5, dpi = 300)
ggsave("../Figs/Pharmacy_HPI.pdf", width = 6, height = 5, dpi = 300)


################################################################################





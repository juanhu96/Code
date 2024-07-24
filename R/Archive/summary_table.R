### Create summary statistics for the dataset

library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

################################################################################
################################################################################
################################################################################

RX_2018 <- read.csv("../Data/RX_2018.csv")
RX_2018 <- RX_2018[which(RX_2018$class == 'Opioid'),]
write.csv(RX_2018, "../Data/OPIOID_2018.csv", row.names = FALSE)

RX_2019 <- read.csv("../Data/RX_2019.csv")
RX_2019 <- RX_2019[which(RX_2019$class == 'Opioid'),]
write.csv(RX_2019, "../Data/OPIOID_2019.csv", row.names = FALSE)

################################################################################
################################################################################
################################################################################

OPIOID_2018 <- read.csv("../Data/OPIOID_2018.csv")
OPIOID_2019 <- read.csv("../Data/OPIOID_2019.csv")

FULL_2018 <- FULL_2018 %>% select(c(patient_id, patient_gender, age, prescriber_id, pharmacy_id, 
                                    date_filled, daily_dose, quantity, days_supply, drug, payment, long_term, long_term_180))
FULL_2019 <- FULL_2019 %>% filter(patient_gender != 'U') %>% select(c(patient_id, patient_gender, age, prescriber_id, pharmacy_id, 
                                                                      date_filled, presc_until, daily_dose, quantity, days_supply, drug, payment, concurrent_benzo, long_term, long_term_180))

################################################################################
################################################################################
################################################################################


# FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv")
# FULL_2019 <- read.csv("../Data/FULL_2019_LONGTERM_UPTOFIRST.csv")
# TEMP <- FULL_2019 %>% filter(patient_gender == 'U')

FULL_2018 <- FULL_2018 %>% select(c(prescription_id, patient_id, patient_gender, age, prescriber_id, pharmacy_id, 
                                    date_filled, presc_until, daily_dose, quantity, days_supply, drug, payment, concurrent_benzo, long_term, long_term_180))
FULL_2019 <- FULL_2019 %>% filter(patient_gender != 'U') %>% select(c(prescription_id, patient_id, patient_gender, age, prescriber_id, pharmacy_id, 
                                    date_filled, presc_until, daily_dose, quantity, days_supply, drug, payment, concurrent_benzo, long_term, long_term_180))

FULL_ALL <- rbind(FULL_2018, FULL_2019)
rm(FULL_2018)
rm(FULL_2019)

################################################################################
################################################################################
################################################################################

c(round(mean(FULL_ALL$daily_dose),1), round(sd(FULL_ALL$daily_dose),1))
c(round(mean(FULL_ALL$quantity),1), round(sd(FULL_ALL$quantity),1))
c(round(mean(FULL_ALL$days_supply),1), round(sd(FULL_ALL$days_supply),1))

TEST1 <- FULL_ALL %>% filter(drug == 'Hydrocodone')
TEST2 <- FULL_ALL %>% filter(drug == 'Oxycodone')
TEST3 <- FULL_ALL %>% filter(drug == 'Codeine')
TEST4 <- FULL_ALL %>% filter(drug == 'Morphine')
TEST5 <- FULL_ALL %>% filter(drug == 'Hydromorphone')
TEST6 <- FULL_ALL %>% filter(drug == 'Methadone')
TEST7 <- FULL_ALL %>% filter(drug == 'Fentanyl')
TEST8 <- FULL_ALL %>% filter(drug == 'Oxymorphone')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4),
  nrow(TEST5), nrow(TEST6), nrow(TEST7), nrow(TEST8))

c(round(nrow(TEST1) / nrow(FULL_ALL)*100),
  round(nrow(TEST2) / nrow(FULL_ALL)*100),
  round(nrow(TEST3) / nrow(FULL_ALL)*100),
  round(nrow(TEST4) / nrow(FULL_ALL)*100),
  round(nrow(TEST5) / nrow(FULL_ALL)*100),
  round(nrow(TEST6) / nrow(FULL_ALL)*100),
  round(nrow(TEST7) / nrow(FULL_ALL)*100),
  round(nrow(TEST8) / nrow(FULL_ALL)*100))

TEST1 <- FULL_ALL %>% filter(payment == 'CommercialIns')
TEST2 <- FULL_ALL %>% filter(payment == 'CashCredit')
TEST3 <- FULL_ALL %>% filter(payment == 'Medicare')
TEST4 <- FULL_ALL %>% filter(payment == 'Medicaid')
TEST5 <- FULL_ALL %>% filter(payment == 'MilitaryIns')
TEST6 <- FULL_ALL %>% filter(payment == 'WorkersComp')
TEST7 <- FULL_ALL %>% filter(payment == 'Other' | payment == 'IndianNation')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4),
  nrow(TEST5), nrow(TEST6), nrow(TEST7))

c(round(nrow(TEST1) / nrow(FULL_ALL)*100),
  round(nrow(TEST2) / nrow(FULL_ALL)*100),
  round(nrow(TEST3) / nrow(FULL_ALL)*100),
  round(nrow(TEST4) / nrow(FULL_ALL)*100),
  round(nrow(TEST5) / nrow(FULL_ALL)*100),
  round(nrow(TEST6) / nrow(FULL_ALL)*100),
  round(nrow(TEST7) / nrow(FULL_ALL)*100))

PATIENT <- FULL_ALL %>% group_by(patient_id) %>% summarize(num_prescriptions = n(), 
                                                           age = age[1],
                                                           gender = patient_gender[1])
c(nrow(PATIENT[PATIENT$gender == 1 | PATIENT$gender == 'M', ]),
  round(nrow(PATIENT[PATIENT$gender == 1 | PATIENT$gender == 'M', ]) / nrow(PATIENT),2),
  nrow(PATIENT[PATIENT$gender == 0 | PATIENT$gender == 'F', ]),
  round(nrow(PATIENT[PATIENT$gender == 0 | PATIENT$gender == 'F', ]) / nrow(PATIENT),2))
round(c(mean(PATIENT$age), sd(PATIENT$age)),1)
round(c(mean(PATIENT$num_prescriptions), sd(PATIENT$num_prescriptions)),1)

BENZO <- FULL_ALL %>% filter(concurrent_benzo != 0) # forgot to include concurrent benzo when importing dataset
length(unique(BENZO$patient_id))
length(unique(BENZO$patient_id))/nrow(PATIENT)


################################################################################
################################################################################
################################################################################

FULL_NONLONGTERM <- FULL_ALL %>% filter(long_term_180 == 0)

round(c(mean(FULL_NONLONGTERM$daily_dose), sd(FULL_NONLONGTERM$daily_dose)),1)
round(c(mean(FULL_NONLONGTERM$quantity), sd(FULL_NONLONGTERM$quantity)),1)
round(c(mean(FULL_NONLONGTERM$days_supply), sd(FULL_NONLONGTERM$days_supply)),1)

TEST1 <- FULL_NONLONGTERM %>% filter(drug == 'Hydrocodone')
TEST2 <- FULL_NONLONGTERM %>% filter(drug == 'Oxycodone')
TEST3 <- FULL_NONLONGTERM %>% filter(drug == 'Codeine')
TEST4 <- FULL_NONLONGTERM %>% filter(drug == 'Morphine')
TEST5 <- FULL_NONLONGTERM %>% filter(drug == 'Hydromorphone')
TEST6 <- FULL_NONLONGTERM %>% filter(drug == 'Methadone')
TEST7 <- FULL_NONLONGTERM %>% filter(drug == 'Fentanyl')
TEST8 <- FULL_NONLONGTERM %>% filter(drug == 'Oxymorphone')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4), nrow(TEST5), nrow(TEST6), nrow(TEST7), nrow(TEST8))
c(round(nrow(TEST1) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST2) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST3) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST4) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST5) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST6) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST7) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST8) / nrow(FULL_NONLONGTERM)*100))

TEST1 <- FULL_NONLONGTERM %>% filter(payment == 'CommercialIns')
TEST2 <- FULL_NONLONGTERM %>% filter(payment == 'CashCredit')
TEST3 <- FULL_NONLONGTERM %>% filter(payment == 'Medicare')
TEST4 <- FULL_NONLONGTERM %>% filter(payment == 'Medicaid')
TEST5 <- FULL_NONLONGTERM %>% filter(payment == 'MilitaryIns')
TEST6 <- FULL_NONLONGTERM %>% filter(payment == 'WorkersComp')
TEST7 <- FULL_NONLONGTERM %>% filter(payment == 'Other' | payment == 'IndianNation')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4), nrow(TEST5), nrow(TEST6), nrow(TEST7))
c(round(nrow(TEST1) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST2) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST3) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST4) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST5) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST6) / nrow(FULL_NONLONGTERM)*100),
  round(nrow(TEST7) / nrow(FULL_NONLONGTERM)*100))

PATIENT <- FULL_ALL %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                           age = age[1],
                                                           first_presc_date = date_filled[1],
                                                           num_prescriptions = n(),
                                                           # longterm = ifelse(sum(long_term) > 0, 1, 0), # because we only keep up to first y = 1, should use longterm180
                                                           longterm = ifelse(sum(long_term_180) > 0, 1, 0), # because we only keep up to first y = 1, should use longterm180
                                                           longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                           longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA))
PATIENT_NONLONGTERM <- PATIENT %>% filter(longterm == 0)
c(nrow(PATIENT_NONLONGTERM[PATIENT_NONLONGTERM$patient_gender == 1 | PATIENT_NONLONGTERM$patient_gender == 'M', ]),
  nrow(PATIENT_NONLONGTERM[PATIENT_NONLONGTERM$patient_gender == 0 | PATIENT_NONLONGTERM$patient_gender == 'F', ]))

c(nrow(PATIENT_NONLONGTERM[PATIENT_NONLONGTERM$patient_gender == 1 | PATIENT_NONLONGTERM$patient_gender == 'M', ]) / nrow(PATIENT_NONLONGTERM),
  nrow(PATIENT_NONLONGTERM[PATIENT_NONLONGTERM$patient_gender == 0 | PATIENT_NONLONGTERM$patient_gender == 'F', ]) / nrow(PATIENT_NONLONGTERM))

round(c(mean(PATIENT_NONLONGTERM$age), sd(PATIENT_NONLONGTERM$age)),1)
round(c(mean(PATIENT_NONLONGTERM$num_prescriptions), sd(PATIENT_NONLONGTERM$num_prescriptions)),1)

BENZO_NONLONGTERM <- FULL_NONLONGTERM %>% filter(concurrent_benzo != 0) # forgot to include concurrent benzo when importing dataset
length(unique(BENZO_NONLONGTERM$patient_id))
length(unique(BENZO_NONLONGTERM$patient_id))/nrow(PATIENT_NONLONGTERM)

################################################################################
################################################################################
################################################################################

FULL_LONGTERM <- FULL_ALL %>% filter(long_term_180 == 1)

round(c(mean(FULL_LONGTERM$daily_dose), sd(FULL_LONGTERM$daily_dose)),1)
round(c(mean(FULL_LONGTERM$quantity), sd(FULL_LONGTERM$quantity)),1)
round(c(mean(FULL_LONGTERM$days_supply), sd(FULL_LONGTERM$days_supply)),1)

TEST1 <- FULL_LONGTERM %>% filter(drug == 'Hydrocodone')
TEST2 <- FULL_LONGTERM %>% filter(drug == 'Oxycodone')
TEST3 <- FULL_LONGTERM %>% filter(drug == 'Codeine')
TEST4 <- FULL_LONGTERM %>% filter(drug == 'Morphine')
TEST5 <- FULL_LONGTERM %>% filter(drug == 'Hydromorphone')
TEST6 <- FULL_LONGTERM %>% filter(drug == 'Methadone')
TEST7 <- FULL_LONGTERM %>% filter(drug == 'Fentanyl')
TEST8 <- FULL_LONGTERM %>% filter(drug == 'Oxymorphone')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4), nrow(TEST5), nrow(TEST6), nrow(TEST7), nrow(TEST8))
c(round(nrow(TEST1) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST2) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST3) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST4) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST5) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST6) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST7) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST8) / nrow(FULL_LONGTERM)*100))

TEST1 <- FULL_LONGTERM %>% filter(payment == 'CommercialIns')
TEST2 <- FULL_LONGTERM %>% filter(payment == 'CashCredit')
TEST3 <- FULL_LONGTERM %>% filter(payment == 'Medicare')
TEST4 <- FULL_LONGTERM %>% filter(payment == 'Medicaid')
TEST5 <- FULL_LONGTERM %>% filter(payment == 'MilitaryIns')
TEST6 <- FULL_LONGTERM %>% filter(payment == 'WorkersComp')
TEST7 <- FULL_LONGTERM %>% filter(payment == 'Other' | payment == 'IndianNation')

c(nrow(TEST1), nrow(TEST2), nrow(TEST3), nrow(TEST4), nrow(TEST5), nrow(TEST6), nrow(TEST7))
c(round(nrow(TEST1) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST2) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST3) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST4) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST5) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST6) / nrow(FULL_LONGTERM)*100),
  round(nrow(TEST7) / nrow(FULL_LONGTERM)*100))

PATIENT_LONGTERM <- PATIENT %>% filter(longterm == 1)

c(nrow(PATIENT_LONGTERM[PATIENT_LONGTERM$patient_gender == 1 | PATIENT_LONGTERM$patient_gender == 'M', ]) / nrow(PATIENT_LONGTERM),
  nrow(PATIENT_LONGTERM[PATIENT_LONGTERM$patient_gender == 0 | PATIENT_LONGTERM$patient_gender == 'F', ]) / nrow(PATIENT_LONGTERM))
c(nrow(PATIENT_LONGTERM[PATIENT_LONGTERM$patient_gender == 1 | PATIENT_LONGTERM$patient_gender == 'M', ]),
  nrow(PATIENT_LONGTERM[PATIENT_LONGTERM$patient_gender == 0 | PATIENT_LONGTERM$patient_gender == 'F', ]))

round(c(mean(PATIENT_LONGTERM$age), sd(PATIENT_LONGTERM$age)),1)
round(c(mean(PATIENT_LONGTERM$num_prescriptions), sd(PATIENT_LONGTERM$num_prescriptions)),1)

BENZO_LONGTERM <- FULL_LONGTERM %>% filter(concurrent_benzo != 0) # forgot to include concurrent benzo when importing dataset
length(unique(BENZO_LONGTERM$patient_id))
length(unique(BENZO_LONGTERM$patient_id))/nrow(PATIENT_LONGTERM)

################################################################################
################################################################################
################################################################################

PATIENT <- FULL_ALL %>% group_by(patient_id) %>% summarize(patient_gender = patient_gender[1],
                                                           age = age[1],
                                                           first_presc_date = date_filled[1],
                                                           num_prescriptions = n(),
                                                           longterm = ifelse(sum(long_term) > 0, 1, 0),
                                                           longterm_180 = ifelse(sum(long_term_180) > 0, 1, 0),
                                                           longterm_filled_date = ifelse(sum(long_term) > 0, date_filled[long_term > 0][1], NA),
                                                           longterm_presc_date = ifelse(sum(long_term) > 0, presc_until[long_term > 0][1], NA))

WEIRD <- PATIENT %>% filter(longterm_180 == 1 & longterm == 0)
PAT <- FULL_ALL %>% filter(patient_id == 1293510) # only two prescription in 2019, why longterm_180 == 1? maybe we drop last two month?
PAT <- OPIOID_2019 %>% filter(patient_id == 1293510) # it's because we drop the gender == 'U'

FULL_ALL <- left_join(FULL_ALL, PATIENT[ , c("patient_id", "longterm_180")], by="patient_id")

PATIENT_NONLONGTERM <- PATIENT %>% filter(longterm_180 == 0)
PATIENT_LONGTERM <- PATIENT %>% filter(longterm_180 == 1)
################################################################################
################################################################################
################################################################################

FULL_NONLONGTERM <- FULL_ALL %>% filter(longterm_180 == 0)
FULL_LONGTERM <- FULL_ALL %>% filter(longterm_180 == 1)

################################################################################
################################################################################
################################################################################
FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv")
colnames(FULL_2018)
TEST <- FULL_2018 %>% mutate(highrisk = ifelse(alert1 + alert2 + alert3 + alert4 + alert5 + alert6 > 0, 1, 0))

HIGHRISK <- TEST %>% filter(highrisk == 1) # 986209
LONGTERM <- TEST %>% filter(long_term_180 == 1) # 103721

HIGHRISK_LONGTERM <- TEST %>% filter(highrisk == 1 & long_term_180 == 1) # 36501



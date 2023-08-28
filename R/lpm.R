# Linear probability model

library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv")
TEST <- FULL_2018[1:200,]

################################################################################
################################################################################
################################################################################

# compute_drug_payment <- function(pat_id, presc_id){
#   
#   PATIENT_PRESC_OPIOIDS <- TEST[which(TEST$prescription_id == presc_id),]
#   
#   drug = ""
#   payment = ""
#   
#   if(PATIENT_PRESC_OPIOIDS$drug == "Codeine"){drug = "Codeine"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Hydrocodone"){drug = "Hydrocodone"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Oxycodone"){drug = "Oxycodone"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Morphine"){drug = "Morphine"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Hydromorphone"){drug = "Hydromorphone"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Methadone"){drug = "Methadone"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Fentanyl"){drug = "Fentanyl"}
#   if(PATIENT_PRESC_OPIOIDS$drug == "Oxymorphone"){drug = "Oxymorphone"}
#   
#   if(PATIENT_PRESC_OPIOIDS$payment == "Medicaid"){payment = "Medicaid"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "CommercialIns"){payment = "CommercialIns"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "Medicare"){payment = "Medicare"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "CashCredit"){payment = "CashCredit"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "MilitaryIns"){payment = "MilitaryIns"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "WorkersComp"){payment = "WorkersComp"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "Other"){payment = "Other"}
#   if(PATIENT_PRESC_OPIOIDS$payment == "IndianNation"){payment = "IndianNation"}
#   
#   
#   return (c(drug, payment))
# }
# 
# results <- mcmapply(compute_drug_payment, TEST$patient_id, TEST$prescription_id, mc.cores=40)
# TEST$drug = results[1, ]
# TEST$payment = results[2, ]

################################################################################
################################################################################
################################################################################

FULL_2018_INPUT <- FULL_2018 %>% select(-c(prescription_id, patient_id, patient_birth_year, patient_zip, prescriber_id,
                                      prescriber_zip, pharmacy_id, pharmacy_zip, strength, date_filled, presc_until, 
                                      quantity, days_supply, quantity_per_day, daily_dose, total_dose, 
                                      concurrent_benzo_same, concurrent_benzo_diff, 
                                      conversion, class, chronic, PRODUCTNDC, PROPRIETARYNAME, LABELERNAME, ROUTENAME,
                                      DEASCHEDULE, MAINDOSE, prescription_year,
                                      overlap, alert1, alert2, alert3, alert4, alert5, alert6,
                                      days_to_long_term, num_alert, opioid_days, long_term, long_term_180_date,
                                      first_switch_drug, first_switch_payment, avgMME, avgDays, HMFO, 
                                      drug_payment, Codeine, Hydrocodone, Oxycodone, Morphine, Hydromorphone, Methadone, Fentanyl, Oxymorphone,
                                      Medicaid, CommercialIns, Medicare, CashCredit, MilitaryIns, WorkersComp, Other, IndianNation,
                                      Codeine_MME, Hydrocodone_MME, Oxycodone_MME, Morphine_MME,
                                      Hydromorphone_MME, Methadone_MME, Fentanyl_MME, Oxymorphone_MME, Codeine_Medicaid, Codeine_CommercialIns,
                                      Codeine_Medicare, Codeine_CashCredit, Codeine_MilitaryIns, Codeine_WorkersComp, Codeine_Other,
                                      Codeine_IndianNation, Hydrocodone_Medicaid, Hydrocodone_CommercialIns, Hydrocodone_Medicare,
                                      Hydrocodone_CashCredit, Hydrocodone_MilitaryIns, Hydrocodone_WorkersComp, Hydrocodone_Other,
                                      Hydrocodone_IndianNation, Oxycodone_Medicaid, Oxycodone_CommercialIns, Oxycodone_Medicare,
                                      Oxycodone_CashCredit, Oxycodone_MilitaryIns, Oxycodone_WorkersComp, Oxycodone_Other,
                                      Oxycodone_IndianNation, Morphine_Medicaid, Morphine_CommercialIns, Morphine_Medicare,
                                      Morphine_CashCredit, Morphine_MilitaryIns, Morphine_WorkersComp, Morphine_Other,
                                      Morphine_IndianNation, Hydromorphone_Medicaid, Hydromorphone_CommercialIns, Hydromorphone_Medicare,
                                      Hydromorphone_CashCredit, Hydromorphone_MilitaryIns, Hydromorphone_WorkersComp, Hydromorphone_Other,
                                      Hydromorphone_IndianNation, Methadone_Medicaid, Methadone_CommercialIns, Methadone_Medicare,
                                      Methadone_CashCredit, Methadone_MilitaryIns, Methadone_WorkersComp, Methadone_Other, Methadone_IndianNation,
                                      Fentanyl_Medicaid, Fentanyl_CommercialIns, Fentanyl_Medicare, Fentanyl_CashCredit, Fentanyl_MilitaryIns,
                                      Fentanyl_WorkersComp, Fentanyl_Other, Fentanyl_IndianNation, Oxymorphone_Medicaid, Oxymorphone_CommercialIns,
                                      Oxymorphone_Medicare, Oxymorphone_CashCredit, Oxymorphone_MilitaryIns, Oxymorphone_WorkersComp, Oxymorphone_Other,
                                      Oxymorphone_IndianNation))

################################################################################
################################################################################
################################################################################

# lpm <- lm(long_term_180 ~ ., data=FULL_2018_INPUT)
# summary(lpm)

lpm <- lm(long_term_180 ~ ., data=FULL_2018_INPUT)
stargazer(lpm, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Long term use"))
stargazer(lpm, digits=3, type="text", no.space=TRUE, dep.var.labels=c("Long term use"))
lpm_cluster = coeftest(lpm, vcov = vcovHC(lpm, type = "HC0", cluster = "patient_id"))
stargazer(lpm_cluster, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Long term use"))
stargazer(lpm_cluster, digits=3, type="text", no.space=TRUE, dep.var.labels=c("Long term use"))
lpm_cluster

## Use the most common group as baseline
FULL_2018_INPUT$drug <- relevel(factor(FULL_2018_INPUT$drug), ref = "Hydrocodone")
FULL_2018_INPUT$payment <- relevel(factor(FULL_2018_INPUT$payment), ref = "CommercialIns")




### Logistic with binary?
lpm_glm <- glm(long_term_180 ~ ., data=FULL_2018_INPUT, family = binomial(link = "logit"))
lpm_glm_cluster = coeftest(lpm_glm, vcov = vcovHC(lpm_glm, type = "HC0", cluster = "patient_id"))
lpm_glm_cluster



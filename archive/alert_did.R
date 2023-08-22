### The effects of alert for the full dataset
library(dplyr)
library(ggplot2)
# Regression Output
library(stargazer)
library(export)
# Robust standard error
library(lmtest)
library(sandwich)
# Fitted data
library(broom)
# Fast fixed-effects
library(fixest)

setwd("/mnt/phd/jihu/opioid/Code")
FULL_2018 <- read.csv("../Data/FULL_2018_ALERT.csv")

################################################################################
################################################################################
################################################################################
### Alert as absorbing state
# What if two prescriptions are on the same date
FULL_2018 <- FULL_2018 %>% mutate(alert = ifelse(num_alert > 0, 1, 0))
PATIENT_ALERT <- FULL_2018 %>% group_by(patient_id) %>% 
  summarize(num_prescriptions = n(),
            ever_alert1 = ifelse(sum(alert1) > 0, 1, 0),
            ever_alert2 = ifelse(sum(alert2) > 0, 1, 0),
            ever_alert3 = ifelse(sum(alert3) > 0, 1, 0),
            ever_alert4 = ifelse(sum(alert4) > 0, 1, 0),
            ever_alert5 = ifelse(sum(alert5) > 0, 1, 0),
            ever_alert6 = ifelse(sum(alert6) > 0, 1, 0),
            ever_alert = ifelse(sum(alert) > 0, 1, 0),
            alert1_date = ifelse(ever_alert1 == 1, date_filled[alert1 == 1][1], NA),
            alert2_date = ifelse(ever_alert2 == 1, date_filled[alert2 == 1][1], NA),
            alert3_date = ifelse(ever_alert3 == 1, date_filled[alert3 == 1][1], NA),
            alert4_date = ifelse(ever_alert4 == 1, date_filled[alert4 == 1][1], NA),
            alert5_date = ifelse(ever_alert5 == 1, date_filled[alert5 == 1][1], NA),
            alert6_date = ifelse(ever_alert6 == 1, date_filled[alert6 == 1][1], NA),
            alert_date = ifelse(ever_alert == 1, date_filled[alert6 == 1][1], NA))
FULL_2018 <- left_join(FULL_2018, PATIENT_ALERT, by = "patient_id")
rm(PATIENT_ALERT)

FULL_2018 <- FULL_2018 %>% 
  mutate(days = as.numeric(as.Date(date_filled,format = "%m/%d/%Y") - as.Date("04/01/2016", format = "%m/%d/%Y"))) %>%
  mutate(prior_alert1 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert1_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert2 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert2_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert3 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert3_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert4 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert4_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert5 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert5_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert6 = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert6_date,format = "%m/%d/%Y"), 1, 0),
         prior_alert = ifelse(as.Date(date_filled,format = "%m/%d/%Y") > as.Date(alert_date,format = "%m/%d/%Y"), 1, 0)) %>%
  mutate(prior_alert1 = coalesce(prior_alert1, 0),
         prior_alert2 = coalesce(prior_alert2, 0),
         prior_alert3 = coalesce(prior_alert3, 0),
         prior_alert4 = coalesce(prior_alert4, 0),
         prior_alert5 = coalesce(prior_alert5, 0),
         prior_alert6 = coalesce(prior_alert6, 0),
         prior_alert = coalesce(prior_alert, 0)) %>%
  dplyr::select(c(prescription_id, patient_id, patient_birth_year, age, patient_gender,
                  prescriber_id, pharmacy_id, strength, quantity, days_supply,
                  date_filled, presc_until, quantity_per_day, conversion, class, drug, daily_dose, total_dose,
                  chronic, payment, prescription_month, prescription_quarter, prescription_year,
                  concurrent_MME, concurrent_MME_methadone, presence_MME, presence_MME_methadone, 
                  presence_num_prescribers, presence_num_pharmacies, num_prescribers, num_pharmacies, 
                  consecutive_days, presence_days, days, 
                  ever_alert1, ever_alert2, ever_alert3, ever_alert4, ever_alert5, ever_alert6, ever_alert,
                  prior_alert1, prior_alert2, prior_alert3, prior_alert4, prior_alert5, prior_alert6, prior_alert))

### Alert as on-off state
### If the patient is still on alert before the current prescription
FULL_2018 <- FULL_2018 %>% mutate(active_alert1 = ifelse(presence_MME >= 90, 1, 0),
                            active_alert2 = ifelse(presence_MME_methadone >= 40, 1, 0),
                            active_alert3 = ifelse(presence_num_prescribers >= 6, 1, 0),
                            active_alert4 = ifelse(presence_num_pharmacies >= 6, 1, 0),
                            active_alert5 = ifelse(presence_days >= 90, 1, 0))

FULL_2018 <- FULL_2018 %>% dplyr::select(c(daily_dose, days_supply, active_alert1, prior_alert1, presence_MME, 
                                           active_alert5, prior_alert5, presence_days, age, drug, patient_gender, patient_id, prescriber_id))

################################################################################

### MME vs. alert 1
MME_ALERT1_BASE <- lm(daily_dose ~ active_alert1 + prior_alert1 + presence_MME, data = FULL_2018)
stargazer(MME_ALERT1_BASE, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))
MME_ALERT1_BASE_CLUSTER = coeftest(MME_ALERT1_BASE, vcov = vcovHC(MME_ALERT1_BASE, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_BASE_CLUSTER, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + age + drug + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1*drug + prior_alert1 + presence_MME + age + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id)


### Days of supply vs. alert 5
DAYS_ALERT5_BASE <- lm(days_supply ~ active_alert5 + prior_alert5 + presence_days, data = FULL_2018)
stargazer(DAYS_ALERT5_BASE, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Days of supply"))
DAYS_ALERT5_BASE_CLUSTER = coeftest(DAYS_ALERT5_BASE, vcov = vcovHC(DAYS_ALERT5_BASE, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT5_BASE_CLUSTER, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Days of supply"))

DAYS_ALERT5_BASE <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days, FULL_2018)
etable(DAYS_ALERT5_BASE, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_FULL <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days + age + drug + patient_gender | prescriber_id, FULL_2018)
etable(DAYS_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_INTERACT <- feols(days_supply ~ active_alert5*drug + prior_alert5 + presence_days + age + patient_gender | prescriber_id, FULL_2018)
etable(DAYS_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)


##### Cross interaction #####
# Also need to print table to get the adjusted R2

### MME vs. alert 5
MME_ALERT5_BASE <- lm(daily_dose ~ active_alert5 + prior_alert5 + presence_MME+ presence_days, data = FULL_2018)
stargazer(MME_ALERT5_BASE, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))
MME_ALERT5_BASE_CLUSTER = coeftest(MME_ALERT5_BASE, vcov = vcovHC(MME_ALERT5_BASE, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT5_BASE_CLUSTER, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))

MME_ALERT5_FULL <- feols(daily_dose ~ active_alert5 + prior_alert5 + presence_MME + presence_days + age + drug + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT5_INTERACT <- feols(daily_dose ~ active_alert5*drug + prior_alert5 + presence_MME + presence_days + age + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)


### Days of supply vs. alert 1
DAYS_ALERT1_BASE <- lm(days_supply ~ active_alert5 + prior_alert5 + presence_MME+ presence_days, data = FULL_2018)
stargazer(DAYS_ALERT1_BASE, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))
DAYS_ALERT1_BASE_CLUSTER = coeftest(DAYS_ALERT1_BASE, vcov = vcovHC(DAYS_ALERT1_BASE, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT1_BASE_CLUSTER, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))

DAYS_ALERT1_FULL <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_MME + presence_days + age + drug + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT1_INTERACT <- feols(days_supply ~ active_alert5*drug + prior_alert5 + presence_MME + presence_days + age + patient_gender | prescriber_id, FULL_2018)
etable(MME_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)





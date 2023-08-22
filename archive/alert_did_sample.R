### The effects of alert for the full dataset
library(dplyr)
library(ggplot2)
# Regression Output
library(stargazer)
# Robust standard error
library(lmtest)
library(sandwich)
# Fitted data
library(broom)
# Parallel
library(partools)
#library(parallel) ## loads as dependency

setwd("/mnt/phd/jihu/opioid/Code")
SAMPLE <- read.csv("../Data/SAMPLE_ALERT.csv")

################################################################################
################################################################################
################################################################################
### Alert as absorbing state
# What if two prescriptions are on the same date
SAMPLE <- SAMPLE %>% mutate(alert = ifelse(num_alert > 0, 1, 0))
PATIENT_ALERT <- SAMPLE %>% group_by(patient_id) %>% 
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
SAMPLE <- left_join(SAMPLE, PATIENT_ALERT, by = "patient_id")
rm(PATIENT_ALERT)

SAMPLE <- SAMPLE %>% 
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
  select(c(prescription_id, patient_id, patient_birth_year, age, patient_gender,
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
SAMPLE <- SAMPLE %>% mutate(active_alert1 = ifelse(presence_MME >= 90, 1, 0),
                            active_alert2 = ifelse(presence_MME_methadone >= 40, 1, 0),
                            active_alert3 = ifelse(presence_num_prescribers >= 6, 1, 0),
                            active_alert4 = ifelse(presence_num_pharmacies >= 6, 1, 0),
                            active_alert5 = ifelse(presence_days >= 90, 1, 0))

################################################################################
################################################################################
################################################################################
### Difference-in-difference model

# Group: if ever alerted or not
# Treatment(cutoff): has alerted or not
# Fixed-effects: prescriber, drug type, payment, age, gender, etc.

################################################################################
### Concurrent MME vs. Alert 1 ###

## Same slope
CONCURRENT_MME_ALERT1 = lm(concurrent_MME ~ days + ever_alert1 + flag_alert1 + presence_MME + factor(prescriber_id) +
                             factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(CONCURRENT_MME_ALERT1, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age")) # for R squared etc.
CONCURRENT_MME_ALERT1_CLUSTER = coeftest(CONCURRENT_MME_ALERT1, vcov = vcovHC(CONCURRENT_MME_ALERT1, type = "HC0", cluster = "patient_id"))
stargazer(CONCURRENT_MME_ALERT1_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Concurrent MME of a patient"))

## Different slope
CONCURRENT_MME_ALERT1_SLOPE = lm(concurrent_MME ~ days*flag_alert1 + ever_alert1 + presence_MME + factor(prescriber_id) +
                             factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(CONCURRENT_MME_ALERT1_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"))
CONCURRENT_MME_ALERT1_SLOPE_CLUSTER = coeftest(CONCURRENT_MME_ALERT1_SLOPE, vcov = vcovHC(CONCURRENT_MME_ALERT1_SLOPE, type = "HC0", cluster = "patient_id"))
stargazer(CONCURRENT_MME_ALERT1_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Concurrent MME of a patient"))

################################################################################
### MME vs. Alert 1 ###

## Same slope
MME_ALERT1 = lm(daily_dose ~ days + ever_alert1 + flag_alert1 + presence_MME + factor(prescriber_id) +
                  factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(MME_ALERT1, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))
MME_ALERT1_CLUSTER = coeftest(MME_ALERT1, vcov = vcovHC(MME_ALERT1, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))

## Different slope
MME_ALERT1_SLOPE = lm(daily_dose ~ days*flag_alert1 + ever_alert1 + presence_MME + factor(prescriber_id) +
                  factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(MME_ALERT1_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))
MME_ALERT1_SLOPE_CLUSTER = coeftest(MME_ALERT1_SLOPE, vcov = vcovHC(MME_ALERT1_SLOPE, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))

################################################################################
## Days supply vs. Alert 5 (consecutive days) ###

## Same slope
DAYS_ALERT5 = lm(days_supply ~ days + ever_alert5 + flag_alert5 + presence_days + factor(prescriber_id) +
                   factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(DAYS_ALERT5, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))
DAYS_ALERT5_CLUSTER = coeftest(DAYS_ALERT5, vcov = vcovHC(DAYS_ALERT5, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT5_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))


## Different slope
DAYS_ALERT5_SLOPE = lm(days_supply ~ days*flag_alert5 + ever_alert5 + presence_days + factor(prescriber_id) +
                   factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(DAYS_ALERT5_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))
DAYS_ALERT5_SLOPE_CLUSTER = coeftest(DAYS_ALERT5_SLOPE, vcov = vcovHC(DAYS_ALERT5_SLOPE, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT5_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))

################################################################################
################################################################################
################################################################################

### Concurrent MME vs. Alert 1 (Dynamic) ###
## Same slope
CONCURRENT_MME_ALERT1 = lm(concurrent_MME ~ days + ever_alert1 + flag_alert1_new + presence_MME + factor(prescriber_id) +
                             factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(CONCURRENT_MME_ALERT1, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age")) # for R squared etc.
# CONCURRENT_MME_ALERT1_CLUSTER = coeftest(CONCURRENT_MME_ALERT1, vcov = vcovHC(CONCURRENT_MME_ALERT1, type = "HC0", cluster = "patient_id"))
# stargazer(CONCURRENT_MME_ALERT1_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Concurrent MME of a patient"))

## Different slope
CONCURRENT_MME_ALERT1_SLOPE = lm(concurrent_MME ~ days*flag_alert1_new + ever_alert1 + presence_MME + factor(prescriber_id) +
                                   factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(CONCURRENT_MME_ALERT1_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"))
# CONCURRENT_MME_ALERT1_SLOPE_CLUSTER = coeftest(CONCURRENT_MME_ALERT1_SLOPE, vcov = vcovHC(CONCURRENT_MME_ALERT1_SLOPE, type = "HC0", cluster = "patient_id"))
# stargazer(CONCURRENT_MME_ALERT1_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Concurrent MME of a patient"))


### MME vs. Alert 1 (dynamic) ###
## Same slope
MME_ALERT1 = lm(daily_dose ~ days + ever_alert1 + flag_alert1_new + presence_MME + factor(prescriber_id) +
                  factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(MME_ALERT1, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))
# MME_ALERT1_CLUSTER = coeftest(MME_ALERT1, vcov = vcovHC(MME_ALERT1, type = "HC0", cluster = "patient_id"))
# stargazer(MME_ALERT1_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))

## Different slope
MME_ALERT1_SLOPE = lm(daily_dose ~ days*flag_alert1_new + ever_alert1 + presence_MME + factor(prescriber_id) +
                        factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(MME_ALERT1_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))
# MME_ALERT1_SLOPE_CLUSTER = coeftest(MME_ALERT1_SLOPE, vcov = vcovHC(MME_ALERT1_SLOPE, type = "HC0", cluster = "patient_id"))
# stargazer(MME_ALERT1_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Daily dose (MME)"))


### Days supply vs. Alert 5 (dynamic) ###
## Same slope
DAYS_ALERT5 = lm(days_supply ~ days + ever_alert5 + flag_alert5_new + presence_days + factor(prescriber_id) +
                   factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(DAYS_ALERT5, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))
# DAYS_ALERT5_CLUSTER = coeftest(DAYS_ALERT5, vcov = vcovHC(DAYS_ALERT5, type = "HC0", cluster = "patient_id"))
# stargazer(DAYS_ALERT5_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))

## Different slope
DAYS_ALERT5_SLOPE = lm(days_supply ~ days*flag_alert5_new + ever_alert5 + presence_days + factor(prescriber_id) +
                         factor(drug) + factor(patient_gender) + age, data = SAMPLE)
stargazer(DAYS_ALERT5_SLOPE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit = c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))
# DAYS_ALERT5_SLOPE_CLUSTER = coeftest(DAYS_ALERT5_SLOPE, vcov = vcovHC(DAYS_ALERT5_SLOPE, type = "HC0", cluster = "patient_id"))
# stargazer(DAYS_ALERT5_SLOPE_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
#           omit=c("prescriber_id", "drug", "patient_gender", "age"), dep.var.labels = c("Days of supply"))


################################################################################
################################################################################
################################################################################

### MME vs. Alert 1 ###

### Same slope
## Basic model
MME_ALERT1_BASE <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME, data = SAMPLE)
stargazer(MME_ALERT1_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_BASE_SAME.tex")
MME_ALERT1_BASE_CLUSTER = coeftest(MME_ALERT1_BASE, vcov = vcovHC(MME_ALERT1_BASE, type = "HC0", cluster = "patient_id"))

## Full model
MME_ALERT1_FULL <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME + 
                        factor(patient_gender) + age + age^2 + factor(prescriber_id) + factor(drug), data = SAMPLE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_FULL_SAME.tex")
MME_ALERT1_FULL_CLUSTER = coeftest(MME_ALERT1_FULL, vcov = vcovHC(MME_ALERT1_FULL, type = "HC0", cluster = "patient_id"))

## Combine
stargazer(MME_ALERT1_BASE_CLUSTER, MME_ALERT1_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_CLUSTER_SAME.tex")

### Different slope
## Basic model
MME_ALERT1_BASE <- lm(daily_dose ~ days*active_alert1 + prior_alert1 + presence_MME, data = SAMPLE)
stargazer(MME_ALERT1_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_BASE_DIFF.tex")
MME_ALERT1_BASE_CLUSTER = coeftest(MME_ALERT1_BASE, vcov = vcovHC(MME_ALERT1_BASE, type = "HC0", cluster = "patient_id"))

## Full model
MME_ALERT1_FULL <- lm(daily_dose ~ days*active_alert1 + prior_alert1 + presence_MME + 
                        factor(patient_gender) + age + age^2 + factor(prescriber_id) + factor(drug), data = SAMPLE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_FULL_DIFF.tex")
MME_ALERT1_FULL_CLUSTER = coeftest(MME_ALERT1_FULL, vcov = vcovHC(MME_ALERT1_FULL, type = "HC0", cluster = "patient_id"))

## Combine
stargazer(MME_ALERT1_BASE_CLUSTER, MME_ALERT1_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_CLUSTER_DIFF.tex")

################################################################################
################################################################################

### Days of supply vs. Alert 5 ###

### Same slope
## Basic model
DAYS_ALERT5_BASE <- lm(days_supply ~ days + active_alert5 + prior_alert5 + presence_days, data = SAMPLE)
stargazer(DAYS_ALERT5_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_BASE_SAME.tex")
DAYS_ALERT5_BASE_CLUSTER = coeftest(DAYS_ALERT5_BASE, vcov = vcovHC(DAYS_ALERT5_BASE, type = "HC0", cluster = "patient_id"))

## Full model
DAYS_ALERT5_FULL <- lm(days_supply ~ days + active_alert5 + prior_alert5 + presence_days + 
                        factor(patient_gender) + age + age^2 + factor(prescriber_id) + factor(drug), data = SAMPLE)
stargazer(DAYS_ALERT5_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_FULL_SAME.tex")
DAYS_ALERT5_FULL_CLUSTER = coeftest(DAYS_ALERT5_FULL, vcov = vcovHC(DAYS_ALERT5_FULL, type = "HC0", cluster = "patient_id"))

## Combine
stargazer(DAYS_ALERT5_BASE_CLUSTER, DAYS_ALERT5_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_CLUSTER_SAME.tex")


### Different slope
## Basic model
DAYS_ALERT5_BASE <- lm(days_supply ~ days*active_alert5 + prior_alert5 + presence_days, data = SAMPLE)
stargazer(DAYS_ALERT5_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_BASE_DIFF.tex")
DAYS_ALERT5_BASE_CLUSTER = coeftest(DAYS_ALERT5_BASE, vcov = vcovHC(DAYS_ALERT5_BASE, type = "HC0", cluster = "patient_id"))

## Full model
DAYS_ALERT5_FULL <- lm(days_supply ~ days*active_alert5 + prior_alert5 + presence_days + 
                         factor(patient_gender) + age + age^2 + factor(prescriber_id) + factor(drug), data = SAMPLE)
stargazer(DAYS_ALERT5_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_FULL_DIFF.tex")
DAYS_ALERT5_FULL_CLUSTER = coeftest(DAYS_ALERT5_FULL, vcov = vcovHC(DAYS_ALERT5_FULL, type = "HC0", cluster = "patient_id"))

## Combine
stargazer(DAYS_ALERT5_BASE_CLUSTER, DAYS_ALERT5_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_CLUSTER_DIFF.tex")

################################################################################
################################################################################
################################################################################
#### Drug specific effects (normal standard errors, just for sanity check) ####
################################################################################

### Hydrocodone ###
SAMPLE_HYDROCODONE <- SAMPLE %>% filter(drug == "Hydrocodone") # 13513 out of 21671

# Base
MME_ALERT1_BASE <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME, data = SAMPLE_HYDROCODONE)
stargazer(MME_ALERT1_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"))
# Full
MME_ALERT1_FULL <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME + 
                        factor(patient_gender) + age + age^2 + factor(prescriber_id), data = SAMPLE_HYDROCODONE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"))


### Codeine ###
SAMPLE_CODEINE <- SAMPLE %>% filter(drug == "Codeine") # 4212 out of 21671

# Base
MME_ALERT1_BASE <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME, data = SAMPLE_CODEINE)
stargazer(MME_ALERT1_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"))
# Full
MME_ALERT1_FULL <- lm(daily_dose ~ days + active_alert1 + prior_alert1 + presence_MME + 
                        factor(patient_gender) + age + age^2 + factor(prescriber_id), data = SAMPLE_CODEINE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"))


################################################################################
################################################################################
################################################################################

### MME vs. Alert 1 ###

## Base
MME_ALERT1_BASE <- lm(daily_dose ~ active_alert1 + prior_alert1 + presence_MME, data = SAMPLE)
stargazer(MME_ALERT1_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))

### Same slope
MME_ALERT1_FULL <- lm(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + factor(patient_gender) + age + factor(drug) + factor(prescriber_id), data = SAMPLE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_FULL_SAME.tex")
MME_ALERT1_FULL_CLUSTER = coeftest(MME_ALERT1_FULL, vcov = vcovHC(MME_ALERT1_FULL, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_CLUSTER_SAME.tex")

### Different slope
MME_ALERT1_FULL <- lm(daily_dose ~ prior_alert1 + presence_MME + factor(patient_gender) + age + factor(drug)*active_alert1 + factor(prescriber_id), data = SAMPLE)
stargazer(MME_ALERT1_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_FULL_DIFF.tex")
MME_ALERT1_FULL_CLUSTER = coeftest(MME_ALERT1_FULL, vcov = vcovHC(MME_ALERT1_FULL, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Daily dose (MME)"), out = "../Data/MME_ALERT1_CLUSTER_DIFF.tex")


### Days of supply vs. Alert 5 ###

## Base
DAYS_ALERT5_BASE <- lm(days_supply ~ active_alert5 + prior_alert5 + presence_days, data = SAMPLE)
stargazer(DAYS_ALERT5_BASE, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Days of supply"))

### Same slope
DAYS_ALERT5_FULL <- lm(days_supply ~ active_alert5 + prior_alert5 + presence_days + factor(patient_gender) + age + factor(drug) + factor(prescriber_id), data = SAMPLE)
stargazer(DAYS_ALERT5_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_FULL_SAME.tex")
DAYS_ALERT5_FULL_CLUSTER = coeftest(DAYS_ALERT5_FULL, vcov = vcovHC(DAYS_ALERT5_FULL, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT5_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_CLUSTER_SAME.tex")

### Different slope
DAYS_ALERT5_FULL <- lm(days_supply ~ prior_alert5 + presence_days + factor(patient_gender) + age + factor(drug)*active_alert5 + factor(prescriber_id), data = SAMPLE)
stargazer(DAYS_ALERT5_FULL, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_FULL_DIFF.tex")
DAYS_ALERT5_FULL_CLUSTER = coeftest(DAYS_ALERT5_FULL, vcov = vcovHC(DAYS_ALERT5_FULL, type = "HC0", cluster = "patient_id"))
stargazer(DAYS_ALERT5_FULL_CLUSTER, star.cutoffs=c(0.01, 0.001, 0.0001), digits=3, type="latex", no.space=TRUE,
          omit=c("prescriber_id"), dep.var.labels=c("Days of supply"), out = "../Data/DAYS_ALERT5_CLUSTER_DIFF.tex")








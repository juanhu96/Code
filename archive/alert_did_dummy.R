### Alert DiD with dummy variable and pseudo alerts

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
FULL_2018_1 <- read.csv("../Data/FULL_DUMMY_2018_1.csv")
FULL_2018_2 <- read.csv("../Data/FULL_DUMMY_2018_2.csv")
FULL_2018_3 <- read.csv("../Data/FULL_DUMMY_2018_3.csv")
FULL_2018_4 <- read.csv("../Data/FULL_DUMMY_2018_4.csv")
FULL_2018 <- rbind(FULL_2018_1, FULL_2018_2)
FULL_2018 <- rbind(FULL_2018, FULL_2018_3)
FULL_2018 <- rbind(FULL_2018, FULL_2018_4)
rm(FULL_2018_1)
rm(FULL_2018_2)
rm(FULL_2018_3)
rm(FULL_2018_4)

################################################################################
############################# Dummy Prescriptions ##############################
################################################################################

### MME vs. alert 1
MME_ALERT1_BASE <- feols(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + days_supply, FULL_2018)
etable(MME_ALERT1_BASE, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1*drug + prior_alert1 + presence_MME + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)


### Days of supply vs. alert 5
DAYS_ALERT5_BASE <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days + daily_dose, FULL_2018)
etable(DAYS_ALERT5_BASE, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_FULL <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days + daily_dose + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_FULL)

DAYS_ALERT5_INTERACT <- feols(days_supply ~ active_alert5*drug + prior_alert5 + presence_days + daily_dose + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_INTERACT)

### Quantity vs. alert 1
QUANT_ALERT1_BASE <- feols(quantity ~ active_alert1 + prior_alert1 + presence_MME + days_supply, FULL_2018)
etable(QUANT_ALERT1_BASE, cluster = ~patient_id)

QUANT_ALERT1_FULL <- feols(quantity ~ active_alert1 + prior_alert1 + presence_MME + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(QUANT_ALERT1_FULL, cluster = ~patient_id)

QUANT_ALERT1_INTERACT <- feols(quantity ~ active_alert1*drug + prior_alert1 + presence_MME + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(QUANT_ALERT1_INTERACT, cluster = ~patient_id)

################################################################################
################################ Time Window ###################################
################################################################################
FULL_2018_NODUMMY <- FULL_2018 %>% filter(quantity != 0 & days_supply != 0)

### MME vs. alert 1
MME_ALERT1_BASE <- feols(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply, FULL_2018_NODUMMY)
etable(MME_ALERT1_BASE, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018_NODUMMY)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1_pseudo*drug + prior_alert1 + presence_MME_pseudo + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018_NODUMMY)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)

rm(FULL_2018_NODUMMY)

### Days of supply vs. alert 5
DAYS_ALERT5_BASE <- feols(days_supply ~ active_alert5_pseudo + prior_alert5 + presence_days_pseudo + daily_dose, FULL_2018_NODUMMY)
etable(DAYS_ALERT5_BASE, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_FULL <- feols(days_supply ~ active_alert5_pseudo + prior_alert5 + presence_days_pseudo + daily_dose + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018_NODUMMY)
etable(DAYS_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_FULL)

DAYS_ALERT5_INTERACT <- feols(days_supply ~ active_alert5_pseudo*drug + prior_alert5 + presence_days_pseudo + daily_dose + age + patient_gender | prescriber_id + drug_switch, FULL_2018_NODUMMY)
etable(DAYS_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_INTERACT)

rm(FULL_2018_NODUMMY)

################################################################################
################################################################################
## Pesudo time window and dummy prescriptions

### MME vs. alert 1
MME_ALERT1_BASE <- lm(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply, data = FULL_2018)
stargazer(MME_ALERT1_BASE, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))
MME_ALERT1_BASE_CLUSTER = coeftest(MME_ALERT1_BASE, vcov = vcovHC(MME_ALERT1_BASE, type = "HC0", cluster = "patient_id"))
stargazer(MME_ALERT1_BASE_CLUSTER, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Daily dose (MME)"))

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1_pseudo*drug + prior_alert1 + presence_MME_pseudo + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)

### Days of supply vs. alert 5
DAYS_ALERT5_BASE <- feols(days_supply ~ active_alert5_pseudo + prior_alert5 + presence_days_pseudo + daily_dose, FULL_2018)
etable(DAYS_ALERT5_BASE, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_FULL <- feols(days_supply ~ active_alert5_pseudo + prior_alert5 + presence_days_pseudo + daily_dose + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_FULL)

DAYS_ALERT5_INTERACT <- feols(days_supply ~ active_alert5_pseudo*drug + prior_alert5 + presence_days_pseudo + daily_dose + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_INTERACT)

################################################################################
### No dummy for one prescription patient
DUMMY <- FULL_2018 %>% filter(quantity == 0 & days_supply == 0) %>% group_by(patient_id)
PATIENT <- FULL_2018 %>% group_by(patient_id) %>% summarize(prescriptions = n())
ONE_PRESC_PATIENT <- left_join(DUMMY, PATIENT, by = 'patient_id') %>% mutate(one_presc_dummy = ifelse(prescriptions <= 2, 1, 0)) %>% select(c(patient_id, prescriptions, one_presc_dummy, quantity))
FULL_2018_HALF_DUMMY <- left_join(FULL_2018, ONE_PRESC_PATIENT, by = c('patient_id', 'quantity')) %>% mutate(one_presc_dummy = coalesce(one_presc_dummy, 0)) %>% filter(one_presc_dummy != 1)
rm(DUMMY)
rm(PATIENT)

### MME vs. alert 1
MME_ALERT1_BASE <- feols(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + days_supply, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_BASE, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1 + prior_alert1 + presence_MME + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)
print(MME_ALERT1_FULL)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1*drug + prior_alert1 + presence_MME + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)
print(MME_ALERT1_INTERACT)

################################################################################
### No dummy for one prescription patient, with time window

### MME vs. alert 1
MME_ALERT1_BASE <- feols(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_BASE, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1_pseudo + prior_alert1 + presence_MME_pseudo + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)
print(MME_ALERT1_FULL)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1_pseudo*drug + prior_alert1 + presence_MME_pseudo + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018_HALF_DUMMY)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)
print(MME_ALERT1_INTERACT)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
library(dplyr)
library(ggplot2)
# Regression Output
library(stargazer)
library(export)
# Fast fixed-effects
library(fixest)
# Marginal effects
library(marginaleffects)

setwd("/mnt/phd/jihu/opioid/Code")
FULL_2018_1 <- read.csv("../Data/FULL_DUMMY_2018_1.csv")
FULL_2018_2 <- read.csv("../Data/FULL_DUMMY_2018_2.csv")
FULL_2018_3 <- read.csv("../Data/FULL_DUMMY_2018_3.csv")
FULL_2018_4 <- read.csv("../Data/FULL_DUMMY_2018_4.csv")
FULL_2018 <- rbind(FULL_2018_1, FULL_2018_2)
FULL_2018 <- rbind(FULL_2018, FULL_2018_3)
FULL_2018 <- rbind(FULL_2018, FULL_2018_4)
rm(FULL_2018_1)
rm(FULL_2018_2)
rm(FULL_2018_3)
rm(FULL_2018_4)

################################################################################

### MME vs. alert 1
MME_ALERT1_BASE <- feols(daily_dose ~ active_alert1_old + prior_alert1_old + presence_MME + days_supply, FULL_2018)
etable(MME_ALERT1_BASE, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_FULL <- feols(daily_dose ~ active_alert1_old + prior_alert1_old + presence_MME + days_supply + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_FULL, cluster = ~patient_id, tex = TRUE)

MME_ALERT1_INTERACT <- feols(daily_dose ~ active_alert1_old*drug + prior_alert1_old + presence_MME + days_supply + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(MME_ALERT1_INTERACT, cluster = ~patient_id, tex = TRUE)

plot_cap(MME_ALERT1_INTERACT, condition = c("drug", "active_alert1_old"))


### Days vs. alert 5
DAYS_ALERT5_BASE <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days + daily_dose, FULL_2018)
etable(DAYS_ALERT5_BASE, cluster = ~patient_id, tex = TRUE)

DAYS_ALERT5_FULL <- feols(days_supply ~ active_alert5 + prior_alert5 + presence_days + daily_dose + age + drug + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_FULL, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_FULL)

DAYS_ALERT5_INTERACT <- feols(days_supply ~ active_alert5*drug + prior_alert5 + presence_days + daily_dose + age + patient_gender | prescriber_id + drug_switch, FULL_2018)
etable(DAYS_ALERT5_INTERACT, cluster = ~patient_id, tex = TRUE)
print(DAYS_ALERT5_INTERACT)

plot_cap(DAYS_ALERT5_INTERACT, condition = c("drug", "active_alert5"))

### COMPARE CURRENT DF W/ PREVIOUS DF

library(dplyr)
library(lubridate)
library(arules)
library(parallel)
library(data.table)

setwd("/export/storage_cures/CURES/Processed/")
year = 2018

FULL_NEW <- read.csv(paste("/export/storage_cures/CURES/Processed/FULL_OPIOID_", year ,"_INPUT.csv", sep="")) # 10025197
# FULL_NEW_FIRST <- read.csv(paste("FULL_OPIOID_", year ,"_FIRST_INPUT.csv", sep="")) # 4978316 prescriptions
FULL_OLD <- read.csv(paste("/mnt/phd/jihu/opioid/Data/FULL_", year ,"_LONGTERM_UPTOFIRST.csv", sep="")) # 5637115

################################################################################

LT_NEW <- FULL_NEW %>% filter(long_term_180 == 1) # 2878090, 15662 with days_supply >= 90, huge concurrent_MME, a lot of prescriptions etc.
LT_OLD <- FULL_OLD %>% filter(long_term_180 == 1) # 103721

# 3,972,629 more prescriptions from 1,112,882 more patients
# E.g., 724, 752
new_patient_ids <- FULL_NEW %>% anti_join(FULL_OLD, by = "patient_id") %>% select(patient_id) %>% distinct() 
new_patient_ids_vector <- new_patient_ids$patient_id
FULL_NEW_subset <- FULL_NEW %>% filter(patient_id %in% new_patient_ids_vector)
PATIENT_NEW <- FULL_NEW_subset %>% 
  group_by(patient_id) %>% 
  summarize(num_presc = n(), 
            long_term = ifelse(sum(long_term_180) > 0, 1, 0))

PATIENT_NEW_SINGLE <- FULL_NEW %>% filter(patient_id == 28210)
PATIENT_OLD_SINGLE <- FULL_OLD %>% filter(patient_id == 28210)

# 1123 prescriptions from 428 patients dropped now
dropped_patient_ids <- FULL_OLD %>% anti_join(FULL_NEW, by = "patient_id") %>% select(patient_id) %>% distinct() 
dropped_patient_ids_vector <- dropped_patient_ids$patient_id
FULL_OLD_subset <- FULL_OLD %>% filter(patient_id %in% dropped_patient_ids_vector)

################################################################################
### PATIENT LEVEL
PATIENT_NEW_LT <- FULL_NEW %>% 
  group_by(patient_id) %>% 
  summarize(num_presc = n(), 
            long_term = ifelse(sum(long_term_180) > 0, 1, 0))
long_term_counts <- PATIENT_NEW_LT %>% group_by(long_term) %>% summarise(count = n())

PATIENT_OLD_LT <- FULL_OLD %>% 
  group_by(patient_id) %>% 
  summarize(num_presc = n(), 
            long_term = ifelse(sum(long_term_180) > 0, 1, 0))
long_term_counts <- PATIENT_OLD_LT %>% group_by(long_term) %>% summarise(count = n())

common_patients <- inner_join(PATIENT_NEW_LT, PATIENT_OLD_LT, by = "patient_id")
patient_diff_num_presc <- common_patients %>% filter(num_presc.x != num_presc.y) # 94154
# up to first long_term_180 vs. up to first long_term


################################################################################
######################## UP TO FIRST LONG_TERM_180 #############################
################################################################################

# TEST <- FULL_NEW[1:20,]
# TEST<- FULL_NEW %>% filter(patient_id == 49670 | patient_id == 47)
# TEST_UPTOFIRST <- TEST %>%
#   group_by(patient_id) %>%
#   arrange(date_filled) %>%
#   mutate(first_long_term_date = ifelse(any(long_term_180 == 1), min(date_filled[long_term_180 == 1], na.rm = TRUE), NA)) %>%
#   filter(is.na(first_long_term_date) | date_filled <= first_long_term_date) %>%
#   select(-first_long_term_date) %>%
#   ungroup()

# 7757463 prescriptions
FULL_NEW_UPTOFIRST <- FULL_NEW %>%
  group_by(patient_id) %>%
  arrange(date_filled) %>%
  mutate(first_long_term_date = ifelse(any(long_term_180 == 1), min(date_filled[long_term_180 == 1], na.rm = TRUE), NA)) %>%
  filter(is.na(first_long_term_date) | date_filled <= first_long_term_date) %>%
  select(-first_long_term_date) %>%
  ungroup()

write.csv(FULL_NEW_UPTOFIRST, paste("/export/storage_cures/CURES/Processed/FULL_OPIOID_", year ,"_UPTOFIRST_INPUT.csv", sep=""), row.names = FALSE)

LT_NEW_UPTOFIRST <- FULL_NEW_UPTOFIRST %>% filter(long_term_180 == 1) # 610360
unique_patient_count <- FULL_NEW_UPTOFIRST %>% summarise(unique_patients = n_distinct(patient_id))

PATIENT_NEW_UPTOFIRST_LT <- FULL_NEW_UPTOFIRST %>% 
  group_by(patient_id) %>% 
  summarize(num_presc = n(), 
            long_term = ifelse(sum(long_term_180) > 0, 1, 0))
long_term_counts <- PATIENT_NEW_UPTOFIRST_LT %>% group_by(long_term) %>% summarise(count = n())

################################################################################
############################### DUPLICATES #####################################
################################################################################

year = 2019
setwd("/export/storage_cures/CURES/Processed/")
FULL_CURRENT <- read.csv(paste("../RX_", year, ".csv", sep="")) 
# 2018: 140566 prescriptions from 27935 patients
# 2019: 12474910 prescriptions from 1643910 patients
DUPLICATES <- FULL_CURRENT %>% group_by(across(-X)) %>% filter(n() > 1) 

UNDERAGE <- FULL_CURRENT %>% filter(patient_birth_year > 2000) # 1396129 prescriptions from 393937 patients
OVERAGE <- FULL_CURRENT %>% filter(patient_birth_year < 1918) # 22279 prescriptions from 4825 patients (we should keep it)

DUP <- DUPLICATES %>% filter(patient_id == 54195156) # 2019
DUP <- DUPLICATES %>% filter(patient_id == 68043086) # 2019
DUP <- DUPLICATES %>% filter(patient_id == 48) # 2019

################################################################################
################################################################################
################################################################################

FULL <- read.csv(paste("../Results/FULL_LTOUR_table.csv", sep="")) 
prop_data <- FULL %>%
  group_by(num_prior_prescriptions_binary) %>%
  summarize(proportion_true = mean(True == 1),
            probability = mean(Prob))

ggplot(prop_data, aes(x = factor(num_prior_prescriptions_binary), y = proportion_true)) +
  geom_bar(stat = "identity", fill = "steelblue") +  # Dodge places bars next to each other
  ggtitle("Count of 'True' by 'Binary'") +
  theme_minimal()


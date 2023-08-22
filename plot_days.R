library(dplyr)
library(arules)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")
## because we drop repeated observation, keep prescription up to first y = 1
################################################################################

### Contains true positives only

PATIENT_CURES_INSAMPLE <- read.csv("../Data/PATIENT_2018_LONGTERM_base_output.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_CURES_INSAMPLE, "../Data/PATIENT_CURES_INSAMPLE.csv", row.names = FALSE)
PATIENT_CURES_OUTSAMPLE <- read.csv("../Data/PATIENT_2019_LONGTERM_base_output.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_CURES_OUTSAMPLE, "../Data/PATIENT_CURES_OUTSAMPLE.csv", row.names = FALSE)


# Result from five tables from nested CV
# Note: the list of patients are almost the same, but the days to first prediction differs
PATIENT_LTOUR_INSAMPLE_ONE <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_one.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_LTOUR_INSAMPLE_ONE, "../Data/PATIENT_LTOUR_INSAMPLE_ONE.csv", row.names = FALSE)

PATIENT_LTOUR_INSAMPLE_TWO <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_two.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_LTOUR_INSAMPLE_TWO, "../Data/PATIENT_LTOUR_INSAMPLE_TWO.csv", row.names = FALSE)

PATIENT_LTOUR_INSAMPLE_THREE <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_three.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_LTOUR_INSAMPLE_THREE, "../Data/PATIENT_LTOUR_INSAMPLE_THREE.csv", row.names = FALSE)

PATIENT_LTOUR_INSAMPLE_FOUR <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_four.csv") %>% dplyr::select(-c(first_long_term_date)) 
write.csv(PATIENT_LTOUR_INSAMPLE_FOUR, "../Data/PATIENT_LTOUR_INSAMPLE_FOUR.csv", row.names = FALSE)

PATIENT_LTOUR_INSAMPLE_FIVE <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_five.csv") %>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_LTOUR_INSAMPLE_FIVE, "../Data/PATIENT_LTOUR_INSAMPLE_FIVE.csv", row.names = FALSE)



PATIENT_LTOUR_OUTSAMPLE_ONE <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_one.csv") %>% dplyr::select(-c(first_long_term_date)) 
write.csv(PATIENT_LTOUR_OUTSAMPLE_ONE, "../Data/PATIENT_LTOUR_OUTSAMPLE_ONE.csv", row.names = FALSE)

PATIENT_LTOUR_OUTSAMPLE_TWO <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_two.csv") %>% dplyr::select(-c(first_long_term_date)) 
write.csv(PATIENT_LTOUR_OUTSAMPLE_TWO, "../Data/PATIENT_LTOUR_OUTSAMPLE_TWO.csv", row.names = FALSE)

PATIENT_LTOUR_OUTSAMPLE_THREE <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_three.csv") %>% dplyr::select(-c(first_long_term_date)) 
write.csv(PATIENT_LTOUR_OUTSAMPLE_THREE, "../Data/PATIENT_LTOUR_OUTSAMPLE_THREE.csv", row.names = FALSE)

PATIENT_LTOUR_OUTSAMPLE_FOUR <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_four.csv")%>% dplyr::select(-c(first_long_term_date))
write.csv(PATIENT_LTOUR_OUTSAMPLE_FOUR, "../Data/PATIENT_LTOUR_OUTSAMPLE_FOUR.csv", row.names = FALSE)

PATIENT_LTOUR_OUTSAMPLE_FIVE <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_five.csv") %>% dplyr::select(-c(first_long_term_date)) 
write.csv(PATIENT_LTOUR_OUTSAMPLE_FIVE, "../Data/PATIENT_LTOUR_OUTSAMPLE_FIVE.csv", row.names = FALSE)

################################################################################

### Days from first prescription to first positive prediction

round(c(mean(PATIENT_CURES_INSAMPLE$firstpred_from_firstpresc), sd(PATIENT_CURES_INSAMPLE$firstpred_from_firstpresc)))
round(c(mean(PATIENT_CURES_OUTSAMPLE$firstpred_from_firstpresc), sd(PATIENT_CURES_OUTSAMPLE$firstpred_from_firstpresc)))

round(c(mean(PATIENT_LTOUR_INSAMPLE_ONE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_INSAMPLE_ONE$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_INSAMPLE_TWO$firstpred_from_firstpresc), sd(PATIENT_LTOUR_INSAMPLE_TWO$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_INSAMPLE_THREE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_INSAMPLE_THREE$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_INSAMPLE_FOUR$firstpred_from_firstpresc), sd(PATIENT_LTOUR_INSAMPLE_FOUR$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_INSAMPLE_FIVE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_INSAMPLE_FIVE$firstpred_from_firstpresc)))
round(mean(c(mean(PATIENT_LTOUR_INSAMPLE_ONE$firstpred_from_firstpresc), mean(PATIENT_LTOUR_INSAMPLE_TWO$firstpred_from_firstpresc), mean(PATIENT_LTOUR_INSAMPLE_THREE$firstpred_from_firstpresc), mean(PATIENT_LTOUR_INSAMPLE_FOUR$firstpred_from_firstpresc), mean(PATIENT_LTOUR_INSAMPLE_FIVE$firstpred_from_firstpresc))))


round(c(mean(PATIENT_LTOUR_OUTSAMPLE_ONE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_OUTSAMPLE_ONE$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_OUTSAMPLE_TWO$firstpred_from_firstpresc), sd(PATIENT_LTOUR_OUTSAMPLE_TWO$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_OUTSAMPLE_THREE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_OUTSAMPLE_THREE$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_OUTSAMPLE_FOUR$firstpred_from_firstpresc), sd(PATIENT_LTOUR_OUTSAMPLE_FOUR$firstpred_from_firstpresc)))
round(c(mean(PATIENT_LTOUR_OUTSAMPLE_FIVE$firstpred_from_firstpresc), sd(PATIENT_LTOUR_OUTSAMPLE_FIVE$firstpred_from_firstpresc)))
round(mean(c(mean(PATIENT_LTOUR_OUTSAMPLE_ONE$firstpred_from_firstpresc), mean(PATIENT_LTOUR_OUTSAMPLE_TWO$firstpred_from_firstpresc), mean(PATIENT_LTOUR_OUTSAMPLE_THREE$firstpred_from_firstpresc), mean(PATIENT_LTOUR_OUTSAMPLE_FOUR$firstpred_from_firstpresc), mean(PATIENT_LTOUR_OUTSAMPLE_FIVE$firstpred_from_firstpresc))))

################################################################################

### Days from first y_hat = 1 to y = 1 (recall that we kept prescriptions up to first y = 1)

round(c(mean(PATIENT_CURES_INSAMPLE$day_to_long_term_180), sd(PATIENT_CURES_INSAMPLE$day_to_long_term_180)))
round(c(mean(PATIENT_CURES_OUTSAMPLE$day_to_long_term_180), sd(PATIENT_CURES_OUTSAMPLE$day_to_long_term_180)))

round(c(mean(PATIENT_LTOUR_INSAMPLE_ONE$day_to_long_term_180), sd(PATIENT_LTOUR_INSAMPLE_ONE$day_to_long_term_180)))
round(c(mean(PATIENT_LTOUR_OUTSAMPLE_ONE$day_to_long_term_180), sd(PATIENT_LTOUR_OUTSAMPLE_ONE$day_to_long_term_180)))

################################################################################

c(mean(PATIENT_CURES$day_to_long_term), sd(PATIENT_CURES$day_to_long_term))
c(mean(PATIENT_CURES$day_to_long_term_180), sd(PATIENT_CURES$day_to_long_term_180))

TEMP <- PATIENT_CURES %>% mutate(Days = day_to_long_term_180, Scenario = 'Days to long term 180')
PATIENT_CURES <- PATIENT_CURES %>% mutate(Days = day_to_long_term, Scenario = 'Days to long term')
PATIENT_CURES_all <- rbind(TEMP, PATIENT_CURES)
PATIENT_CURES_all %>% ggplot(aes(x = Days, color = factor(Scenario))) + 
  geom_density(alpha = 0.5, size=1) + 
  geom_vline(xintercept=0, linetype="dashed", color = "black", size=1) +
  labs(x = "Days", y = "Density", color = "Scenario") +
  theme(legend.position = c(0.8, 0.5))
ggsave("../Figs/density_days_base.pdf", bg="white", width=10, height=4, dpi=300)

################################################################################
PATIENT_full <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_one.csv")
c(mean(PATIENT_full$day_to_long_term), sd(PATIENT_full$day_to_long_term))
c(mean(PATIENT_full$day_to_long_term_180), sd(PATIENT_full$day_to_long_term_180))

TEMP <- PATIENT_full %>% mutate(Days = day_to_long_term_180, Scenario = 'Days to long term 180')
PATIENT_full <- PATIENT_full %>% mutate(Days = day_to_long_term, Scenario = 'Days to long term')
PATIENT_full_all <- rbind(TEMP, PATIENT_full)
PATIENT_full_all %>% ggplot(aes(x = Days, color = factor(Scenario))) + 
  geom_density(alpha = 0.5, size=1) + 
  geom_vline(xintercept=0, linetype="dashed", color = "black", size=1) +
  labs(x = "Days", y = "Density", color = "Scenario") +
  theme(legend.position = c(0.8, 0.5))
ggsave("../Figs/density_days_full.pdf", bg="white", width=10, height=4, dpi=300)

################################################################################

FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM.csv")

c(mean(FULL_2018$days_supply), sd(FULL_2018$days_supply)) # 10.25, 10.18

FULL_2018 %>% ggplot(aes(x = ifelse(days_supply>40, 40, days_supply))) + 
  geom_density(alpha = 0.5, size=1, color="darkblue") + 
  geom_vline(xintercept=mean(FULL_2018$days_supply), linetype="dashed", color = "black", size=1) +
  labs(x = "Days of supply per prescription", y = "Density") +
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 14))
ggsave("../Figs/density_days_supply.pdf", bg="white", width=10, height=8, dpi=300)

c(mean(FULL_2018$consecutive_days), sd(FULL_2018$consecutive_days)) # 14.21, 19.66

FULL_2018 %>% ggplot(aes(x = ifelse(consecutive_days>90, 90, consecutive_days))) + 
  geom_density(alpha = 0.5, size=1, color="darkblue") + 
  geom_vline(xintercept=mean(FULL_2018$consecutive_days), linetype="dashed", color = "black", size=1) +
  labs(x = "Consecutive days per prescription", y = "Density") +
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 14))
ggsave("../Figs/density_consecutive_days.pdf", bg="white", width=10, height=8, dpi=300)


colnames(FULL_2018)



################################################################################
################################################################################
################################################################################
PATIENT_CURES <- read.csv("../Data/PATIENT_2018_LONGTERM_base_output.csv")
# PATIENT_CURES <- read.csv("../Data/PATIENT_2019_LONGTERM_base_output.csv")

round(c(mean(PATIENT_CURES$day_to_long_term), sd(PATIENT_CURES$day_to_long_term)),2)
round(c(mean(PATIENT_CURES$day_to_long_term_180), sd(PATIENT_CURES$day_to_long_term_180)),2)
round(c(mean(PATIENT_CURES$firstpred_from_firstpresc), sd(PATIENT_CURES$firstpred_from_firstpresc)),2)

################################################################################
PATIENT_full <- read.csv("../Data/PATIENT_2018_LONGTERM_full_output_one.csv")
# PATIENT_full <- read.csv("../Data/PATIENT_2019_LONGTERM_full_output_one.csv")

round(c(mean(PATIENT_full$day_to_long_term), sd(PATIENT_full$day_to_long_term)),2)
round(c(mean(PATIENT_full$day_to_long_term_180), sd(PATIENT_full$day_to_long_term_180)),2)
round(c(mean(PATIENT_full$firstpred_from_firstpresc), sd(PATIENT_full$firstpred_from_firstpresc)),2)

################################################################################

PATIENT_CURES$Scenario = 'CURES'
PATIENT_full$Scenario = 'LTOUR'
PATIENT_all <- rbind(PATIENT_CURES, PATIENT_full)
PATIENT_all$Scenario = as.factor(PATIENT_all$Scenario)

x_base_avg = round(mean(PATIENT_CURES$firstpred_from_firstpresc),1)
x_full_avg = round(mean(PATIENT_full$firstpred_from_firstpresc),1)

PATIENT_all %>% ggplot(aes(x = firstpred_from_firstpresc, fill = factor(Scenario))) +
  geom_histogram(binwidth = 10) +
  geom_vline(xintercept=x_base_avg, linetype="dashed", color = "#404080", size=1) +
  geom_vline(xintercept=x_full_avg, linetype="dashed", color = "#69b3a2", size=1) +
  labs(x = "Days from first prescription", y = "Frequency", color = "Scenario")  +
  theme(legend.position = c(0.8, 0.5)) +
  scale_fill_manual(values=c("#69b3a2", "#404080")) + 
  xlim(0, 100) + ylim(0, 1e+05)

ggsave("../Figs/histogram_daysfromfirstpresc.pdf", bg="white", width=8, height=6, dpi=300)

################################################################################

x_base_avg = round(mean(PATIENT_CURES$day_to_long_term),1)
x_full_avg = round(mean(PATIENT_full$day_to_long_term),1)

PATIENT_all %>% ggplot(aes(x = day_to_long_term, fill = factor(Scenario))) +
  geom_histogram(binwidth = 10) +
  geom_vline(xintercept=x_base_avg, linetype="dashed", color = "#404080", size=1) +
  geom_vline(xintercept=x_full_avg, linetype="dashed", color = "#69b3a2", size=1) +
  labs(x = "Days to long term", y = "Frequency", color = "Scenario")  +
  theme(legend.position = c(0.2, 0.8)) +
  scale_fill_manual(values=c("#69b3a2", "#404080")) + 
  xlim(0, 100) + ylim(0, 1e+05)

ggsave("../Figs/histogram_daystolongterm.pdf", bg="white", width=8, height=6, dpi=300)

################################################################################

x_base_avg = round(mean(PATIENT_CURES$day_to_long_term_180),1)
x_full_avg = round(mean(PATIENT_full$day_to_long_term_180),1)

PATIENT_all %>% ggplot(aes(x = -day_to_long_term_180, fill = factor(Scenario))) +
  geom_histogram(binwidth = 10) +
  geom_vline(xintercept=-x_base_avg, linetype="dashed", color = "#404080", size=1) +
  geom_vline(xintercept=-x_full_avg, linetype="dashed", color = "#69b3a2", size=1) +
  labs(x = "Days to long term 180", y = "Frequency", color = "Scenario")  +
  theme(legend.position = c(0.8, 0.8)) +
  scale_fill_manual(values=c("#69b3a2", "#404080")) + 
  xlim(0, 100) + ylim(0, 1e+05)

ggsave("../Figs/histogram_daystolongterm180.pdf", bg="white", width=8, height=6, dpi=300)




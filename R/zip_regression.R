### Merge zip code information

library(dplyr)
library(lubridate)
library(arules)
library(ggplot2)
library(scales)
library(stargazer)
library(export)

setwd("/mnt/phd/jihu/opioid/Code")
year = 2018

###############################################################################
###############################################################################
###############################################################################

# FULL_ALL <- read.csv(paste("../Data/FULL_", year, "_ALERT.csv", sep=""))
FULL_ALL <- read.csv(paste("../Data/FULL_", year, "_LONGTERM_NOILLICIT.csv", sep=""))
CA_VISIT <- read.csv("../Data/CA/CA_ED_Visit.csv")
CA_DEATH <- read.csv("../Data/CA/CA_Death.csv")
CA_VISIT$Zip.code <- as.character(CA_VISIT$Zip.code)
CA_VISIT$Visit_Rate <- CA_VISIT$Rates
CA_DEATH$Zip.code <- as.character(CA_DEATH$Zip.code)
CA_DEATH$Death_Rate <- CA_DEATH$Rates

###############################################################################
###############################################################################
###############################################################################
# Total number of prescriptions
# Total number of patients
# Average MME of all prescriptions
# Number of high-risk/long-term users (distinct value that satisfies certain condition)

ZIP <- FULL_ALL %>% group_by(patient_zip) %>% summarize(num_prescriptions = n(),
                                                    num_patient = n_distinct(patient_id),
                                                    avg_MME = mean(daily_dose),
                                                    avg_days = mean(days_supply))

## Count the number of long term user & high risk user
LONG_TERM <- FULL_ALL %>% group_by(patient_id) %>% summarize(long_term_user = ifelse(sum(long_term) > 0, 1, 0),
                                                             high_risk_user = ifelse(sum(num_alert) > 0, 1, 0))
FULL_ALL <- left_join(FULL_ALL, LONG_TERM, by = "patient_id")
ZIP_LONGTERM <- FULL_ALL %>% filter(long_term_user == 1) %>% group_by(patient_zip) %>% summarize(num_long_term = n_distinct(patient_id))
ZIP_HIGHRISK <- FULL_ALL %>% filter(high_risk_user == 1) %>% group_by(patient_zip) %>% summarize(num_high_risk = n_distinct(patient_id))
ZIP <- left_join(ZIP, ZIP_LONGTERM, by = "patient_zip") %>% replace(is.na(.), 0)
ZIP <- left_join(ZIP, ZIP_HIGHRISK, by = "patient_zip") %>% replace(is.na(.), 0)

## Merge with the visit data
## Note: some of the patient's zip code are outside CA
## Instead of replacing NA with 0, we filter out those with NA
ZIP <- left_join(ZIP, CA_VISIT, by = c('patient_zip'='Zip.code')) %>% filter(!is.na(Visit_Rate))
ZIP <- left_join(ZIP, CA_DEATH, by = c('patient_zip'='Zip.code')) %>% filter(!is.na(Death_Rate))

## Merge with HPI & Population
CA_HPI <- read.csv("../Data/CA/HPI.csv")
CA_ZIP <- read.csv("../Data/CA/CaliforniaZip.csv")
CA_ZIP <- CA_ZIP %>% select(c(Zip, Population, Race_White, Race_Black, Race_Asian, Race_Hispanic, Race_Other,
                              MedianHHIncome, Poverty, Unemployment))

CA_HPI$Zip <- as.character(CA_HPI$Zip)
CA_ZIP$Zip <- as.character(CA_ZIP$Zip)

## Some in CA_ZIP, CA_HPI but not in ZIP
notinZIP = CA_ZIP$Zip[!CA_ZIP$Zip %in% ZIP$patient_zip] # in CA_ZIP but not in ZIP, most are schools, Disney studios etc.
notinCA = ZIP$patient_zip[!ZIP$patient_zip %in% CA_ZIP$Zip] # in ZIP but not in CA_ZIP, shopping mall, stadiums, hotel

ZIP <- left_join(ZIP, CA_HPI, by = c('patient_zip'='Zip'))
ZIP <- left_join(ZIP, CA_ZIP, by = c('patient_zip'='Zip'))

## Percentage of high-risk/long-term users
ZIP$prec_high_risk = ZIP$num_high_risk/ZIP$Population
ZIP$prec_long_term = ZIP$num_long_term/ZIP$Population

## Actual death/visit weighted by population (age-adjusted rate per 100k residents)
# Age-adjusted rates are computed using the direct method by applying age-specific 
# rates in a population of interest to a standardized age distribution. This eliminates
# differences in observed rates that result from age differences in population composition.

ZIP$Visit = ZIP$Visit_Rate * ZIP$Population / 100000
ZIP$Death = ZIP$Death_Rate * ZIP$Population / 100000
ZIP <- ZIP %>% select(c(patient_zip, num_prescriptions, num_patient, avg_MME, avg_days, num_long_term,
                        num_high_risk, Visit_Rate, Visit, Death_Rate, Death, HPI, HPIQuartile, Population,
                        Race_White, Race_Black, Race_Asian, Race_Hispanic, Race_Other, 
                        MedianHHIncome, Poverty, Unemployment, prec_high_risk, prec_long_term))

write.csv(ZIP, "../Data/CA_ZIP_Rates.csv", row.names = FALSE)

###############################################################################
###############################################################################
###############################################################################

ZIP <- read.csv("../Data/CA_ZIP_Rates.csv")
ZIP <- ZIP %>% filter(Population != 0) # ~30 zip code

## Scatter plot of long-term vs high-risk
ggplot(ZIP, aes(x=num_long_term, y=num_high_risk)) + 
  xlab("Number of long term users") + ylab("Number of high risk users") + 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) + 
  geom_point()
ggsave("../Result/num_users_zip.pdf", bg="white", width=10, height=8, dpi=300)

###############################################################################
## Long-term, high-risk vs. Population, HPI
# scatter
ggplot(ZIP, aes(x=HPI, y=prec_long_term)) + xlab("HPI") + ylab("Proportion of long term users") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) + 
  geom_point()
ggsave("../Result/long_term_HPI_scatter.pdf", bg="white", width=10, height=8, dpi=300)

ggplot(ZIP, aes(x=HPI, y=prec_high_risk)) + xlab("HPI") + ylab("Proportion of high risk users") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point()
ggsave("../Result/high_risk_HPI_scatter.pdf", bg="white", width=10, height=8, dpi=300)

# boxplot (long term, high risk)
TEMP <- ZIP %>% filter(!is.na(HPIQuartile))
LONGTERM <- TEMP
LONGTERM$Rate = LONGTERM$num_long_term/LONGTERM$Population
LONGTERM$Scenario = "Long term users"
HIGHRISK <- TEMP
HIGHRISK$Rate = HIGHRISK$num_high_risk/HIGHRISK$Population
HIGHRISK$Scenario = "High risk users"
TEMP <- rbind(LONGTERM, HIGHRISK)

ggplot(TEMP, aes(x=factor(HPIQuartile), y=Rate, fill = Scenario)) + 
  xlab("HPI Quartile") + ylab("Proportion") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12)) + 
  geom_boxplot()
ggsave("../Result/rate_HPI_boxplot.pdf", bg="white", width=10, height=8, dpi=300)

# boxplot (death, visit)
# TEMP <- ZIP %>% filter(!is.na(HPIQuartile))
# DEATH <- TEMP
# DEATH$Rate = DEATH$Death_Rate
# DEATH$Scenario = "Death"
# VISIT <- TEMP
# VISIT$Rate = VISIT$Visit_Rate
# VISIT$Scenario = "Visit"
# TEMP <- rbind(DEATH, VISIT)
# 
# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Rate, fill = Scenario)) + 
#   xlab("HPI Quartile") + ylab("Number") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14),
#         legend.title = element_text(size=12),
#         legend.text = element_text(size=12)) + 
#   geom_boxplot()

###############################################################################
## Death/Visit rate vs. HPI
## NOTE: different HPI have different population, so the plot could be deceiving

TEMP <- ZIP %>% filter(!is.na(HPIQuartile))

## The value of death vs. visit is significantly different, plot separately
# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Rate, fill = Scenario)) + xlab("HPI") + ylab("Rates") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14)) + 
#   geom_boxplot()

# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Death_Rate)) + xlab("HPI") + ylab("Death Rate") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14)) +
#   geom_boxplot()
# ggsave("../Result/death_HPI_full.pdf", bg="white", width=10, height=8, dpi=300)
# 
# TEMP <- TEMP %>% filter(Death_Rate < 100)
# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Death_Rate)) + xlab("HPI") + ylab("Death Rate") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14)) +
#   geom_boxplot()
# ggsave("../Result/death_HPI_partial.pdf", bg="white", width=10, height=8, dpi=300)

########
# TEMP <- ZIP %>% filter(!is.na(HPIQuartile))
# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Visit_Rate)) + xlab("HPI") + ylab("Visit Rate") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14)) +
#   geom_boxplot()
# ggsave("../Result/visit_HPI_full.pdf", bg="white", width=10, height=8, dpi=300)
# 
# TEMP <- TEMP %>% filter(Visit_Rate < 100)
# ggplot(TEMP, aes(x=factor(HPIQuartile), y=Visit_Rate)) + xlab("HPI") + ylab("Visit Rate") +
#   theme(axis.title = element_text(size = 18),
#         axis.text = element_text(size = 14)) +
#   geom_boxplot()
# ggsave("../Result/visit_HPI_partial.pdf", bg="white", width=10, height=8, dpi=300)

###############################################################################
## Sanity check
ggplot(ZIP, aes(x=Population, y=num_long_term)) + xlab("Population") + ylab("Number of high risk users") + 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point()
ggsave("../Result/long_term_population.pdf", bg="white", width=10, height=8, dpi=300)

ggplot(ZIP, aes(x=Population, y=num_high_risk)) + xlab("Population") + ylab("Number of high risk users") + 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point()
ggsave("../Result/high_risk_population.pdf", bg="white", width=10, height=8, dpi=300)

###############################################################################
###############################################################################
###############################################################################

ZIP <- read.csv("../Data/CA_ZIP_Rates.csv")
ZIP <- ZIP %>% filter(Population != 0)

## Scatterplot of ED visit vs. number of long-term use, high-risk use
ggplot(ZIP, aes(x=prec_high_risk, y=Visit_Rate)) +
  xlab("Proportion of high risk users") +
  ylab("Visit rate (per 100k residents)") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point(aes(size = Population)) +
  scale_size_continuous(range = c(0.1, 3))
ggsave("../Result/high_risk_visit.pdf", bg="white", width=10, height=8, dpi=300)

TEMP <- ZIP %>% filter(Visit_Rate > 0 & Visit_Rate < 500 & prec_high_risk < 0.1)
ggplot(TEMP, aes(x=prec_high_risk, y=Visit_Rate)) +
  xlab("Proportion of high risk users") +
  ylab("Visit rate (per 100k residents)") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point(aes(size = Population)) +
  scale_size_continuous(range = c(0.1, 3))
ggsave("../Result/high_risk_visit_trunc.pdf", bg="white", width=10, height=8, dpi=300)

ggplot(ZIP, aes(x=prec_high_risk, y=log(Visit_Rate + 0.001))) +
  xlab("Proportion of high risk users") +
  ylab("log(Visit rate)") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point(aes(size = Population)) +
  scale_size_continuous(range = c(0.1, 3))
ggsave("../Result/high_risk_visit_log.pdf", bg="white", width=10, height=8, dpi=300)


## Death
TEMP <- ZIP %>% filter(Death_Rate > 0 & Death_Rate < 200 & prec_high_risk < 0.1)
ggplot(TEMP, aes(x=prec_high_risk, y=Death_Rate)) +
  xlab("Proportion of high risk users") +
  ylab("Death rate (per 100k residents)") +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  geom_point(aes(size = Population)) +
  scale_size_continuous(range = c(0.1, 3))
ggsave("../Result/high_risk_death_trunc.pdf", bg="white", width=10, height=8, dpi=300)

###############################################################################
###############################################################################
###############################################################################
## Poisson regression (y is count data, although it is not integer)

high_risk <- glm(num_high_risk ~ Population + avg_MME + avg_days + factor(HPIQuartile), family="poisson", data = ZIP)
summary(high_risk)
stargazer(high_risk, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Visit Rate"))

# VISIT - Without avg_MME etc.
high_risk <- glm(round(Visit) ~ prec_high_risk + Population + factor(HPIQuartile), family="poisson", data = ZIP)
summary(high_risk)
stargazer(high_risk, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Visit Rate"))

# DEATH
high_risk <- glm(round(Death) ~ prec_high_risk + Population + factor(HPIQuartile), family="poisson", data = ZIP)
summary(high_risk)
stargazer(high_risk, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Visit Rate"))

###############################################################################

# VISIT - Without avg_MME etc.
high_risk <- glm(round(Visit) ~ prec_long_term + Population + factor(HPIQuartile), family="poisson", data = ZIP)
summary(high_risk)
stargazer(high_risk, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Visit Rate"))

# DEATH
high_risk <- glm(round(Death) ~ prec_long_term + Population + factor(HPIQuartile), family="poisson", data = ZIP)
summary(high_risk)
stargazer(high_risk, digits=3, type="latex", no.space=TRUE, dep.var.labels=c("Visit Rate"))

###############################################################################
###############################################################################
###############################################################################

library(ggplot2)

ZIP_Visit <- ZIP
ZIP_Death <- ZIP
ZIP_Visit$Count <- ZIP_Visit$Visit
ZIP_Death$Count <- ZIP_Death$Death
ZIP_Visit$Scenario <- "Visit"
ZIP_Death$Scenario <- "Death"
ZIP_TEMP <- rbind(ZIP_Visit, ZIP_Death)

### Scatterplot between visit/death vs. long-term use
ggplot(ZIP_TEMP, aes(x=num_long_term, y=Count, group=Scenario)) + 
  xlab("Number of Long Term Users") +
  ylab("Number of ED Visits/Deaths") +
  theme(axis.title = element_text(size=16),
        axis.text = element_text(size=14),
        legend.title = element_text(size=16),
        legend.text = element_text(size=14)) +
  geom_point(aes(size = Population, color = Scenario), alpha = 0.6) +
  geom_smooth(method=lm, se=TRUE, fullrange=FALSE, level=0.95) +
  scale_size_continuous(range = c(0.1, 3))
ggsave("../Result/visit_death_longterm.pdf", bg="white", width=10, height=6, dpi=300)


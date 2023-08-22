library(dplyr)
library(arules)
library(parallel)
library(ggplot2)
setwd("/mnt/phd/jihu/opioid/Code")

FULL_2018 <- read.csv("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv")
colnames(FULL_2018)
LONGTERM <- FULL_2018 %>% filter(long_term_180 == 1)

n_size = 10000 # 500, 2500, 5000, 10000
stratified <- FULL_2018 %>% group_by(long_term_180) %>% sample_n(size=n_size)
stratified <- stratified %>% dplyr::select(c(concurrent_MME, concurrent_methadone_MME, consecutive_days, 
                                             num_prescribers, num_pharmacies, concurrent_benzo,
                                             age, num_presc, dose_diff, MME_diff, days_diff,
                                             Codeine, Hydrocodone, Oxycodone, Morphine, HMFO,
                                             Medicaid, CommercialIns, Medicare, CashCredit, MilitaryIns, WorkersComp, Other, IndianNation, 
                                             switch_drug, switch_payment, ever_switch_drug, ever_switch_payment,
                                             long_term_180))

write.csv(stratified, paste("../Data/SAMPLE_2018_LONGTERM_stratified_", n_size*2,".csv", sep = ""), row.names = FALSE)

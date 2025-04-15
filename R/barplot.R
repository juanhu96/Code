library(ggplot2)
library(dplyr)
library(gridExtra)

setwd("/export/storage_cures/CURES/Results/")
export_path <- "../Plots/"

FULL_LTOUR = read.csv("FULL_LTOUR_table.csv", header = TRUE) 
FULL_LTOUR <- FULL_LTOUR %>% rename(top_prescriber_binary = prescriber_yr_avg_days_median_binary, 
                                    top_pharmacy_binary = pharmacy_yr_avg_days_median_binary)

# ============================================================================
binary_vars <- c("num_prior_prescriptions_binary", 
                 "top_prescriber_binary", 
                 "concurrent_MME_binary", 
                 "age_binary", 
                 "Medicare_Medicaid_binary", 
                 "top_pharmacy_binary")

PRIOR = LTOUR %>% 
  group_by(num_prior_prescriptions_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = num_prior_prescriptions_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Prior opioid prescription")

PRESCRIBER = LTOUR %>% 
  group_by(prescriber_yr_avg_days_median_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = prescriber_yr_avg_days_median_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Top opioid prescriber")


PHARMACY = LTOUR %>% 
  group_by(pharmacy_yr_avg_days_median_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = pharmacy_yr_avg_days_median_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Top opioid dispenser")


MME = LTOUR %>% 
  group_by(concurrent_MME_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = concurrent_MME_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Total daily MME above 40")


AGE = LTOUR %>% 
  group_by(age_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = age_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Age above 30 years")


LONGACT = LTOUR %>% 
  group_by(long_acting_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = long_acting_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Long acting opioid")


SUMMARY = rbind(PRIOR, PRESCRIBER, PHARMACY, MME, AGE, LONGACT)
SUMMARY$Feature = factor(SUMMARY$Feature, levels=c("Prior opioid prescription", "Top opioid prescriber", "Top opioid dispenser", "Total daily MME above 40", "Age above 30 years", "Long acting opioid"))


ggplot(data=SUMMARY) + facet_wrap(~Feature, nrow=2) +
  geom_col(aes(x=Group, y=PropPos, fill=Group), width=0.5) +
  geom_point(aes(x=Group, y=LTOUR, color=""), size=3) +
  geom_text(aes(x=Group, y=-0.01, label=paste("n =", scales::comma(n))), size=3, color="black") +
  geom_text(aes(x=Group, y=LTOUR+0.02, label=scales::percent(LTOUR)), size=3, color="blue") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "blue") +
  scale_fill_manual("Rule outcome", values=c("azure3", "azure4")) +
  scale_x_discrete("Rule outcome") +
  scale_y_continuous("Proportion that become long-term opioid users", labels=scales::percent, limits=c(-0.01,0.45)) +
  theme_bw() + theme(legend.position = "bottom")
# ggsave("C:/Users/elong/Dropbox/Interpretable Opioids/LTOUR/FeatureBars.pdf",  width = 6, height = 10)
ggsave(paste(export_path, "FeatureBars_horizontal.pdf", sep = ""),  width = 10, height = 6, dpi=300)



# ============================================================================
# ============================================================================
# ============================================================================


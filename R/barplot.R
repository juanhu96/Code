library(dplyr)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(patchwork)

setwd("/export/storage_cures/CURES/Results/")
export_path <- "../Plots/"

# FULL_LTOUR = read.csv("FULL_LTOUR_2019.csv", header = TRUE)
# LTOUR <- FULL_LTOUR %>% rename(top_prescriber_binary = prescriber_yr_avg_days_median_binary,
#                                top_pharmacy_binary = pharmacy_yr_avg_days_median_binary)
# binary_vars <- c("num_prior_prescriptions_binary", 
#                  "days_supply_binary",
#                  "daily_dose_binary",
#                  "HMFO_binary", 
#                  "long_acting_binary",
#                  "top_prescriber_binary", 
#                  "top_pharmacy_binary")

FULL_LTOUR = read.csv("FULL_LTOUR_6_2019.csv", header = TRUE)
LTOUR <- FULL_LTOUR %>% rename(top_prescriber_binary = prescriber_yr_avg_days_above75_binary)
binary_vars <- c("num_prior_prescriptions_binary", 
                 "days_supply_binary",
                 "daily_dose_binary",
                 "HMFO_binary", 
                 "long_acting_binary",
                 "top_prescriber_binary")
# ============================================================================

PRIOR = LTOUR %>% 
  group_by(num_prior_prescriptions_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = num_prior_prescriptions_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Prior opioid Rx")

PRESCRIBER = LTOUR %>% 
  group_by(top_prescriber_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = top_prescriber_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Top opioid prescriber")

# PHARMACY = LTOUR %>% 
#   group_by(top_pharmacy_binary) %>% 
#   summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
#   rename(Group = top_pharmacy_binary) %>%
#   mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Top opioid dispenser")

DAYS = LTOUR %>% 
  group_by(days_supply_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = days_supply_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Days of supply >= 10")

# MME = LTOUR %>% 
#   group_by(concurrent_MME_binary) %>% 
#   summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
#   rename(Group = concurrent_MME_binary) %>%
#   mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Total daily MME above 40")

MME = LTOUR %>% 
  group_by(daily_dose_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = daily_dose_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Daily MME >= 90")

# AGE = LTOUR %>% 
#   group_by(age_binary) %>% 
#   summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
#   rename(Group = age_binary) %>%
#   mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Age above 30 years")

HMFO = LTOUR %>% 
  group_by(HMFO_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = HMFO_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "HMFO")

LONGACT = LTOUR %>% 
  group_by(long_acting_binary) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = long_acting_binary) %>%
  mutate(Group = ifelse(Group == 1, "Yes", "No"), Feature = "Long acting opioid")


# SUMMARY = rbind(PRIOR, PRESCRIBER, PHARMACY, MME, AGE, LONGACT)
# SUMMARY$Feature = factor(SUMMARY$Feature, levels=c("Prior opioid prescription", "Top opioid prescriber", "Top opioid dispenser", "Total daily MME above 40", "Age above 30 years", "Long acting opioid"))
SUMMARY = rbind(PRIOR, PRESCRIBER, DAYS, MME, HMFO, LONGACT)
SUMMARY$Feature = factor(SUMMARY$Feature, levels=c("Prior opioid Rx", 
                                                   "Days of supply >= 10",
                                                   "Daily MME >= 90",
                                                   "HMFO",
                                                   "Long acting opioid",
                                                   "Top opioid prescriber"))

SUMMARY <- SUMMARY %>% select(Group, n, PropPos, LTOUR, Feature)

ggplot(data=SUMMARY) + facet_wrap(~Feature, nrow=1) +
  geom_col(aes(x=Group, y=PropPos, fill=Group), width=0.35) +
  geom_point(aes(x=Group, y=LTOUR, color=""), size=3) +
  geom_text(aes(x=Group, y=-0.03, label=paste("", scales::comma(n))), size=4.1, color="black") +
  geom_text(aes(x=Group, y=LTOUR+0.048, label=scales::percent(LTOUR, accuracy = 0.1)), size=4.2, color="black") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "black") +
  scale_fill_manual("Rule outcome", values=c("azure3", "azure4"), guide = "none") +
  scale_x_discrete("Rule outcome") +
  scale_y_continuous("Proportion that become \n long-term opioid users", labels=scales::percent, limits=c(-0.05,0.55)) +
  theme_bw(base_size = 16) + theme(legend.position = "bottom", strip.text = element_text(size = 14))

# ggplot(data = SUMMARY) + 
#   facet_wrap(~Feature, ncol = 7) +
#   geom_col(aes(x = Group, y = PropPos, fill = Group), width = 0.35, show.legend = FALSE) +  # hides bar legend
#   geom_point(aes(x = Group, y = LTOUR, color = ""), size = 3) +
#   geom_text(aes(x = Group, y = -0.03, label = paste("", scales::comma(n))), size = 4.1, color = "black") +
#   geom_text(aes(x = Group, y = LTOUR + 0.04, label = paste0(round(LTOUR * 100, 1), "%")), size = 4.2, color = "black") +
#   scale_color_manual("Mean predicted probability (LTOUR)", values = "black") +
#   scale_fill_manual(values = c("azure3", "azure4")) +  # no legend title
#   scale_x_discrete("Rule outcome") +
#   scale_y_continuous("Proportion that become \n long-term opioid users", labels = scales::percent, limits = c(-0.05, 0.51)) +
#   theme_bw(base_size = 16) +
#   theme(legend.position = "bottom", strip.text = element_text(size = 14))

ggsave(paste(export_path, "FeatureBars.pdf", sep = ""),  width = 15.25, height = 5, dpi=500)

# ggplot(data=SUMMARY) + facet_wrap(~Feature, nrow=2) +
#   geom_col(aes(x=Group, y=PropPos, fill=Group), width=0.5) +
#   geom_point(aes(x=Group, y=LTOUR, color=""), size=3) +
#   geom_text(aes(x=Group, y=-0.01, label=paste("n =", scales::comma(n))), size=3, color="black") +
#   geom_text(aes(x=Group, y=LTOUR+0.04, label=scales::percent(LTOUR)), size=3, color="blue") +
#   scale_color_manual("Mean predicted probability (LTOUR)", values = "blue") +
#   scale_fill_manual("Rule outcome", values=c("azure3", "azure4")) +
#   scale_x_discrete("Rule outcome") +
#   scale_y_continuous("Proportion that become long-term opioid users", labels=scales::percent, limits=c(-0.01,0.5)) +
#   theme_bw() + theme(legend.position = "bottom")
# ggsave(paste(export_path, "FeatureBars_horizontal.pdf", sep = ""),  width = 10, height = 6, dpi=300)


# ============================================================================
# ============================================================================
# ============================================================================

### BY COUNTY
CA = read.csv("../CA/California_DemographicsByZip2020.csv", header = TRUE) %>% rename(zip = X......name) %>% select(zip, county_name)
CA$county <- sub(";.*", "", CA$county)
CA <- CA %>% select(zip, county)
# write.csv(CA, "../CA/zip_county.csv", row.names = FALSE)

MERGED <- merge(CA, FULL_LTOUR, by.x = "zip", by.y = "patient_zip", all = FALSE)

COUNTY_SUMMARY <- MERGED %>% group_by(county) %>%
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  mutate(county = reorder(county, PropPos))
write.csv(COUNTY_SUMMARY, "County_Summary.csv", row.names = FALSE)

# county_list = c("Lasson", "Butte", "Kern", "Shasta",
#                 "San Bernardino", "Stanislaus", "Riverside",
#                 "San Joaquin", "Sacremento", "Placer", "Fresno", "San Diego",
#                 "Orange", "Los Angeles", "San Francisco", "Alameda",
#                 "San Mateo", "Santa Clara")
# COUNTY_SUMMARY <- COUNTY_SUMMARY %>% filter(county %in% county_list)

ggplot(COUNTY_SUMMARY, aes(x = PropPos * 100, y = county)) +
  geom_col(aes(x=PropPos * 100, y=county), width=0.5, alpha = 0.5) +
  geom_point(aes(x=LTOUR * 100, y=county, color=""), size=3) +
  geom_text(aes(x=-2, y=county, label=paste("n =", scales::comma(n))), size=3, color="black") +
  geom_text(aes(x = max(PropPos * 100, LTOUR) + 2, y=county, label=scales::percent(LTOUR, accuracy = 0.1)), size=3, color="black") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "black") +
  scale_y_discrete("County") +
  labs(x = "Proportion Positive", y = "County") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )
ggsave(paste(export_path, "CountyBars_Sorted.pdf", sep = ""),  width = 11, height = 14, dpi=300)
# ggsave(paste(export_path, "CountyBars_Sorted_Selected.pdf", sep = ""),  width = 10, height = 8, dpi=300)


ggplot(COUNTY_SUMMARY, aes(x = PropPos, y = LTOUR, size = n, label = county)) +
  geom_point(alpha = 0.7, color = "#2774AE") +
  scale_size_continuous(name = "#Prescriptions", range = c(0.5, 15)) +
  geom_text_repel(size = 2, max.overlaps = Inf) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  coord_fixed(xlim = c(0, 0.2), ylim = c(0, 0.2), ratio = 1) +
  labs(x = "Proportion Positive", y = "Mean predicted probability (LTOUR)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )
ggsave(paste(export_path, "CountyScatter.pdf", sep = ""),  width = 8, height = 7, dpi=300)


# ============================================================================
### OTHER TABLES
CA = read.csv("../CA/zip_county.csv", header = TRUE)

TABLE_KERN = read.csv("FULL_LTOUR_TableKern.csv", header = TRUE) 
TABLE_SF = read.csv("FULL_LTOUR_TableSF.csv", header = TRUE) 

MERGED_KERN <- merge(CA, TABLE_KERN, by.x = "zip", by.y = "patient_zip", all = FALSE)
MERGED_SF <- merge(CA, TABLE_SF, by.x = "zip", by.y = "patient_zip", all = FALSE)

COUNTY_SUMMARY_KERN <- MERGED_KERN %>% group_by(county) %>%
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  mutate(county = reorder(county, PropPos))
COUNTY_SUMMARY_SF <- MERGED_SF %>% group_by(county) %>%
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  mutate(county = reorder(county, PropPos))

ggplot(COUNTY_SUMMARY_KERN, aes(x = PropPos * 100, y = county)) +
  geom_col(aes(x=PropPos * 100, y=county), width=0.5, alpha = 0.5) +
  geom_point(aes(x=LTOUR * 100, y=county, color=""), size=3) +
  geom_text(aes(x=-2, y=county, label=paste("n =", scales::comma(n))), size=3, color="black") +
  geom_text(aes(x = max(PropPos * 100, LTOUR) + 2, y=county, label=scales::percent(LTOUR)), size=3, color="blue") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "blue") +
  scale_y_discrete("County") +
  labs(x = "Proportion Positive", y = "County") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )
ggsave(paste(export_path, "CountyBars_Kern.pdf", sep = ""),  width = 11, height = 14, dpi=300)


ggplot(COUNTY_SUMMARY_SF, aes(x = PropPos * 100, y = county)) +
  geom_col(aes(x=PropPos * 100, y=county), width=0.5, alpha = 0.5) +
  geom_point(aes(x=LTOUR * 100, y=county, color=""), size=3) +
  geom_text(aes(x=-2, y=county, label=paste("n =", scales::comma(n))), size=3, color="black") +
  geom_text(aes(x = max(PropPos * 100, LTOUR) + 2, y=county, label=scales::percent(LTOUR)), size=3, color="blue") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "blue") +
  scale_y_discrete("County") +
  labs(x = "Proportion Positive", y = "County") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )
ggsave(paste(export_path, "CountyBars_SF.pdf", sep = ""),  width = 11, height = 14, dpi=300)


# ============================================================================
### BY GENDER

GENDER = LTOUR %>% 
  group_by(patient_gender) %>% 
  summarize(TruePos = sum(True), n = n(), PropPos = TruePos/n, LTOUR = round(mean(Prob), digits=3)) %>%
  rename(Group = patient_gender) %>%
  mutate(Group = ifelse(Group == 1, "Female", "Male"), Feature = "Gender")

ggplot(data=GENDER) +
  geom_col(aes(x=Group, y=PropPos, fill=Group), width=0.35) +
  geom_point(aes(x=Group, y=LTOUR, color=""), size=3) +
  geom_text(aes(x=Group, y=-0.01, label=paste("n = ", scales::comma(n))), size=5.2, color="black") +
  geom_text(aes(x=Group, y=LTOUR+0.01, label=scales::percent(LTOUR, accuracy = 0.1)), size=5.2, color="black") +
  scale_color_manual("Mean predicted probability (LTOUR)", values = "black") +
  scale_fill_manual("Patient Gender", values=c("azure3", "azure4"), guide = "none") +
  scale_x_discrete("Patient Gender") +
  scale_y_continuous("Proportion that become \n long-term opioid users", labels=scales::percent, limits=c(-0.015,0.13)) +
  theme_bw(base_size = 16) + theme(legend.position = "bottom", strip.text = element_text(size = 16))

ggsave(paste(export_path, "GenderBar.pdf", sep = ""),  width = 10, height = 6, dpi=500)


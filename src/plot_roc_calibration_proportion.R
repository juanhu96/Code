################################################################################
################################# PLOT ROC #####################################
################################################################################
library(ggplot2)
library(dplyr)
library(readr)
library(tibble)

csv_dir <- "/export/storage_cures/CURES/Results_R/"
export_path <- "/export/storage_cures/CURES/Plots/"
files <- list.files(csv_dir, pattern = "roc.*\\.csv$", full.names = TRUE)

roc_data <- lapply(files, function(f) {
  df <- read_csv(f)
  df$Model <- gsub("_roc.*", "", basename(f))  # Extract model name
  df
}) %>% bind_rows()

# dplyr::count(roc_data, Model)

# roc_data <- roc_data %>% mutate(Presc = factor(Presc, levels = c("All", "Naive")), PrescLabel = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"))
roc_data <- roc_data %>% mutate(Presc = recode(Presc, "All" = "All Rx", "Naive" = "1st Rx"), 
                                Presc = factor(Presc, levels = c("All Rx", "1st Rx")), 
                                PrescLabel = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"))
roc_data$Model <- factor(roc_data$Model, levels = c("LTOUR", "DecisionTree", "RandomForest", "Logistic", "L1", "L2", "LinearSVM", "NN", "XGB"))
roc_data <- roc_data %>%
  mutate(Model = recode(Model,
                        "DecisionTree" = "Decision Tree",
                        "RandomForest" = "Random Forest",
                        "Logistic" = "Logistic",
                        "L1" = "L1 Logistic",
                        "L2" = "L2 Logistic",
                        "LinearSVM" = "Linear SVM",
                        "NN" = "Neural Network",
                        "XGB" = "XGBoost")) %>%
  filter(Model %in% c("LTOUR", "Decision Tree", "Random Forest", "Logistic", "L1 Logistic", "L2 Logistic", "Linear SVM", "Neural Network", "XGBoost"))

roc_labels <- roc_data %>%
  group_by(Model, Presc) %>%
  summarize(auc = mean(auc), .groups = "drop") %>%
  mutate(
    label = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"),
    x = 0.55,   # X/Y position for text
    y = ifelse(Presc == "All Rx", 0.15, 0.05)
  )

# ALL
ggplot(roc_data,
       aes(x = fpr, y = tpr, color = Presc, linetype = Presc)) +
  geom_abline(linetype = "dotted", color = "black") +
  geom_line(linewidth = 0.8) +
  geom_text(data = roc_labels, aes(x = x, y = y, label = label, color = Presc),
            inherit.aes = FALSE, size = 3.5, hjust = 0) +
  facet_wrap(~Model, nrow = 3)+#, scales = "free") +
  scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
  scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  guides(
    color = "none",
    linetype = "none"
  ) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom", strip.text = element_text(size = 12))

ggsave(paste(export_path, "ROC_full.pdf", sep = ""),  width = 11, height = 10.5, dpi=500)


# SELECTED

roc_labels <- roc_data %>%
  group_by(Model, Presc) %>%
  summarize(auc = mean(auc), .groups = "drop") %>%
  mutate(
    label = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"),
    x = 0.45,   # X/Y position for text
    y = ifelse(Presc == "All Rx", 0.15, 0.05)
  )

ggplot(roc_data %>% filter(Model %in% c("LTOUR", "Decision Tree", "Logistic", "Neural Network")),
       aes(x = fpr, y = tpr, color = Presc, linetype = Presc)) +
  geom_abline(linetype = "dotted", color = "black") +
  geom_line(linewidth = 0.8) +
  geom_text(data = roc_labels %>% filter(Model %in% c("LTOUR", "Decision Tree", "Logistic", "Neural Network")), 
            aes(x = x, y = y, label = label, color = Presc),
            inherit.aes = FALSE, size = 3.5, hjust = 0) +
  facet_wrap(~Model, nrow = 1) +
  scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
  scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  guides(
    color = "none",
    linetype = "none"
  ) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom", strip.text = element_text(size = 12))

ggsave(paste(export_path, "ROC_selected.pdf", sep = ""),  width = 12, height = 3.5, dpi=500)


################################################################################
############################ PLOT CALIBRATION ##################################
################################################################################
library(ggplot2)
library(dplyr)
library(readr)
library(tibble)


csv_dir <- "/export/storage_cures/CURES/Results_R/"
export_path <- "/export/storage_cures/CURES/Plots/"

files <- list.files(csv_dir, pattern = "calibration.*\\.csv$", full.names = TRUE)

calibration_data <- lapply(files, function(f) {
  df <- read_csv(f)
  df$Model <- gsub("_calibration.*", "", basename(f))
  df
}) %>% bind_rows()

# calibration_data <- calibration_data %>% mutate(Presc = factor(Presc, levels = c("All", "Naive")), PrescLabel = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"))
calibration_data <- calibration_data %>% mutate(Presc = recode(Presc, "All" = "All Rx", "Naive" = "1st Rx"), 
                                                Presc = factor(Presc, levels = c("All Rx", "1st Rx")), 
                                                PrescLabel = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"))

calibration_data$Model <- factor(calibration_data$Model, levels = c("LTOUR", "DecisionTree", "RandomForest", "Logistic", "L1", "L2", "LinearSVM", "NN", "XGB"))
calibration_data <- calibration_data %>%
  mutate(Model = recode(Model,
                        "DecisionTree" = "Decision Tree",
                        "RandomForest" = "Random Forest",
                        "Logistic" = "Logistic",
                        "L1" = "L1 Logistic",
                        "L2" = "L2 Logistic",
                        "LinearSVM" = "Linear SVM",
                        "NN" = "Neural Network",
                        "XGB" = "XGBoost")) %>% 
  filter(Model %in% c("LTOUR", "Decision Tree", "Random Forest", "Logistic", "L1 Logistic", "L2 Logistic", "Linear SVM", "Neural Network", "XGBoost"))

ece_labels <- calibration_data %>%
  group_by(Model, Presc) %>%
  summarize(ece = mean(ece), .groups = "drop") %>%
  mutate(
    label = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"),
    x = 0.02,   # X/Y position for text
    y = ifelse(Presc == "All Rx", 0.95, 0.87)
  )

# ALL
ggplot(calibration_data,
       aes(x = prob_pred, y = prob_true, color = Presc, linetype = Presc, size = observations / 1e6)) +
  geom_abline(linetype = "dotted", color = "black") +
  geom_line(linewidth = 0.8) +
  geom_point(alpha = 0.5) +
  geom_text(data = ece_labels, aes(x = x, y = y, label = label, color = Presc),
            inherit.aes = FALSE, size = 3.5, hjust = 0) +
  facet_wrap(~Model, nrow = 3) +
  scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
  scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
  scale_size_continuous(
    name = "Number of prescriptions (million)",
    range = c(1, 8),
    breaks = c(0.01, 0.1, 1, 2),
    labels = c("0.01", "0.1", "1", "2")
  ) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  labs(x = "Predicted Risk", y = "Observed Risk") +
  guides(
    color = "none",
    linetype = "none"
  ) +
  theme_bw(base_size = 14) +
  #theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",  strip.text = element_text(size = 12))

ggsave(paste(export_path, "Calibration_full.pdf", sep = ""),  width = 11, height = 11.2, dpi=500)


# SELECTED
ggplot(calibration_data %>% filter(Model %in% c("LTOUR", "Decision Tree", "Logistic", "Neural Network")),
       aes(x = prob_pred, y = prob_true, color = Presc, linetype = Presc, size = observations / 1e6)) +
  geom_abline(linetype = "dotted", color = "black") +
  geom_line(linewidth = 0.8) +
  geom_point(alpha = 0.5) +
  geom_text(data = ece_labels %>% filter(Model %in% c("LTOUR", "Decision Tree", "Logistic", "Neural Network")), aes(x = x, y = y, label = label, color = Presc),
            inherit.aes = FALSE, size = 3.5, hjust = 0) +
  facet_wrap(~Model, nrow = 1) +
  scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
  scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
  scale_size_continuous(
    name = "Number of prescriptions (million)",
    range = c(1, 8),
    breaks = c(0.01, 0.1, 1, 2),
    labels = c("0.01", "0.1", "1", "2")
  ) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  labs(x = "Predicted Risk", y = "Observed Risk") +
  guides(
    color = "none",
    linetype = "none"
  ) +
   theme_bw(base_size = 14) +
  theme(legend.position = "bottom",  strip.text = element_text(size = 12))

ggsave(paste(export_path, "Calibration_selected.pdf", sep = ""),  width = 12, height = 4.2, dpi=500)


################################################################################
############################# PLOT PROPORTION ##################################
################################################################################
library(ggplot2)
library(dplyr)
library(readr)
library(tibble)

csv_dir <- "/export/storage_cures/CURES/Results_R/"
export_path <- "/export/storage_cures/CURES/Plots/"

files <- list.files(csv_dir, pattern = "proportion.*\\.csv$", full.names = TRUE)

proportion_data <- lapply(files, function(f) {
  df <- read_csv(f)
  df$Model <- gsub("_proportion.*", "", basename(f))
  df
}) %>% bind_rows()

proportion_data <- proportion_data %>% filter(Presc == 'All')
proportion_data$month <- factor(proportion_data$month, levels = sort(unique(proportion_data$month)))
proportion_data$Model <- factor(proportion_data$Model, levels = c("LTOUR", "DecisionTree", "RandomForest", "Logistic", "L1", "L2", "LinearSVM", "NN", "XGB"))
proportion_data <- proportion_data %>%
  mutate(Model = recode(Model,
                        "DecisionTree" = "Decision Tree",
                        "RandomForest" = "Random Forest",
                        "Logistic" = "Logistic",
                        "L1" = "L1 Logistic",
                        "L2" = "L2 Logistic",
                        "LinearSVM" = "Linear SVM",
                        "NN" = "Neural Network",
                        "XGB" = "XGBoost"))

ggplot(proportion_data, aes(x = month, y = proportion, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(x = "Month", y = "Proportion (%)") +
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Paired") +
  theme(legend.position = "bottom", axis.text.x = element_text(hjust = 1))

ggsave(paste(export_path, "Proportion_full.pdf", sep = ""),  width = 10, height = 7, dpi=500)


################################################################################
############################# PLOT ROC COUNTY ##################################
################################################################################
library(ggplot2)
library(dplyr)
library(readr)
library(tibble)

csv_dir <- "/export/storage_cures/CURES/Results_R/"
export_path <- "/export/storage_cures/CURES/Plots/"

files <- list.files(csv_dir, pattern = "^LTOUR.*roc*\\.csv$", full.names = TRUE)
# files <- list.files(csv_dir, pattern = "^LTOUR_.*_roc(_naive)?\\.csv$", full.names = TRUE)
roc_data <- lapply(files, function(f) {
  df <- read_csv(f)
  df$Model <- gsub("_roc.*", "", basename(f))  # Extract model name
  df
}) %>% bind_rows()

roc_data <- roc_data %>% mutate(Presc = factor(Presc, levels = c("All", "Naive")), PrescLabel = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"))
roc_data <- roc_data %>% filter(!is.na(County))
roc_labels <- roc_data %>%
  group_by(County, Presc) %>%
  summarize(auc = mean(auc), .groups = "drop") %>%
  mutate(label = paste0(Presc, " (AUC = ", sprintf("%.3f", auc), ")"), 
         x = 0.6,
         y = ifelse(Presc == "All", 0.15, 0.05))

ggplot(roc_data %>% filter(!is.na(County)), aes(x = fpr, y = tpr, color = Presc, linetype = Presc)) +
  geom_abline(linetype = "dotted", color = "black") +
  geom_line(linewidth = 0.8) +
  geom_text(data = roc_labels, aes(x = x, y = y, label = label, color = Presc),
            inherit.aes = FALSE, size = 3.5, hjust = 0) +
  facet_wrap(~County, scales = "free") +
  scale_color_manual(values = c(All = "firebrick3", Naive = "mediumblue")) +
  scale_linetype_manual(values = c(All = "solid", Naive = "dashed")) +
  scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  labs(x = "False Positive Rate", y = "True Positive Rate") +
  guides(
    color = "none",
    linetype = "none"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

ggsave(paste(export_path, "ROC_county.pdf", sep = ""),  width = 10, height = 6, dpi=600)


################################################################################
######################## PLOT CALIBRATION COUNTY ###############################
################################################################################
library(ggplot2)
library(dplyr)
library(readr)
library(tibble)

csv_dir <- "/export/storage_cures/CURES/Results_R/"
export_path <- "/export/storage_cures/CURES/Plots/"

files <- list.files(csv_dir, pattern = "^LTOUR.*calibration.*\\.csv$", full.names = TRUE)
# files <- list.files(csv_dir, pattern = "^LTOUR_.*calibration(_naive)?\\.csv$", full.names = TRUE)
files <- files[files != "/export/storage_cures/CURES/Results_R//LTOUR_AllCounties_calibration.csv"]

calibration_data <- lapply(files, function(f) {
  df <- read_csv(f)
  df$Model <- gsub("_calibration.*", "", basename(f))
  df
}) %>% bind_rows()

calibration_data <- calibration_data %>% mutate(Presc = recode(Presc, "All" = "All Rx", "Naive" = "1st Rx"),
                                                Presc = factor(Presc, levels = c("All Rx", "1st Rx")), 
                                                PrescLabel = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"))
calibration_data <- calibration_data %>% filter(!is.na(County))

# counties1 <- unique(calibration_data$County)[1:16]
# counties2 <- unique(calibration_data$County)[17:32]
# counties3 <- unique(calibration_data$County)[33:48]
# counties4 <- unique(calibration_data$County)[49:58]
# county_groups <- list(counties1, counties2, counties3, counties4)
# group_names <- c("county1", "county2", "county3", "county4")

counties1 <- unique(calibration_data$County)[1:20]
counties2 <- unique(calibration_data$County)[21:40]
counties3 <- unique(calibration_data$County)[41:58]
county_groups <- list(counties1, counties2, counties3)
group_names <- c("county1", "county2", "county3")

for (i in seq_along(county_groups)) {
  current_counties <- county_groups[[i]]
  group_name <- group_names[[i]]
  
  calibration_subset <- calibration_data[calibration_data$County %in% current_counties, ]
  
  ece_labels <- calibration_subset %>%
    group_by(County, Presc) %>%
    summarize(ece = mean(ece), .groups = "drop") %>%
    mutate(
      label = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"),
      x = 0.02,
      y = ifelse(Presc == "All Rx", 0.95, 0.87)
    )
  
  p <- ggplot(calibration_subset,
              aes(x = prob_pred, y = prob_true, color = Presc, linetype = Presc, size = observations / 1e6)) +
    geom_abline(linetype = "dotted", color = "black") +
    geom_line(linewidth = 0.8) +
    geom_point(alpha = 0.5) +
    geom_text(data = ece_labels, aes(x = x, y = y, label = label, color = Presc),
              inherit.aes = FALSE, size = 3.5, hjust = 0) +
    facet_wrap(~County, nrow = 5) +
    scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
    scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
    scale_size_continuous(name = "Number of prescriptions (million)",
                          range = c(2, 8.5),
                          limits = c(0, 0.5),  # FIXED scale from 0 to 1 million
                          breaks = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
                          labels = c("0.001", "0.005", "0.01", "0.05", "0.1", "0.5")) +
    scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
    labs(x = "Predicted Risk", y = "Observed Risk") +
    guides(color = "none", linetype = "none") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom", strip.text = element_text(size = 12))
    
  # if (group_name == "county4") {
  #   ggsave(filename = paste0(export_path, "Calibration_", group_name, ".pdf"), plot = p, width = 11, height = 9.5, dpi = 500)
  #   } else {
  #     ggsave(filename = paste0(export_path, "Calibration_", group_name, ".pdf"), plot = p, width = 11, height = 11.4, dpi = 500)
  #   }
  
  ggsave(filename = paste0(export_path, "Calibration_", group_name, "_new.pdf"), plot = p, width = 11, height = 14, dpi = 500)
}



# calibration_subset <- calibration_data[calibration_data$County %in% counties1, ]
# 
# ece_labels <- calibration_subset %>%
#   group_by(County, Presc) %>%
#   summarize(ece = mean(ece), .groups = "drop") %>%
#   mutate(
#     label = paste0(Presc, " (ECE = ", sprintf("%.3f", ece), ")"),
#     x = 0.02,   # X/Y position for text
#     y = ifelse(Presc == "All Rx", 0.95, 0.87)
#   )

# ggplot(calibration_subset,
#        aes(x = prob_pred, y = prob_true, color = Presc, linetype = Presc, size = observations / 1e6)) +
#   geom_abline(linetype = "dotted", color = "black") +
#   geom_line(linewidth = 0.8) +
#   geom_point(alpha = 0.5) +
#   geom_text(data = ece_labels, aes(x = x, y = y, label = label, color = Presc),
#             inherit.aes = FALSE, size = 3.5, hjust = 0) +
#   facet_wrap(~County) +
#   scale_color_manual(values = c("All Rx" = "firebrick3", "1st Rx" = "mediumblue")) +
#   scale_linetype_manual(values = c("All Rx" = "solid", "1st Rx" = "dashed")) +
#   scale_size_continuous(
#     name = "Number of prescriptions (million)",
#     range = c(1, 8),
#     breaks = c(0.01, 0.05, 0.1, 0.5, 1),
#     labels = c("0.01", "0.05", "0.1", "0.5", "1")
#   ) +
#   scale_x_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
#   scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
#   labs(x = "Predicted Risk", y = "Observed Risk") +
#   guides(
#     color = "none",
#     linetype = "none"
#   ) +
#   theme_minimal(base_size = 14) +
#   theme(legend.position = "bottom", strip.text = element_text(size = 12))
# 
# ggsave(paste(export_path, "Calibration_county.pdf", sep = ""),  width = 10, height = 6, dpi=500)

library(dplyr)
setwd("/export/storage_cures/CURES/Processed/")

zip_county = read.csv("../CA/zip_county.csv", header = TRUE)

year = 2018
RAW <- read.csv(paste("../RX_", year, ".csv", sep="")) 

# this is tricky - could be two prescriptions rather than mistakes
# DUPLICATES <- RAW %>% group_by(across(-X)) %>% filter(n() > 1) %>% arrange(across(-X))  
# DUPLICATES %>% distinct(across(-X)) %>% nrow()

MERGED <- merge(zip_county, RAW, by.x = "zip", by.y = "patient_zip", all = FALSE)
# LA <- MERGED %>% filter(county == 'Los Angeles')
# LA %>% select(-X) %>% distinct() %>% nrow()

# LA_DUPLICATES <- LA %>% group_by(across(-X)) %>% filter(n() > 1) %>% arrange(across(-X))
# LA_DUPLICATES %>% distinct(across(-X)) %>% nrow()


#===============================================================================

SUMMARY <- MERGED %>%
  select(-X) %>%
  group_by(county) %>%
  summarise(
    num_prescriptions = n(),
    num_unique_patients = n_distinct(patient_id),
    num_unique_rows = n_distinct(pick(everything())),
    num_duplicates = num_prescriptions - num_unique_rows
  )


write.csv(SUMMARY, paste("../CA/Raw_duplicate_", year, ".csv", sep=""), row.names = FALSE)



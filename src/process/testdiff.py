import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import sys


def print_summary(filename, 
                  datadir_new = "/export/storage_cures/CURES/Processed/",
                  datadir_old = "/export/storage_cures/CURES/Processed(archive)/",
                  export_sample=True):
    
    print(f"Comparing {filename}...")
    
    FULL_NEW = pd.read_csv(f"{datadir_new}{filename}")
    FULL_OLD = pd.read_csv(f"{datadir_old}{filename}")
    print(f"FULL_NEW: {FULL_NEW.shape}, FULL_OLD: {FULL_OLD.shape}")
    print(FULL_NEW.head(), FULL_OLD.head())

    # Check for missing columns
    missing_columns_new = set(FULL_NEW.columns) - set(FULL_OLD.columns)
    missing_columns_old = set(FULL_OLD.columns) - set(FULL_NEW.columns)
    if missing_columns_new:
        print(f"In new but not old: {missing_columns_new}")
    if missing_columns_old:
        print(f"In old but not new: {missing_columns_old}")  

    if export_sample:
        # export the first 50 rows of each file
        print(f"Exporting first 50 rows of each file...")
        
        filename_sample = filename.replace('.csv', '_sample.csv')
        FULL_NEW.iloc[:50].to_csv(f"{datadir_new}{filename_sample}", index=False)
        FULL_OLD.iloc[:50].to_csv(f"{datadir_old}{filename_sample}", index=False)  

    return


# print_summary("FULL_OPIOID_2018_ONE.csv")
# print_summary("FULL_OPIOID_2018_ATLEASTTWO_1.csv")
# print_summary("FULL_OPIOID_2018_ONE_TEMP.csv")
# print_summary("FULL_OPIOID_2018_ATLEASTTWO_1_TEMP.csv")
# print_summary("FULL_OPIOID_2019_ONE.csv")
# print_summary("FULL_OPIOID_2019_ONE_TEMP.csv")
print_summary("FULL_OPIOID_2019_INPUT.csv")


def compare_cols(year,
                 datadir_new = "/export/storage_cures/CURES/Processed/",
                 datadir_old = "/export/storage_cures/CURES/Processed(archive)/"):


    FULL_NEW = pd.read_csv(f"{datadir_new}FULL_OPIOID_{year}_FEATURE.csv")
    FULL_OLD = pd.read_csv(f"{datadir_old}FULL_OPIOID_{year}_INPUT.csv")

    columns_new = set(FULL_NEW.columns) - set(FULL_OLD.columns)
    columns_old = set(FULL_OLD.columns) - set(FULL_NEW.columns)
    
    if columns_new:
        print(f"In new but not old: {columns_new}")
    if columns_old:
        print(f"In old but not new: {columns_old}")

    return

# compare_cols(2018)
'''
STEP 7
FINAL CHECK ON INPUT DATA FOR RISKSLIM

INPUT: FULL_OPIOID_2018_INPUT.csv
'''

from multiprocessing import Pool
import pandas as pd
import numpy as np
import sys

year = int(sys.argv[1])
datadir = "/export/storage_cures/CURES/Processed/"
FULL_INPUT = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_INPUT.csv")

# Summary stats from the 2019 file
age_min = FULL_INPUT['age'].min()
age_max = FULL_INPUT['age'].max()
quantity_max = FULL_INPUT['quantity'].max()
concurrent_MME_max = FULL_INPUT['concurrent_MME'].max()
num_prescribers_past180_max = FULL_INPUT['num_prescribers_past180'].max()
num_pharmacies_past180_max = FULL_INPUT['num_pharmacies_past180'].max()
num_prior_prescriptions_max = FULL_INPUT['num_prior_prescriptions'].max()

# Print for inspection
print(f"Summary Statistics from {year}:")
print(f"Min age: {age_min}")
print(f"Max age: {age_max}")
print(f"Max quantity: {quantity_max}")
print(f"Max concurrent MME: {concurrent_MME_max}")
print(f"Max prescribers past 180 days: {num_prescribers_past180_max}")
print(f"Max pharmacies past 180 days: {num_pharmacies_past180_max}")
print(f"Max prior prescriptions: {num_prior_prescriptions_max}")


# get the percentile of concurrent MME
concurrent_MME_percentiles = FULL_INPUT['concurrent_MME'].quantile([i / 100 for i in range(10, 100, 10)])
dailydose_percentiles = FULL_INPUT['daily_dose'].quantile([i / 100 for i in range(10, 100, 10)])
print(f"Percentiles of concurrent MME: {concurrent_MME_percentiles} \n")
print(f"Percentiles of daily dose: {dailydose_percentiles} \n")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2024
Compute spatial effect

@author: Jingyuan Hu
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

pd.set_option('display.max_columns', None) # show all columns

datadir = "/export/storage_cures/CURES/Processed/"
exportdir = "/export/storage_cures/CURES/Processed/Patient_zip/"
year = 2018

# Import
FULL_ONE = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ONE_FEATURE.csv")
FULL_ATLEASTTWO_1 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_1_FEATURE.csv")
FULL_ATLEASTTWO_2 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_2_FEATURE.csv")
FULL_ATLEASTTWO_3 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_3_FEATURE.csv")
FULL_ATLEASTTWO_4 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_4_FEATURE.csv")

FULL_ONE['patient_zip'] = FULL_ONE['patient_zip'].astype(str)
FULL_ATLEASTTWO_1['patient_zip'] = FULL_ATLEASTTWO_1['patient_zip'].astype(str)
FULL_ATLEASTTWO_2['patient_zip'] = FULL_ATLEASTTWO_2['patient_zip'].astype(str)
FULL_ATLEASTTWO_3['patient_zip'] = FULL_ATLEASTTWO_3['patient_zip'].astype(str)
FULL_ATLEASTTWO_4['patient_zip'] = FULL_ATLEASTTWO_4['patient_zip'].astype(str)
FULL = pd.concat([FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4], ignore_index=True)

del FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4 # free up memory (optional)

# FULL = FULL.iloc[0:10000, :] # test

PATIENT_ZIP = FULL.groupby(['patient_zip', 'date_filled']).agg(
    day_prescriptions=('prescription_id', 'count'), 
    day_patients=('patient_id', pd.Series.nunique)
).reset_index()

# PATIENT_ZIP.to_csv(f"{exportdir}PATIENT_ZIP_{year}_table_TEMP.csv", index=False)

# Compute patient zip spatial effect
def compute_patient_zip_info(FULL, patient_zip, date_filled):

    date_filled = pd.to_datetime(date_filled, format="%m/%d/%Y")
    window_start = date_filled - pd.Timedelta(days=180)
    window_end = date_filled

    PATIENT_ZIP_ACTIVE_PRESC = FULL[(FULL['patient_zip'] == patient_zip) &
                                   (pd.to_datetime(FULL['date_filled'], format="%m/%d/%Y") >= window_start) &
                                   (pd.to_datetime(FULL['date_filled'], format="%m/%d/%Y") <= window_end)]

    num_prescriptions = len(PATIENT_ZIP_ACTIVE_PRESC)
    num_patients = PATIENT_ZIP_ACTIVE_PRESC['patient_id'].nunique()
    patient_zip_avg_days = PATIENT_ZIP_ACTIVE_PRESC['days_supply'].mean()
    patient_zip_avg_quantity = PATIENT_ZIP_ACTIVE_PRESC['quantity'].mean()
    patient_zip_avg_MME = PATIENT_ZIP_ACTIVE_PRESC['daily_dose'].mean()

    return (num_prescriptions, num_patients, patient_zip_avg_days, patient_zip_avg_quantity, patient_zip_avg_MME)

def parallel_processing(FULL, PATIENT_ZIP, num_cores=50):

    with Pool(num_cores) as pool:
        results = pool.starmap(partial(compute_patient_zip_info, FULL), zip(PATIENT_ZIP['patient_zip'], PATIENT_ZIP['date_filled']))

    return results

# Split into chunks
# prob_list = np.linspace(0.0, 1.0, num=21, endpoint=False)
# patient_zip_list = PATIENT_ZIP['patient_zip'].quantile(prob_list).round().astype(int).values
# for index, cutoff in enumerate(patient_zip_list):

#     if index < len(patient_zip_list) - 1: # not the last one
#         next_cutoff = patient_zip_list[index + 1]
#         PATIENT_ZIP_CURRENT = PATIENT_ZIP[(PATIENT_ZIP['patient_zip'] > cutoff) & (PATIENT_ZIP['patient_zip'] <= next_cutoff)]
#         FULL_CURRENT = FULL[(FULL['patient_zip'] > cutoff) & (FULL['patient_zip'] <= next_cutoff)]
#     else:
#         PATIENT_ZIP_CURRENT = PATIENT_ZIP[PATIENT_ZIP['patient_zip'] > cutoff]
#         FULL_CURRENT = FULL[FULL['patient_zip'] > cutoff]
#     print(f'Processing the {index} cutoff {cutoff}; {len(PATIENT_ZIP_CURRENT)} unique patient_zips and {len(FULL_CURRENT)} prescriptions\n')

patient_zip_list = np.array_split(PATIENT_ZIP, 20)

for index, PATIENT_ZIP_CURRENT in enumerate(patient_zip_list):   
     
    FULL_CURRENT = FULL[FULL['patient_zip'].isin(PATIENT_ZIP_CURRENT['patient_zip'])]
    print(f'Processing the {index}; {len(PATIENT_ZIP_CURRENT)} unique patient_zips and {len(FULL_CURRENT)} prescriptions\n')

    results = parallel_processing(FULL_CURRENT, PATIENT_ZIP_CURRENT)
    results_df = pd.DataFrame(results, columns=['patient_zip_num_prescriptions', 'patient_zip_num_patients', 'patient_zip_avg_days', 'patient_zip_avg_quantity', 'patient_zip_avg_MME'])

    PATIENT_ZIP_CURRENT = pd.concat([PATIENT_ZIP_CURRENT.reset_index(drop=True), results_df], axis=1)
    PATIENT_ZIP_CURRENT = PATIENT_ZIP_CURRENT.drop(columns=['day_prescriptions', 'day_patients'])
    PATIENT_ZIP_CURRENT.to_csv(f"{exportdir}PATIENT_ZIP_{year}_{index}_TEMP.csv", index=False)

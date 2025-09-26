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

# FULL = FULL.iloc[0:1000, :] # test

PRESCRIBER = FULL.groupby(['prescriber_id', 'date_filled']).agg(
    day_prescriptions=('prescription_id', 'count'), 
    day_patients=('patient_id', pd.Series.nunique)
).reset_index()


# Compute prescriber spatial effect
def compute_prescriber_info(FULL, prescriber_id, date_filled):

    date_filled = pd.to_datetime(date_filled, format="%m/%d/%Y")
    window_start = date_filled - pd.Timedelta(days=31)
    window_end = date_filled

    PRESCRIBER_ACTIVE_PRESC = FULL[(FULL['prescriber_id'] == prescriber_id) &
                                   (pd.to_datetime(FULL['date_filled'], format="%m/%d/%Y") >= window_start) &
                                   (pd.to_datetime(FULL['date_filled'], format="%m/%d/%Y") <= window_end)]

    num_prescriptions = len(PRESCRIBER_ACTIVE_PRESC)
    monthly_patients = PRESCRIBER_ACTIVE_PRESC['patient_id'].nunique()
    prescriber_avg_days = PRESCRIBER_ACTIVE_PRESC['days_supply'].mean()
    prescriber_avg_quantity = PRESCRIBER_ACTIVE_PRESC['quantity'].mean()
    prescriber_avg_MME = PRESCRIBER_ACTIVE_PRESC['daily_dose'].mean()

    return (num_prescriptions, monthly_patients, prescriber_avg_days, prescriber_avg_quantity, prescriber_avg_MME)

def parallel_processing(FULL, PRESCRIBER, num_cores=50):

    with Pool(num_cores) as pool:
        results = pool.starmap(partial(compute_prescriber_info, FULL), zip(PRESCRIBER['prescriber_id'], PRESCRIBER['date_filled']))

    return results

# Split into chunks
prob_list = np.linspace(0.0, 1.0, num=21, endpoint=False)
prescriber_id_list = PRESCRIBER['prescriber_id'].quantile(prob_list).round().astype(int).values

for index, cutoff in enumerate(prescriber_id_list):

    if index < len(prescriber_id_list) - 1: # not the last one
        next_cutoff = prescriber_id_list[index + 1]
        PRESCRIBER_CURRENT = PRESCRIBER[(PRESCRIBER['prescriber_id'] > cutoff) & (PRESCRIBER['prescriber_id'] <= next_cutoff)]
        FULL_CURRENT = FULL[(FULL['prescriber_id'] > cutoff) & (FULL['prescriber_id'] <= next_cutoff)]
    else:
        PRESCRIBER_CURRENT = PRESCRIBER[PRESCRIBER['prescriber_id'] > cutoff]
        FULL_CURRENT = FULL[FULL['prescriber_id'] > cutoff]

    print(f'Processing the {index} cutoff {cutoff}; {len(PRESCRIBER_CURRENT)} unique prescribers and {len(FULL_CURRENT)} prescriptions\n')

    results = parallel_processing(FULL_CURRENT, PRESCRIBER_CURRENT)
    results_df = pd.DataFrame(results, columns=['prescriber_monthly_prescriptions', 'prescriber_monthly_patients', 'prescriber_monthly_avg_days', 'prescriber_monthly_avg_quantity', 'prescriber_monthly_avg_MME'])

    PRESCRIBER_CURRENT = pd.concat([PRESCRIBER_CURRENT.reset_index(drop=True), results_df], axis=1)
    PRESCRIBER_CURRENT = PRESCRIBER_CURRENT.drop(columns=['day_prescriptions', 'day_patients'])
    PRESCRIBER_CURRENT.to_csv(f"{datadir}PRESCRIBER_{year}_{index}_TEMP.csv", index=False)

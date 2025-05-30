'''
STEP 2
Identify CURES alert (for patients w/ single prescriptions)

STEP 3
Identify longterm & longterm180 (for patients w/ single prescriptions)
Keep prescriptions up to first long term

INPUT: FULL_OPIOID_2018_ONE.csv
OUTPUT: FULL_OPIOID_2018_ONE_TEMP.csv
'''

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import sys
cores = 8 # 256 in total


year = int(sys.argv[1])
datadir = "/export/storage_cures/CURES/Processed/"
FULL_SINGLE = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ONE.csv")
BENZO_TABLE = pd.read_csv(f"{datadir}FULL_BENZO_{year}.csv")


# === Add metadata
# Patient alerts if any of the following satisfies:
# More than 90 MME (daily dose) per day
# More than 40 MME (daily dose) methadone
# 6 or more perscribers last 6 months
# 6 or more pharmacies last 6 months
# On opioids more than 90 consecutive days
# On both benzos and opioids

FULL_SINGLE['prescription_id'] = range(1, len(FULL_SINGLE) + 1)
FULL_SINGLE['age'] = pd.to_datetime(FULL_SINGLE['date_filled'], errors='coerce').dt.year - FULL_SINGLE['patient_birth_year']
FULL_SINGLE['concurrent_MME'] = FULL_SINGLE['daily_dose']
FULL_SINGLE['adjusted_MME'] = FULL_SINGLE['daily_dose']
FULL_SINGLE['concurrent_methadone_MME'] = np.where(FULL_SINGLE['drug'] == 'Methadone', FULL_SINGLE['daily_dose'], 0)
FULL_SINGLE['num_prescribers'] = 1
FULL_SINGLE['num_pharmacies'] = 1
FULL_SINGLE['overlap'] = 0
FULL_SINGLE['consecutive_days'] = FULL_SINGLE['days_supply']


# === Compute concurrent benzos
def compute_concurrent_benzo(args):
    pat_id, presc_id = args
    opioid_row = FULL_SINGLE.loc[FULL_SINGLE['prescription_id'] == presc_id].iloc[0]
    presc_date = pd.to_datetime(opioid_row['date_filled'], errors='coerce')
    presc_until = presc_date + pd.to_timedelta(opioid_row['days_supply'], unit='D')
    prescriber_id = opioid_row['prescriber_id']
    
    benzos = BENZO_TABLE[BENZO_TABLE['patient_id'] == pat_id].copy()
    if benzos.empty:
        return (0, 0)

    benzos['date_filled'] = pd.to_datetime(benzos['date_filled'], errors='coerce')
    benzos['presc_until'] = benzos['date_filled'] + pd.to_timedelta(benzos['days_supply'], unit='D')

    before = benzos[(benzos['date_filled'] <= presc_date) & (benzos['presc_until'] > presc_date)]
    after = benzos[(benzos['date_filled'] > presc_date) & (benzos['date_filled'] <= presc_until)]

    num_benzo = len(before) + len(after)
    same_prescriber = len(before[before['prescriber_id'] == prescriber_id]) + len(after[after['prescriber_id'] == prescriber_id])
    return (num_benzo, same_prescriber)

with Pool(processes=cores) as pool:
    results = pool.map(compute_concurrent_benzo, zip(FULL_SINGLE['patient_id'], FULL_SINGLE['prescription_id']))
FULL_SINGLE['concurrent_benzo'], FULL_SINGLE['concurrent_benzo_same'] = zip(*results)
FULL_SINGLE['concurrent_benzo_diff'] = FULL_SINGLE['concurrent_benzo'] - FULL_SINGLE['concurrent_benzo_same']

print(FULL_SINGLE[['prescription_id', 'date_filled', 'days_supply', 'daily_dose', 'concurrent_MME', 'consecutive_days', 'concurrent_benzo']].head())


# === Alerts
def patient_alert(args):
    presc_id = args
    row = FULL_SINGLE.loc[FULL_SINGLE['prescription_id'] == presc_id].iloc[0]
    return [
        int(row['concurrent_MME'] >= 90),
        int(row['concurrent_methadone_MME'] >= 40),
        int(row['num_prescribers'] >= 6),
        int(row['num_pharmacies'] >= 6),
        int(row['consecutive_days'] >= 90),
        int(row['concurrent_benzo'] > 0)
    ]

with Pool(processes=cores) as pool:
    alert_rows = pool.map(patient_alert, FULL_SINGLE['prescription_id'])
alert_df = pd.DataFrame(alert_rows, columns=[f'alert{i+1}' for i in range(6)])
FULL_SINGLE = pd.concat([FULL_SINGLE.reset_index(drop=True), alert_df], axis=1)
FULL_SINGLE['num_alert'] = FULL_SINGLE[[f'alert{i+1}' for i in range(6)]].sum(axis=1)
FULL_SINGLE['any_alert'] = (FULL_SINGLE['num_alert'] > 0).astype(int)

print(FULL_SINGLE[['prescription_id', 'date_filled', 'days_supply', 'consecutive_days', 'num_alert']].head())


# === Long-term use logic
FULL_SINGLE['date_filled'] = pd.to_datetime(FULL_SINGLE['date_filled'], errors='coerce')
FULL_SINGLE['presc_until'] = FULL_SINGLE['date_filled'] + pd.to_timedelta(FULL_SINGLE['days_supply'], unit='D')
FULL_SINGLE['overlap_lt'] = 0
FULL_SINGLE['opioid_days'] = FULL_SINGLE['days_supply']
FULL_SINGLE['long_term'] = (FULL_SINGLE['opioid_days'] >= 90).astype(int)


patient_summary = FULL_SINGLE.groupby('patient_id').apply(lambda df: pd.Series({
    'first_presc_date': df['date_filled'].iloc[0],
    'longterm_filled_date': df.loc[df['long_term'] == 1, 'date_filled'].min() if df['long_term'].sum() > 0 else pd.NaT,
    'longterm_presc_date': df.loc[df['long_term'] == 1, 'presc_until'].min() if df['long_term'].sum() > 0 else pd.NaT,
    'first_longterm_presc': df.loc[df['long_term'] == 1].index.min() if df['long_term'].sum() > 0 else np.nan,
    'first_longterm_presc_id': df.loc[df['long_term'] == 1, 'prescription_id'].min() if df['long_term'].sum() > 0 else np.nan,
})).reset_index()

print(patient_summary.head())

FULL_SINGLE = FULL_SINGLE.merge(patient_summary, on='patient_id', how='left')

### Use presc_until of the long-term prescription to compute
# NA: patient never become long term
# >0: patient is going to become long term
# =0: patient is long term right after this prescription
# <0: patient is already long term

print(FULL_SINGLE[['prescription_id', 'date_filled', 'days_supply', 'long_term', 'longterm_presc_date', 'days_to_long_term']].head())

FULL_SINGLE['days_to_long_term'] = (FULL_SINGLE['longterm_presc_date'] - FULL_SINGLE['date_filled']).dt.days
FULL_SINGLE = FULL_SINGLE[(FULL_SINGLE['days_to_long_term'].isna()) | (FULL_SINGLE['days_to_long_term'] > 0)]
FULL_SINGLE['long_term_180'] = ((FULL_SINGLE['days_to_long_term'] <= 180) & (~FULL_SINGLE['days_to_long_term'].isna())).astype(int)

print(f'Keep up to first long term use. Number of prescriptions left: {len(FULL_SINGLE)}')

print(FULL_SINGLE[['prescription_id', 'date_filled', 'days_supply', 'long_term', 'longterm_presc_date', 'days_to_long_term']].head())

drop_cols = ['first_presc_date', 'longterm_filled_date', 'longterm_presc_date', 
             'first_longterm_presc', 'first_longterm_presc_id']
FULL_SINGLE.drop(columns=drop_cols, inplace=True)


FULL_SINGLE.to_csv(f"{datadir}FULL_OPIOID_{year}_ONE_TEMP.csv", index=False)

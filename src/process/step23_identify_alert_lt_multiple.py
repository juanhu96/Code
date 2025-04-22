'''
STEP 2
Identify CURES alert (patient w/ 2+ prescriptions)

STEP 3
Identify longterm & longterm180 (patients w/ 2+ prescriptions)
Keep prescriptions up to first long term

INPUT: FULL_OPIOID_2018_ATLEASTTWO_1.csv
OUTPUT: FULL_OPIOID_2018_ATLEASTTWO_1_TEMP.csv
'''

import pandas as pd
import numpy as np
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import sys


year = int(sys.argv[1])
case = sys.argv[2]
datadir = "/export/storage_cures/CURES/Processed/"
cores = 8

FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_{case}.csv")
# FULL = FULL.head(5000)
# print("Testing with 5k rows\n")

BENZO = pd.read_csv(f"{datadir}FULL_BENZO_{year}.csv")

# Preprocess
FULL['prescription_id'] = np.arange(1, len(FULL) + 1)
FULL['date_filled'] = pd.to_datetime(FULL['date_filled'], errors='coerce')
FULL['presc_until'] = FULL['date_filled'] + pd.to_timedelta(FULL['days_supply'], unit='D')
FULL['age'] = FULL['date_filled'].dt.year - FULL['patient_birth_year']

BENZO['date_filled'] = pd.to_datetime(BENZO['date_filled'], errors='coerce')
BENZO['presc_until'] = BENZO['date_filled'] + pd.to_timedelta(BENZO['days_supply'], unit='D')


# 1. Concurrent MME & Methadone
def compute_mme(args):
    pid, pres_id = args
    patient = FULL[FULL['patient_id'] == pid]
    presc = patient[patient['prescription_id'] == pres_id].iloc[0]
    active = patient[(patient['date_filled'] <= presc['date_filled']) & (patient['presc_until'] > presc['date_filled'])]
    meth = active[active['drug'] == 'Methadone']
    return active['daily_dose'].sum(), meth['daily_dose'].sum()

with Pool(cores) as pool:
    mme_vals = pool.map(compute_mme, zip(FULL['patient_id'], FULL['prescription_id']))
FULL['concurrent_MME'], FULL['concurrent_methadone_MME'] = zip(*mme_vals)


# 2. Prescribers / Pharmacies last 180 days
def compute_presc_pharm(args):
    pid, pres_id = args
    patient = FULL[FULL['patient_id'] == pid]
    presc = patient[patient['prescription_id'] == pres_id].iloc[0]
    window = patient[(patient['date_filled'] >= presc['date_filled'] - timedelta(days=180)) &
                     (patient['date_filled'] <= presc['date_filled'])]
    return window['prescriber_id'].nunique(), window['pharmacy_id'].nunique()

with Pool(cores) as pool:
    presc_vals = pool.map(compute_presc_pharm, zip(FULL['patient_id'], FULL['prescription_id']))
FULL['num_prescribers_past180'], FULL['num_pharmacies_past180'] = zip(*presc_vals)


# 3. Overlap & Consecutive Days
FULL.sort_values(['patient_id', 'date_filled'], inplace=True)

def compute_overlap(args):
    pid, pres_id = args

    patient = FULL[FULL['patient_id'] == pid]
    presc = FULL[FULL['prescription_id'] == pres_id]
    presc_date = pd.to_datetime(presc.iloc[0]['date_filled'])
    presc_index = patient[patient['prescription_id'] == pres_id].index

    if presc_index.min() == patient.index.min(): return 0

    prev = patient[patient['date_filled'] < presc_date]
    if prev.empty: return 0

    prev_presc_until = prev['presc_until'].iloc[-1]

    if prev_presc_until + pd.Timedelta(days=1) >= presc_date:
        return (prev_presc_until + pd.Timedelta(days=2) - presc_date).days
    else:
        return 0

def compute_consec_days(args):
    pid, pres_id = args

    patient = FULL[FULL['patient_id'] == pid]
    curr = patient[patient['prescription_id'] == pres_id]
    curr = curr.iloc[0]

    if curr['overlap'] == 0:
        return curr['days_supply']

    prev = patient[patient['date_filled'] < curr['date_filled']]
    prev_zero_overlap = prev[prev['overlap'] == 0]
    last_idx = prev_zero_overlap.index.max()

    return (curr['presc_until'] - patient.loc[last_idx, 'date_filled']).days

with Pool(cores) as pool:
    FULL['overlap'] = pool.map(compute_overlap, zip(FULL['patient_id'], FULL['prescription_id']))

with Pool(cores) as pool:
    FULL['consecutive_days'] = pool.map(compute_consec_days, zip(FULL['patient_id'], FULL['prescription_id']))


# 4. Concurrent Benzos
def compute_benzos(args):
    pid, pres_id = args
    benzos = BENZO[BENZO['patient_id'] == pid]
    
    if benzos.empty: return 0, 0

    presc = FULL[FULL['prescription_id'] == pres_id].iloc[0]
    before = benzos[(benzos['date_filled'] <= presc['date_filled']) & (benzos['presc_until'] > presc['date_filled'])]
    after = benzos[(benzos['date_filled'] > presc['date_filled']) & (benzos['date_filled'] <= presc['presc_until'])]
    same_doc = len(before[before['prescriber_id'] == presc['prescriber_id']]) + len(after[after['prescriber_id'] == presc['prescriber_id']])

    return len(before) + len(after), same_doc

with Pool(cores) as pool:
    benzo_vals = pool.map(compute_benzos, zip(FULL['patient_id'], FULL['prescription_id']))

FULL['concurrent_benzo'], FULL['concurrent_benzo_same'] = zip(*benzo_vals)
FULL['concurrent_benzo_diff'] = FULL['concurrent_benzo'] - FULL['concurrent_benzo_same']


# 5. Alerts
def alert_flags(row):
    return [
        int(row['concurrent_MME'] >= 90),
        int(row['concurrent_methadone_MME'] >= 40),
        int(row['num_prescribers_past180'] >= 6),
        int(row['num_pharmacies_past180'] >= 6),
        int(row['consecutive_days'] >= 90),
        int(row['concurrent_benzo'] > 0)
    ]

alerts = FULL.apply(alert_flags, axis=1, result_type='expand')
alerts.columns = [f'alert{i+1}' for i in range(6)]
FULL = pd.concat([FULL, alerts], axis=1)
FULL['num_alert'] = alerts.sum(axis=1)
FULL['any_alert'] = (FULL['num_alert'] > 0).astype(int)


# 6. Long-Term Use Logic
FULL['period_start'] = FULL['presc_until'] - timedelta(days=180)

def compute_overlap_lt(args):
    pid, pres_id = args
    patient = FULL[FULL['patient_id'] == pid]
    idx = patient[patient['prescription_id'] == pres_id].index[0]
    
    if idx == patient.index.min(): return 0
    
    prev_until = patient.loc[idx - 1, 'presc_until']
    return max((prev_until - patient.loc[idx, 'date_filled']).days, 0) if prev_until >= patient.loc[idx, 'date_filled'] else 0

with Pool(cores) as pool:
    FULL['overlap_lt'] = pool.map(compute_overlap_lt, zip(FULL['patient_id'], FULL['prescription_id']))


def compute_opioid_days(args):
    pid, pres_id = args
    patient = FULL[FULL['patient_id'] == pid]
    presc = patient[patient['prescription_id'] == pres_id].iloc[0]
    prev = patient[(patient['presc_until'] > presc['period_start']) &
                   (patient['date_filled'] < presc['date_filled']) &
                   (patient['prescription_id'] < pres_id)]
    
    if prev.empty: return presc['days_supply']

    total_days = prev.iloc[1:]['days_supply'].sum()
    total_overlap = prev.iloc[1:]['overlap_lt'].sum()
    first = prev.iloc[0]
    first_accum = (first['presc_until'] - presc['period_start']).days if first['date_filled'] <= presc['period_start'] else first['days_supply']
    opioid_days = first_accum + total_days + presc['days_supply'] - total_overlap - presc['overlap_lt']

    return min(opioid_days, 180)

with Pool(cores) as pool:
    FULL['opioid_days'] = pool.map(compute_opioid_days, zip(FULL['patient_id'], FULL['prescription_id']))

FULL['long_term'] = (FULL['opioid_days'] >= 90).astype(int)


# First LT per patient
patient_LT = FULL[FULL['long_term'] == 1].groupby('patient_id').agg({
    'presc_until': 'min'
}).rename(columns={'presc_until': 'longterm_presc_date'}).reset_index()

FULL = FULL.merge(patient_LT, on='patient_id', how='left')
FULL['days_to_long_term'] = (FULL['longterm_presc_date'] - FULL['date_filled']).dt.days
FULL = FULL[(FULL['days_to_long_term'].isna()) | (FULL['days_to_long_term'] > 0)]
FULL['long_term_180'] = ((FULL['days_to_long_term'] <= 180) & (~FULL['days_to_long_term'].isna())).astype(int)

print(f'Keep up to first long term use. Number of prescriptions left: {len(FULL)}')

# drop_cols = ['first_presc_date', 'longterm_filled_date', 'longterm_presc_date', 
#              'first_longterm_presc', 'first_longterm_presc_id']
drop_cols = ['longterm_presc_date']
FULL.drop(columns=drop_cols, inplace=True)


# sanity check
FULL_LT = FULL[FULL['long_term_180'] == 1]
sample_ids = FULL_LT['patient_id'].drop_duplicates().sample(3, random_state=0)
FULL_sample = FULL[FULL['patient_id'].isin(sample_ids)]
pd.set_option('display.max_columns', None)
print(FULL_sample[['patient_id', 'date_filled', 'days_supply', 'consecutive_days', 'days_to_long_term', 'long_term_180']].head(20))


FULL.to_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_{case}_TEMP.csv", index=False)

'''
STEP 6
Combine data and encode/convert to input form for riskSLIM
Note that input is only for creating stumps

INPUT: FULL_OPIOID_2018_FEATURE.csv
OUTPUT: FULL_OPIOID_2018_INPUT.csv
'''

from multiprocessing import Pool
import pandas as pd
import numpy as np
import sys

year = int(sys.argv[1])
datadir = "/export/storage_cures/CURES/Processed/"
cores = 8

FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_FEATURE.csv")
print(f"Imported FULL_OPIOID_{year}_FEATURE.csv, with shape {FULL.shape}, columns {FULL.columns.tolist()}")


# ======================================
# Double check
# ======================================

RAW = pd.read_csv(f"{datadir}../RX_{year}.csv")
RAW.rename(columns={'Unnamed: 0': 'X'}, inplace=True)
print(f"Imported RX_{year}.csv, columns {RAW.columns.tolist()}")

UNDERAGE = FULL[FULL['patient_birth_year'] > (year - 18)] # empty set
print(UNDERAGE[['patient_id', 'patient_birth_year', 'date_filled']].head(10))

previous_year = year - 1
FULL_PREVIOUS = pd.read_csv(f"{datadir}../RX_{previous_year}.csv")

CHRONIC = FULL_PREVIOUS[FULL_PREVIOUS['class'] == 'Opioid'].copy()
CHRONIC['prescription_month'] = pd.to_datetime(CHRONIC['date_filled'], format="%m/%d/%Y").dt.month
CHRONIC = CHRONIC[CHRONIC['prescription_month'] > 10]
CHRONIC_vector = CHRONIC['patient_id'].unique()
FULL = FULL[~FULL['patient_id'].isin(CHRONIC_vector)]
print(f"Chronic prescriptions removed. Number of prescriptions left: {FULL.shape[0]}")


# ======================================
# To input
# ======================================

FULL['patient_gender'] = (FULL['patient_gender'] != 'M').astype(int)

for drug in ['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone']:
    FULL[drug] = (FULL[f"{drug}_MME"] > 0).astype(int)

payment_types = ["Medicaid", "CommercialIns", "Medicare", "CashCredit", "MilitaryIns", "WorkersComp", "Other", "IndianNation"]
for payment in payment_types:
    FULL[payment] = (FULL['payment'] == payment).astype(int)

# print(f"Columns after encoding: {FULL.columns.tolist()}")
# raise SystemExit("Stopping here")

DROP_COLS = [
    'X', 'patient_birth_year', 'prescriber_id', 'prescriber_zip', 'pharmacy_id', 'pharmacy_zip', 'strength',
    'MAINDOSE', 'period_start', 'drx_refill_number', 'drx_refill_authorized_number', 
    'quantity_per_day', 'conversion', 'class', 'drug', 'payment', 'num_prescriptions',
    'presc_until', 'prescription_id', 'overlap', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6',
    'num_alert', 'any_alert', 'overlap_lt', 'opioid_days', 'city_name'
]
FULL_INPUT = FULL.drop(columns=DROP_COLS, errors='ignore')

# Enrich ZIP-level demographics
FULL_INPUT['zip_pop'] = FULL_INPUT['zip_pop'].replace({',': ''}, regex=True).astype(float)
FULL_INPUT['zip_pop_density'] = (FULL_INPUT['zip_pop_density'].replace({',': ''}, regex=True).astype(float))
FULL_INPUT['patient_zip_yr_num_prescriptions_per_pop'] = FULL_INPUT['patient_zip_yr_num_prescriptions'] / FULL_INPUT['zip_pop']
FULL_INPUT['patient_zip_yr_num_patients_per_pop'] = FULL_INPUT['patient_zip_yr_num_patients'] / FULL_INPUT['zip_pop']

percentiles = [50, 75]
cols = [
    'zip_pop_density',
    'median_household_income',
    'family_poverty_pct',
    'unemployment_pct',
    'patient_zip_yr_num_prescriptions_per_pop',
    'patient_zip_yr_num_patients_per_pop'
]
for col in cols:
    for p in percentiles:
        cutoff = np.percentile(FULL_INPUT[col].dropna(), p)
        FULL_INPUT[f'{col}_above{p}'] = (FULL_INPUT[col] >= cutoff).astype(int)
# FULL_INPUT['zip_pop_density_quartile'] = pd.qcut(FULL_INPUT['zip_pop_density'], 4, labels=False, duplicates='drop') + 1
# FULL_INPUT['median_household_income_quartile'] = pd.qcut(FULL_INPUT['median_household_income'], 4, labels=False, duplicates='drop') + 1
# FULL_INPUT['family_poverty_pct_quartile'] = pd.qcut(FULL_INPUT['family_poverty_pct'], 4, labels=False, duplicates='drop') + 1
# FULL_INPUT['unemployment_pct_quartile'] = pd.qcut(FULL_INPUT['unemployment_pct'], 4, labels=False, duplicates='drop') + 1
# FULL_INPUT['patient_zip_yr_num_prescriptions_per_pop_quartile'] = pd.qcut(FULL_INPUT['patient_zip_yr_num_prescriptions_per_pop'], 4, labels=False, duplicates='drop') + 1
# FULL_INPUT['patient_zip_yr_num_patients_per_pop_quartile'] = pd.qcut(FULL_INPUT['patient_zip_yr_num_patients_per_pop'], 4, labels=False, duplicates='drop') + 1

# Export final result
FULL_INPUT.to_csv(f"{datadir}FULL_OPIOID_{year}_INPUT.csv", index=False)


# ======================================
# To input (first prescription only)
# ======================================

FULL_INPUT_FIRST = FULL_INPUT.groupby('patient_id', as_index=False).first()
print(f"Number of patients before: {FULL_INPUT['patient_id'].nunique()}, number of patients now: {FULL_INPUT_FIRST['patient_id'].nunique()}")
print(f"Number of prescriptions before: {FULL_INPUT.shape[0]}, number of prescriptions now: {FULL_INPUT_FIRST.shape[0]}")

FULL_INPUT_FIRST.to_csv(f"{datadir}FULL_OPIOID_{year}_FIRST_INPUT.csv", index=False)
print("Done")
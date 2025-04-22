'''
STEP 1
Filter chronic/outlier/illicit prescriptions, export Opioid and Benzo prescriptions
Split the data based on number of prescriptions

INPUT: RX_2018.csv
OUTPUT: FULL_OPIOID_2018_ATLEASTONE.csv, FULL_BENZO_2018.csv
'''

import pandas as pd
import numpy as np
from datetime import timedelta
import sys

basedir = "/export/storage_cures/CURES/"
export_dir = basedir + "Processed/"
year = int(sys.argv[1])
previous_year = year - 1

FULL_CURRENT = pd.read_csv(f"{basedir}RX_{year}.csv")
FULL_PREVIOUS = pd.read_csv(f"{basedir}RX_{previous_year}.csv")
print(f"Number of prescriptions in {year}: {len(FULL_CURRENT)}, Number of prescriptions in {previous_year}: {len(FULL_PREVIOUS)}")
print(FULL_CURRENT.columns.to_list())

FULL_CURRENT.rename(columns={'Unnamed: 0': 'X'}, inplace=True)

# patient_779_rows = FULL_CURRENT[FULL_CURRENT['patient_id'] == 779]
# print(f"Rows for patient_id 779:\n{patient_779_rows}")

# ========== Drop Chronic Patients ==========
# - those who filled an opioid prescription in the last 60 days of the prior year

FULL_PREVIOUS['date_filled'] = pd.to_datetime(FULL_PREVIOUS['date_filled'], format="%m/%d/%Y", errors='coerce')
chronic_ids = FULL_PREVIOUS[
    (FULL_PREVIOUS['class'] == 'Opioid') & 
    (FULL_PREVIOUS['date_filled'].dt.month > 10)
]['patient_id'].dropna().unique()

FULL_OPIOID = FULL_CURRENT[FULL_CURRENT['class'] == 'Opioid'].copy()
FULL_OPIOID = FULL_OPIOID[~FULL_OPIOID['patient_id'].isin(chronic_ids)]

print(f"Number of chronic patients removed: {len(chronic_ids)}, Number of prescriptions left: {len(FULL_OPIOID)}")


# ========== Remove Outliers ==========
# - prescriptions that exceed 1,000 daily MME
# - patients with more than 100 prior prescriptions)

agg = FULL_OPIOID.groupby('patient_id').agg(
    num_presc=('patient_id', 'count'),
    max_dose=('daily_dose', 'max')
).reset_index()
agg['outlier'] = ((agg['num_presc'] >= 100) | (agg['max_dose'] >= 1000)).astype(int)

FULL_OPIOID = FULL_OPIOID.merge(agg[['patient_id', 'outlier']], on='patient_id', how='left')
FULL_OPIOID = FULL_OPIOID[FULL_OPIOID['outlier'] == 0].drop(columns='outlier')

print(f"Outliers removed. Number of prescriptions left: {len(FULL_OPIOID)}")


# ========== Remove Illicit Users ==========
# - filled prescriptions from three or more providers on their first date

FULL_OPIOID['date_filled'] = pd.to_datetime(FULL_OPIOID['date_filled'], format="%m/%d/%Y", errors='coerce')
FULL_OPIOID.sort_values(['patient_id', 'date_filled'], inplace=True)

first_presc = FULL_OPIOID.groupby('patient_id')['date_filled'].min().reset_index(name='first_presc_date')
FIRST_PRESC = FULL_OPIOID.merge(first_presc, on='patient_id')
prescriber_counts = FIRST_PRESC[FIRST_PRESC['date_filled'] == FIRST_PRESC['first_presc_date']].groupby('patient_id')['prescriber_id'].nunique()
illicit_ids = prescriber_counts[prescriber_counts >= 3].index
FULL_OPIOID = FULL_OPIOID[~FULL_OPIOID['patient_id'].isin(illicit_ids)]

print(f"Illicit users removed. Number of prescriptions left: {len(FULL_OPIOID)}")


# ========== Remove Underage Patients ==========
# - age < 18

FULL_OPIOID = FULL_OPIOID[FULL_OPIOID['patient_birth_year'] <= (year - 18)]

print(f"Underage patients removed. Number of prescriptions left: {len(FULL_OPIOID)}")


# ========== Remove Duplicate Rows ==========
# - based on all columns except 'X'
dedup_cols = [col for col in FULL_OPIOID.columns if col != 'X']
FULL_OPIOID = FULL_OPIOID.drop_duplicates(subset=dedup_cols)

print(f"Duplicate rows removed. Number of prescriptions left: {len(FULL_OPIOID)}")
# print("779 found") if 779 in FULL_OPIOID['patient_id'].unique() else print("779 not found")


# ========== Split Data ==========
# - split into two groups based on the number of prescriptions

patient_counts = FULL_OPIOID.groupby('patient_id').size().reset_index(name='num_prescriptions')
FULL_NUM_PRESC = FULL_OPIOID.merge(patient_counts, on='patient_id', how='left')

FULL_NUM_PRESC_ONE = FULL_NUM_PRESC[FULL_NUM_PRESC['num_prescriptions'] == 1]
FULL_NUM_PRESC_ATLEASTTWO = FULL_NUM_PRESC[FULL_NUM_PRESC['num_prescriptions'] > 1]

# print("779 found in PRESC_ONE") if 779 in FULL_NUM_PRESC_ONE['patient_id'].unique() else print("779 not found in PRESC_ONE")
# print("779 found in PRESC_ATLEASTTWO") if 779 in FULL_NUM_PRESC_ATLEASTTWO['patient_id'].unique() else print("779 not found in PRESC_ATLEASTTWO")
# raise SystemExit("Stopping here")

# Split into 4 parts (year-specific thresholds)
if year == 2018:
    part1 = FULL_NUM_PRESC_ATLEASTTWO[FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 41846152]
    part2 = FULL_NUM_PRESC_ATLEASTTWO[(FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 41846152) & (FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 54195156)]
    part3 = FULL_NUM_PRESC_ATLEASTTWO[(FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 54195156) & (FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 68042918)]
    part4 = FULL_NUM_PRESC_ATLEASTTWO[FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 68042918]

elif year == 2019:
    part1 = FULL_NUM_PRESC_ATLEASTTWO[FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 43247790]
    part2 = FULL_NUM_PRESC_ATLEASTTWO[(FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 43247790) & (FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 59871298)]
    part3 = FULL_NUM_PRESC_ATLEASTTWO[(FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 59871298) & (FULL_NUM_PRESC_ATLEASTTWO['patient_id'] < 71429081)]
    part4 = FULL_NUM_PRESC_ATLEASTTWO[FULL_NUM_PRESC_ATLEASTTWO['patient_id'] >= 71429081]

else:
    raise ValueError("Year not supported. Please use 2018 or 2019.")

print(f"ONE: {len(FULL_NUM_PRESC_ONE)}, part1: {len(part1)}, part2: {len(part2)}, part3: {len(part3)}, part4: {len(part4)}")


# Write to CSV
FULL_NUM_PRESC_ONE.to_csv(f"{export_dir}FULL_OPIOID_{year}_ONE.csv", index=False)
part1.to_csv(f"{export_dir}FULL_OPIOID_{year}_ATLEASTTWO_1.csv", index=False)
part2.to_csv(f"{export_dir}FULL_OPIOID_{year}_ATLEASTTWO_2.csv", index=False)
part3.to_csv(f"{export_dir}FULL_OPIOID_{year}_ATLEASTTWO_3.csv", index=False)
part4.to_csv(f"{export_dir}FULL_OPIOID_{year}_ATLEASTTWO_4.csv", index=False)


# ========== Benzodiazepine Section ==========

FULL_BENZO = FULL_CURRENT[FULL_CURRENT['class'] == 'Benzodiazepine'].copy()
FULL_BENZO['date_filled'] = pd.to_datetime(FULL_BENZO['date_filled'], format="%m/%d/%Y", errors='coerce')
FULL_BENZO.sort_values(['patient_id', 'date_filled'], inplace=True)
FULL_BENZO['prescription_month'] = FULL_BENZO['date_filled'].dt.month
FULL_BENZO['prescription_year'] = FULL_BENZO['date_filled'].dt.year
FULL_BENZO['presc_until'] = FULL_BENZO['date_filled'] + pd.to_timedelta(FULL_BENZO['days_supply'], unit='D')

print(f"Number of prescriptions in Benzodiazepine: {len(FULL_BENZO)}")

# Remove duplicates
dedup_cols_benzo = [col for col in FULL_BENZO.columns if col != 'X']
FULL_BENZO = FULL_BENZO.drop_duplicates(subset=dedup_cols_benzo)

print(f"Duplicate rows removed from Benzodiazepine. Number of prescriptions left: {len(FULL_BENZO)}")

FULL_BENZO.to_csv(f"{export_dir}FULL_BENZO_{year}.csv", index=False)

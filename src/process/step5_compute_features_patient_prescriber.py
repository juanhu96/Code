'''
STEP 5
Feature engineering (patient-based demographics / prescriber info / prescriber info / NDC code)

INPUT: FULL_OPIOID_2018_ONE_FEATURE.csv, FULL_OPIOID_2018_ATLEASTTWO_FEATURE.csv
OUTPUT: FULL_OPIOID_2018_FEATURE.csv
'''


from multiprocessing import Pool
import pandas as pd
import numpy as np
import sys

year = int(sys.argv[1])
datadir = "/export/storage_cures/CURES/Processed/"
cores = 8

FULL_ONE = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ONE_FEATURE.csv")
FULL_ATLEASTTWO_1 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_1_FEATURE.csv")
FULL_ATLEASTTWO_2 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_2_FEATURE.csv")
FULL_ATLEASTTWO_3 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_3_FEATURE.csv")
FULL_ATLEASTTWO_4 = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_4_FEATURE.csv")
print(f'ONE: {FULL_ONE.shape[0]}, 1: {FULL_ATLEASTTWO_1.shape[0]}, 2: {FULL_ATLEASTTWO_2.shape[0]}, 3: {FULL_ATLEASTTWO_3.shape[0]}, 4: {FULL_ATLEASTTWO_4.shape[0]}')

# Ensure zip codes are strings
for df in [FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4]:
    df['patient_zip'] = df['patient_zip'].astype(str)

FULL = pd.concat([FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4], ignore_index=True)
print(f"Combined prescriptions: {FULL.shape[0]}")

del FULL_ONE, FULL_ATLEASTTWO_1, FULL_ATLEASTTWO_2, FULL_ATLEASTTWO_3, FULL_ATLEASTTWO_4


# ======================================
# Patient-based
# ======================================
HPI = pd.read_csv(f"{datadir}../CA/HPI.csv")
HPI['Zip'] = HPI['Zip'].astype(str)
HPI_patient = HPI.rename(columns={'Zip': 'patient_zip', 'HPIQuartile': 'patient_HPIQuartile'}).drop(columns=['HPI'])
HPI_prescriber = HPI.rename(columns={'Zip': 'prescriber_zip', 'HPIQuartile': 'prescriber_HPIQuartile'}).drop(columns=['HPI'])
HPI_pharmacy = HPI.rename(columns={'Zip': 'pharmacy_zip', 'HPIQuartile': 'pharmacy_HPIQuartile'}).drop(columns=['HPI'])


# Load zip demographics
ZIP_DEMO = pd.read_csv(f"{datadir}../CA/California_DemographicsByZip2020.csv")
# ZIP_DEMO = ZIP_DEMO.rename(columns={'X......name': 'Zip'}) # in R
ZIP_DEMO = ZIP_DEMO.rename(columns={ZIP_DEMO.columns[0]: 'Zip'})
ZIP_DEMO['patient_zip'] = ZIP_DEMO['Zip'].astype(str)
ZIP_DEMO['zip_pop'] = ZIP_DEMO['population'].str.replace(",", "").astype(float)
ZIP_DEMO['zip_pop_density'] = ZIP_DEMO['population_density_sq_mi']

# median_household_income: set zip with median_household_income = $1 to 0.
ZIP_DEMO['median_household_income'] = ZIP_DEMO['median_household_income'].str.replace(r"[\$,]", "", regex=True)
ZIP_DEMO['median_household_income'] = ZIP_DEMO['median_household_income'].replace("($1)", "0").str.replace("(", "-").str.replace(")", "")
ZIP_DEMO['median_household_income'] = ZIP_DEMO['median_household_income'].astype(float)

ZIP_DEMO['family_poverty_pct'] = ZIP_DEMO['family_poverty_pct'].str.replace("%", "").astype(float)
ZIP_DEMO['unemployment_pct'] = ZIP_DEMO['unemployment_pct'].str.replace("%", "").astype(float)
ZIP_DEMO = ZIP_DEMO[['patient_zip', 'city_name', 'zip_pop', 'zip_pop_density', 'median_household_income', 'family_poverty_pct', 'unemployment_pct']]


# Merge HPI and ZIP_DEMO
FULL[['patient_zip', 'prescriber_zip', 'pharmacy_zip']] = FULL[['patient_zip', 'prescriber_zip', 'pharmacy_zip']].astype(str)
FULL = FULL.merge(HPI_patient, how='left', on='patient_zip')
FULL = FULL.merge(HPI_prescriber, how='left', on='prescriber_zip')
FULL = FULL.merge(HPI_pharmacy, how='left', on='pharmacy_zip')
FULL['patient_HPIQuartile'] = FULL['patient_HPIQuartile'].fillna(FULL['prescriber_HPIQuartile']).fillna(FULL['pharmacy_HPIQuartile'])
FULL['prescriber_HPIQuartile'] = FULL['prescriber_HPIQuartile'].fillna(FULL['patient_HPIQuartile']).fillna(FULL['pharmacy_HPIQuartile'])
FULL['pharmacy_HPIQuartile'] = FULL['pharmacy_HPIQuartile'].fillna(FULL['patient_HPIQuartile']).fillna(FULL['prescriber_HPIQuartile'])
FULL = FULL.merge(ZIP_DEMO, how='left', on='patient_zip')

# FULL = FULL[~FULL['patient_zip'].isna()]
# print(f"Patient zip not in CA dropped, {FULL.shape[0]} prescriptions left")

# Load previous year data
previous_year = year - 1
FULL_PREVIOUS = pd.read_csv(f"{datadir}../RX_{previous_year}.csv")
FULL_PREVIOUS_OPIOID = FULL_PREVIOUS[FULL_PREVIOUS['class'] == 'Opioid']
del FULL_PREVIOUS

# Patient-based zip features
count_rows = FULL_PREVIOUS_OPIOID.groupby('patient_zip').size().reset_index(name='patient_zip_yr_num_prescriptions')
agg_features = FULL_PREVIOUS_OPIOID.groupby('patient_zip').agg(
    patient_zip_yr_num_patients=('patient_id', pd.Series.nunique),
    patient_zip_yr_num_pharmacies=('pharmacy_id', pd.Series.nunique),
    patient_zip_yr_avg_MME=('daily_dose', 'mean'),
    patient_zip_yr_avg_days=('days_supply', 'mean'),
    patient_zip_yr_avg_quantity=('quantity', 'mean')
).reset_index()
PATIENT_ZIP = pd.merge(count_rows, agg_features, on='patient_zip')

for col in [
    'patient_zip_yr_num_prescriptions',
    'patient_zip_yr_num_patients',
    'patient_zip_yr_num_pharmacies',
    'patient_zip_yr_avg_MME',
    'patient_zip_yr_avg_days',
    'patient_zip_yr_avg_quantity']:

    percentiles = [50, 75]
    cutoffs = np.percentile(PATIENT_ZIP[col].dropna(), percentiles)
    for p, cutoff in zip(percentiles, cutoffs):
        PATIENT_ZIP[f'{col}_above{p}'] = (PATIENT_ZIP[col] >= cutoff).astype(int)
    # PATIENT_ZIP[f'{col}_quartile'] = pd.qcut(PATIENT_ZIP[col], 4, labels=False, duplicates='drop') + 1

FULL = FULL.merge(PATIENT_ZIP, how='left', on='patient_zip')


# ======================================
# Prescriber-based features
# ======================================
PRESCRIBER = FULL_PREVIOUS_OPIOID.groupby('prescriber_id').agg(
    prescriber_yr_num_prescriptions=('patient_id', 'count'),
    prescriber_yr_num_patients=('patient_id', pd.Series.nunique),
    prescriber_yr_num_pharmacies=('pharmacy_id', pd.Series.nunique),
    prescriber_yr_avg_MME=('daily_dose', 'mean'),
    prescriber_yr_avg_days=('days_supply', 'mean'),
    prescriber_yr_avg_quantity=('quantity', 'mean')
).reset_index()

for col in [
    'prescriber_yr_num_prescriptions',
    'prescriber_yr_num_patients',
    'prescriber_yr_num_pharmacies',
    'prescriber_yr_avg_MME',
    'prescriber_yr_avg_days',
    'prescriber_yr_avg_quantity']:

    percentiles = [50, 75]
    cutoffs = np.percentile(PRESCRIBER[col].dropna(), percentiles)
    for p, cutoff in zip(percentiles, cutoffs):
        PRESCRIBER[f'{col}_above{p}'] = (PRESCRIBER[col] >= cutoff).astype(int)
    # PRESCRIBER[f'{col}_quartile'] = pd.qcut(PRESCRIBER[col], 4, labels=False, duplicates='drop') + 1

FULL = FULL.merge(PRESCRIBER, how='left', on='prescriber_id')


# ======================================
# Pharmacy-based features
# ======================================
PHARMACY = FULL_PREVIOUS_OPIOID.groupby('pharmacy_id').agg(
    pharmacy_yr_num_prescriptions=('patient_id', 'count'),
    pharmacy_yr_num_patients=('patient_id', pd.Series.nunique),
    pharmacy_yr_num_prescribers=('prescriber_id', pd.Series.nunique),
    pharmacy_yr_avg_MME=('daily_dose', 'mean'),
    pharmacy_yr_avg_days=('days_supply', 'mean'),
    pharmacy_yr_avg_quantity=('quantity', 'mean')
).reset_index()

for col in [
    'pharmacy_yr_num_prescriptions',
    'pharmacy_yr_num_patients',
    'pharmacy_yr_num_prescribers',
    'pharmacy_yr_avg_MME',
    'pharmacy_yr_avg_days',
    'pharmacy_yr_avg_quantity']:

    percentiles = [50, 75]
    cutoffs = np.percentile(PHARMACY[col].dropna(), percentiles)
    for p, cutoff in zip(percentiles, cutoffs):
        PHARMACY[f'{col}_above{p}'] = (PHARMACY[col] >= cutoff).astype(int)
    # PHARMACY[f'{col}_quartile'] = pd.qcut(PHARMACY[col], 4, labels=False, duplicates='drop') + 1

FULL = FULL.merge(PHARMACY, how='left', on='pharmacy_id')


# ======================================
# NDC code, long-acting drugs
# ======================================
NDC = pd.read_csv(f"{datadir}../NDCcodes.csv")
NDC = NDC[['PRODUCTNDC', 'PRODUCTTYPENAME', 'DOSAGEFORMNAME']]
FULL = FULL.merge(NDC, how='left', on='PRODUCTNDC')

long_acting_forms = [
    "CAPSULE, EXTENDED RELEASE",
    "PATCH, EXTENDED RELEASE",
    "SUSPENSION, EXTENDED RELEASE",
    "TABLET, EXTENDED RELEASE",
    "TABLET, FILM COATED, EXTENDED RELEASE"
]
FULL['long_acting'] = FULL['DOSAGEFORMNAME'].isin(long_acting_forms).astype(int)

FULL.to_csv(f"{datadir}FULL_OPIOID_{year}_FEATURE.csv", index=False)
print(f"Combined prescriptions with patient/prescriber/pharmacy features, and NDC features: {FULL.shape}")

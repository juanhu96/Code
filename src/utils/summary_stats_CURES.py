import os
import sys
import pandas as pd
import numpy as np
basedir = "/export/storage_cures/CURES/"
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

# --------------------------------

def drop_na_rows(FULL):

    FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)

    feature_list = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo', 
                    'patient_gender', 'days_supply', 'daily_dose',
                    'num_prior_prescriptions', 'diff_MME', 'diff_days',
                    'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                    'patient_zip_yr_avg_days', 'patient_zip_yr_avg_MME']

    percentile_list = ['patient_zip_yr_num_prescriptions', 'patient_zip_yr_num_patients', 
                        'patient_zip_yr_num_pharmacies', 'patient_zip_yr_avg_MME', 
                        'patient_zip_yr_avg_days', 'patient_zip_yr_avg_quantity', 
                        'patient_zip_yr_num_prescriptions_per_pop', 'patient_zip_yr_num_patients_per_pop',
                        'prescriber_yr_num_prescriptions', 'prescriber_yr_num_patients', 
                        'prescriber_yr_num_pharmacies', 'prescriber_yr_avg_MME', 
                        'prescriber_yr_avg_days', 'prescriber_yr_avg_quantity',
                        'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 
                        'pharmacy_yr_num_prescribers', 'pharmacy_yr_avg_MME', 
                        'pharmacy_yr_avg_days', 'pharmacy_yr_avg_quantity',
                        'zip_pop_density', 'median_household_income', 
                        'family_poverty_pct', 'unemployment_pct']
    percentile_features = [col for col in FULL.columns if any(col.startswith(f"{prefix}_above") for prefix in percentile_list)]
    feature_list_extended = feature_list + percentile_features
    FULL = FULL.dropna(subset=feature_list_extended) # drop NA rows to match the stumps

    return FULL

# --------------------------------

FULL_INPUT_2018 = pd.read_csv(f"{datadir}FULL_OPIOID_2018_INPUT.csv")
FULL_INPUT_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
FULL_INPUT_2018 = drop_na_rows(FULL_INPUT_2018)
FULL_INPUT_2019 = drop_na_rows(FULL_INPUT_2019)
FULL_INPUT = pd.concat([FULL_INPUT_2018, FULL_INPUT_2019], ignore_index=True)

# Base counts
total_lt = (FULL_INPUT['long_term_180'] == 1).sum()
total_nonlt = (FULL_INPUT['long_term_180'] == 0).sum()

def count_by_condition(condition):
    subset = FULL_INPUT[condition]
    total = len(subset)
    lt_count = (subset['long_term_180'] == 1).sum()
    nonlt_count = (subset['long_term_180'] == 0).sum()

    return {
        'lt': f"{lt_count} ({lt_count / total:.1%})" if total > 0 else "0 (0.0%)",
        'nonlt': f"{nonlt_count} ({nonlt_count / total:.1%})" if total > 0 else "0 (0.0%)"
    }

# Define conditions
conditions = {
    'concurrentMME >= 90': FULL_INPUT['concurrent_MME'] >= 90,
    'concurrentMME >= 40 & drug == "Methadone"': FULL_INPUT['concurrent_methadone_MME'] >= 40,
    'num_prescriber >= 6': FULL_INPUT['num_prescribers_past180'] >= 6,
    'num_pharmacies >= 6': FULL_INPUT['num_pharmacies_past180'] >= 6,
    'concurrent_benzo >= 1': FULL_INPUT['concurrent_benzo'] >= 1,
    'not_flagged': (
        (FULL_INPUT['concurrent_MME'] < 90) &
        (FULL_INPUT['concurrent_methadone_MME'] < 40) &
        (FULL_INPUT['num_prescribers_past180'] < 6) &
        (FULL_INPUT['num_pharmacies_past180'] < 6) &
        (FULL_INPUT['consecutive_days'] < 90) &
        (FULL_INPUT['concurrent_benzo'] < 1)
    )
}

# Apply all
print(f"Total long_term_180 == 0: {total_nonlt}")
print(f"Total long_term_180 == 1: {total_lt}")

for label, cond in conditions.items():
    counts = count_by_condition(cond)
    print(f"{label}:")
    print(f"  long_term_180 == 0: {counts['nonlt']}")
    print(f"  long_term_180 == 1: {counts['lt']}")


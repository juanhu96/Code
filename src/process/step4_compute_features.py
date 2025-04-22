'''
STEP 4
Feature engineering (prescription-based)

INPUT: FULL_OPIOID_2018_ONE_TEMP.csv, FULL_OPIOID_2018_ATLEASTTWO_TEMP.csv
OUTPUT: FULL_OPIOID_2018_ONE_FEATURE.csv, FULL_OPIOID_2018_ATLEASTTWO_FEATURE.csv
'''

import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys


year = int(sys.argv[1])
case = sys.argv[2]
datadir = "/export/storage_cures/CURES/Processed/"
cores = 4


# Define benzo count function
def compute_num_prior_benzo(args):
    pat_id, presc_id = args
    row = FULL[FULL['prescription_id'] == presc_id].iloc[0]
    presc_date = row['date_filled']
    pat_benzo = BENZO_TABLE[BENZO_TABLE['patient_id'] == pat_id].copy()
    if pat_benzo.empty:
        return [0, 0, 0]

    def count_in_window(days):
        start = presc_date - pd.Timedelta(days=days)
        end = presc_date
        dates = pd.to_datetime(pat_benzo['date_filled'], errors='coerce')
        return ((dates >= start) & (dates <= end)).sum()

    return [count_in_window(180), count_in_window(90), count_in_window(30)]


if case == 'single':

    FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ONE_TEMP.csv")

    # FULL = FULL.head(500)
    # print("Testing with first 500 rows\n")

    BENZO_TABLE = pd.read_csv(f"{datadir}FULL_BENZO_{year}.csv")
    print(f"FULL_OPIOID_{year}_ONE_TEMP.csv loaded. Prescriptions: {FULL.shape[0]}")


    FULL['patient_zip'] = FULL['patient_zip'].astype(str)
    FULL = FULL.rename(columns={
        'num_prescribers': 'num_prescribers_past180',
        'num_pharmacies': 'num_pharmacies_past180'
    })
    FULL['date_filled'] = pd.to_datetime(FULL['date_filled'], errors='coerce')
    FULL['presc_until'] = pd.to_datetime(FULL['presc_until'], errors='coerce')
    FULL = FULL.sort_values(by=['patient_id', 'date_filled', 'presc_until'])


    # Drop outlier patients based on comprehensive metrics
    outlier_mask = (
        (FULL['quantity'] >= 1000) |
        (FULL['concurrent_MME'] >= 1000) |
        (FULL['concurrent_methadone_MME'] >= 1000) |
        (FULL['num_prescribers_past180'] > 10) |
        (FULL['num_pharmacies_past180'] > 10) |
        (FULL['num_prescriptions'] >= 100) |
        (FULL['age'] >= 100) |
        (FULL['age'] < 18)
    )
    outlier_patients = FULL.loc[outlier_mask, 'patient_id'].unique()
    FULL = FULL[~FULL['patient_id'].isin(outlier_patients)].copy()
    print(f"Outlier patients removed. Prescriptions remaining: {FULL.shape[0]}")


    # Initialize fields
    FULL = FULL.reset_index(drop=True)
    FULL['prescription_id'] = FULL.index + 1

    init_cols = {
        'num_prior_prescriptions': 0,
        'num_prior_prescriptions_past180': 0,
        'num_prior_prescriptions_past90': 0,
        'num_prior_prescriptions_past30': 0,
        'switch_drug': 0,
        'switch_payment': 0,
        'ever_switch_drug': 0,
        'ever_switch_payment': 0,
        'dose_diff': 0,
        'concurrent_MME_diff': 0,
        'quantity_diff': 0,
        'days_diff': 0,
        'avgMME_past180': FULL['daily_dose'],
        'avgDays_past180': FULL['days_supply'],
        'avgMME_past90': FULL['daily_dose'],
        'avgDays_past90': FULL['days_supply'],
        'avgMME_past30': FULL['daily_dose'],
        'avgDays_past30': FULL['days_supply'],
        'gap': 360
    }
    for col, val in init_cols.items():
        FULL[col] = val

    HMFO_drugs = {"Hydromorphone", "Methadone", "Fentanyl", "Oxymorphone"}
    FULL['HMFO'] = FULL['drug'].isin(HMFO_drugs).astype(int)

    for drug in ["Codeine", "Hydrocodone", "Oxycodone", "Morphine",
                "Hydromorphone", "Methadone", "Fentanyl", "Oxymorphone"]:
        FULL[f"{drug}_MME"] = np.where(FULL['drug'] == drug, FULL['daily_dose'], 0)


    with Pool(cores) as pool:
        results = pool.map(compute_num_prior_benzo, zip(FULL['patient_id'], FULL['prescription_id']))

    results = np.array(results)
    FULL['num_prior_prescriptions_benzo_past180'] = results[:, 0]
    FULL['num_prior_prescriptions_benzo_past90'] = results[:, 1]
    FULL['num_prior_prescriptions_benzo_past30'] = results[:, 2]

    FULL.to_csv(f"{datadir}FULL_OPIOID_{year}_ONE_FEATURE.csv", index=False)


else: # at least two prescriptions

    FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_{case}_TEMP.csv")

    # FULL = FULL.head(500)
    # print("Testing with first 500 rows\n")

    BENZO_TABLE = pd.read_csv(f"{datadir}FULL_BENZO_{year}.csv")

    FULL['patient_zip'] = FULL['patient_zip'].astype(str)
    FULL['date_filled'] = pd.to_datetime(FULL['date_filled'], errors='coerce')
    FULL['presc_until'] = pd.to_datetime(FULL['presc_until'], errors='coerce')
    FULL = FULL.sort_values(by=['patient_id', 'date_filled', 'presc_until'])
    print(f"FULL_OPIOID_{year}_ATLEASTTWO_{case}_TEMP.csv loaded. Prescriptions: {FULL.shape[0]}")


    # Drop outlier patients
    outlier_mask = (
        (FULL['quantity'] >= 1000) |
        (FULL['concurrent_MME'] >= 1000) |
        (FULL['concurrent_methadone_MME'] >= 1000) |
        (FULL['num_prescribers_past180'] > 10) |
        (FULL['num_pharmacies_past180'] > 10) |
        (FULL['num_prescriptions'] >= 100) |
        (FULL['age'] >= 100)
    )
    outlier_patients = FULL.loc[outlier_mask, 'patient_id'].unique()
    FULL = FULL[~FULL['patient_id'].isin(outlier_patients)].copy()
    print(f"Outlier patients removed. Prescriptions remaining: {FULL.shape[0]}")

    FULL['num_prior_prescriptions'] = FULL.groupby('patient_id').cumcount()

    # Compute num prior prescriptions in past 180/90/30 days
    def compute_num_prior_presc(args):
        pat_id, presc_id = args
        patient_data = FULL[FULL['patient_id'] == pat_id]
        row = FULL[FULL['prescription_id'] == presc_id].iloc[0]
        presc_date = row['date_filled']

        def count(days):
            start = presc_date - pd.Timedelta(days=days)
            return patient_data[(patient_data['date_filled'] >= start) &
                                (patient_data['date_filled'] <= presc_date) &
                                (patient_data['prescription_id'] < presc_id)].shape[0]

        return [count(180), count(90), count(30)]

    with Pool(cores) as pool:
        results = pool.map(compute_num_prior_presc, zip(FULL['patient_id'], FULL['prescription_id']))

    results = np.array(results)
    FULL['num_prior_prescriptions_past180'] = results[:, 0]
    FULL['num_prior_prescriptions_past90'] = results[:, 1]
    FULL['num_prior_prescriptions_past30'] = results[:, 2]


    # Switch in drug/payment
    FULL = FULL.sort_values(by=['patient_id', 'date_filled', 'presc_until'])
    FULL['switch_drug'] = ((FULL['patient_id'] == FULL['patient_id'].shift()) & (FULL['drug'] != FULL['drug'].shift())).astype(int)
    FULL['switch_payment'] = ((FULL['patient_id'] == FULL['patient_id'].shift()) & (FULL['payment'] != FULL['payment'].shift())).astype(int)
    FULL.iloc[0, FULL.columns.get_loc('switch_drug')] = 0
    FULL.iloc[0, FULL.columns.get_loc('switch_payment')] = 0

    first_switch = FULL[FULL['switch_drug'] == 1].groupby('patient_id')['date_filled'].min().fillna(pd.Timestamp("2022-01-01"))
    FULL['first_switch_drug'] = FULL['patient_id'].map(first_switch)
    FULL['ever_switch_drug'] = (FULL['date_filled'] >= FULL['first_switch_drug']).astype(int)

    first_switch = FULL[FULL['switch_payment'] == 1].groupby('patient_id')['date_filled'].min().fillna(pd.Timestamp("2022-01-01"))
    FULL['first_switch_payment'] = FULL['patient_id'].map(first_switch)
    FULL['ever_switch_payment'] = (FULL['date_filled'] >= FULL['first_switch_payment']).astype(int)

    FULL.drop(columns=['first_switch_drug', 'first_switch_payment'], inplace=True)


    # Change in dosage/MME/quantity/days
    for col, new_col in [('daily_dose', 'dose_diff'), ('concurrent_MME', 'concurrent_MME_diff'),
                        ('quantity', 'quantity_diff'), ('days_supply', 'days_diff')]:
        FULL[new_col] = FULL[col] - FULL[col].shift()
        FULL.loc[FULL['patient_id'] != FULL['patient_id'].shift(), new_col] = 0


    # Average MME/days in past 180/90/30 days
    def compute_avg(args):
        pat_id, presc_id = args
        patient_data = FULL[FULL['patient_id'] == pat_id]
        row = FULL[FULL['prescription_id'] == presc_id].iloc[0]
        presc_date = row['date_filled']

        def avg(days):
            start = presc_date - pd.Timedelta(days=days)
            prev = patient_data[(patient_data['date_filled'] >= start) &
                                (patient_data['date_filled'] <= presc_date) &
                                (patient_data['prescription_id'] <= presc_id)]
            return [prev['daily_dose'].mean(), prev['days_supply'].mean()]

        return avg(180) + avg(90) + avg(30)

    with Pool(cores) as pool:
        avg_results = pool.map(compute_avg, zip(FULL['patient_id'], FULL['prescription_id']))

    avg_array = np.array(avg_results)
    FULL['avgMME_past180'], FULL['avgDays_past180'] = avg_array[:, 0], avg_array[:, 1]
    FULL['avgMME_past90'], FULL['avgDays_past90'] = avg_array[:, 2], avg_array[:, 3]
    FULL['avgMME_past30'], FULL['avgDays_past30'] = avg_array[:, 4], avg_array[:, 5]

    FULL['HMFO'] = FULL['drug'].isin(["Hydromorphone", "Methadone", "Fentanyl", "Oxymorphone"]).astype(int)

    for drug in ["Codeine", "Hydrocodone", "Oxycodone", "Morphine",
                "Hydromorphone", "Methadone", "Fentanyl", "Oxymorphone"]:
        FULL[f"{drug}_MME"] = np.where(FULL['drug'] == drug, FULL['daily_dose'], 0)


    # Gap between refills
    FULL = FULL.sort_values(['patient_id', 'date_filled'])
    FULL['gap'] = FULL.groupby('patient_id').apply(
        lambda group: pd.Series(
            np.where(group['overlap'] > 0, 0, (group['date_filled'] - group['presc_until'].shift(fill_value=group['date_filled'].iloc[0])).dt.days),
            index=group.index
        )
    ).reset_index(level=0, drop=True)


    # compute_num_prior_benzo defined above
    with Pool(cores) as pool:
        results = pool.map(compute_num_prior_benzo, zip(FULL['patient_id'], FULL['prescription_id']))


    results = np.array(results)
    FULL['num_prior_prescriptions_benzo_past180'] = results[:, 0]
    FULL['num_prior_prescriptions_benzo_past90'] = results[:, 1]
    FULL['num_prior_prescriptions_benzo_past30'] = results[:, 2]

    # print("500 sample rows processed")
    # FULL.to_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_{case}_FEATURE_500.csv", index=False)
    
    FULL.to_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_{case}_FEATURE.csv", index=False)



import os
import sys
import pandas as pd
import numpy as np
basedir = "/export/storage_cures/CURES/"
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

year = sys.argv[1]

FULL = pd.read_csv(f"{basedir}RX_{year}.csv")
FULL_OPIOID = FULL[FULL['class'] == 'Opioid']
print(f"Number of opioid prescriptions in {year}: {len(FULL_OPIOID)}")
FULL_OPIOID.rename(columns={'Unnamed: 0': 'X'}, inplace=True)
dedup_cols = [col for col in FULL_OPIOID.columns if col != 'X']
FULL_OPIOID = FULL_OPIOID.drop_duplicates(subset=dedup_cols)
print(f"Duplicate rows removed. Number of prescriptions left: {len(FULL_OPIOID)}")
FULL_OPIOID = FULL_OPIOID[np.isfinite(FULL_OPIOID['daily_dose'])]
print(f"Number of opioid prescriptions with finite daily_dose in {year}: {len(FULL_OPIOID)}")

# ============ Summary statistics for prescribers and dispensers ============
# Prescriber
PRESCRIBER = FULL_OPIOID.groupby('prescriber_id').agg(
    total_prescriptions=('prescriber_id', 'count'),
    num_unique_patients=('patient_id', 'nunique'),
    num_unique_dispensers=('pharmacy_id', 'nunique'),
    avg_daily_MME=('daily_dose', 'mean'),
    avg_days_supply=('days_supply', 'mean')
).reset_index()

def fmt_mean_sd(series): return f"{series.mean():.1f} ({series.std():.1f})"

latex_rows = [
    f"Yearly number of prescriptions, mean (SD) & {fmt_mean_sd(PRESCRIBER['total_prescriptions'])} \\\\",
    f"Yearly number of unique patients, mean (SD) & {fmt_mean_sd(PRESCRIBER['num_unique_patients'])} \\\\",
    f"Yearly number of unique dispensers, mean (SD) & {fmt_mean_sd(PRESCRIBER['num_unique_dispensers'])} \\\\",
    f"Avg. daily MME per prescribed script, mean (SD) & {fmt_mean_sd(PRESCRIBER['avg_daily_MME'])} \\\\",
    f"Avg. days of supply per prescribed script, mean (SD) & {fmt_mean_sd(PRESCRIBER['avg_days_supply'])} \\\\"
]
latex_table = "\n".join(latex_rows)
print(latex_table)

# dispenser
PHARMACY = FULL_OPIOID.groupby('pharmacy_id').agg(
    total_prescriptions=('pharmacy_id', 'count'),
    num_unique_patients=('patient_id', 'nunique'),
    num_unique_prescribers=('prescriber_id', 'nunique'),
    avg_daily_MME=('daily_dose', 'mean'),
    avg_days_supply=('days_supply', 'mean')
).reset_index()

latex_rows = [
    f"Yearly number of prescriptions, mean (SD) & {fmt_mean_sd(PHARMACY['total_prescriptions'])} \\\\",
    f"Yearly number of unique patients, mean (SD) & {fmt_mean_sd(PHARMACY['num_unique_patients'])} \\\\",
    f"Yearly number of unique prescribers, mean (SD) & {fmt_mean_sd(PHARMACY['num_unique_prescribers'])} \\\\",
    f"Avg. daily MME per prescribed script, mean (SD) & {fmt_mean_sd(PHARMACY['avg_daily_MME'])} \\\\",
    f"Avg. days of supply per prescribed script, mean (SD) & {fmt_mean_sd(PHARMACY['avg_days_supply'])} \\\\"
]
latex_table = "\n".join(latex_rows)
print(latex_table)
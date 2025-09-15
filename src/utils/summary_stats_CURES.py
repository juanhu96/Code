import os
import sys
import pandas as pd
import numpy as np
basedir = "/export/storage_cures/CURES/"
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

FULL_INPUT_2018 = pd.read_csv(f"{datadir}FULL_OPIOID_2018_INPUT.csv")
FULL_INPUT_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
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


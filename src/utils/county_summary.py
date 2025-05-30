'''
STEP 8

INPUT: FULL_OPIOID_2018_INPUT.csv, FULL_OPIOID_2019_INPUT.csv
OUTPUT: SUMMARY TABLE IN LATEX FORM
'''

import sys 
import os
import pandas as pd
import numpy as np
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

year = sys.argv[1]

if year == 'total':
    FULL_INPUT_2018 = pd.read_csv(f"{datadir}FULL_OPIOID_2018_INPUT.csv")
    FULL_INPUT_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
    FULL_INPUT = pd.concat([FULL_INPUT_2018, FULL_INPUT_2019], ignore_index=True)
else:
    FULL_INPUT = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_INPUT.csv")

CA = pd.read_csv(f"{datadir}../CA/zip_county.csv")
FULL_INPUT = pd.merge(CA, FULL_INPUT, left_on="zip", right_on="patient_zip", how="inner")

summary = (FULL_INPUT.groupby("county").agg(
    num_prescriptions=('county', 'size'),
    num_longterm=('long_term_180', 'sum'),
    # num_benzo=('concurrent_benzo', 'sum'),
    num_longterm_benzo=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'concurrent_benzo'] == 1)).sum()),
    num_longterm_age30=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'age'] >= 30)).sum()),
    num_longterm_mme30=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'concurrent_MME'] >= 30)).sum()),
    num_longterm_mme75=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'concurrent_MME'] >= 75)).sum()),
    num_longterm_mme100=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'concurrent_MME'] >= 100)).sum()),
    num_longterm_dailymme100=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'daily_dose'] >= 100)).sum()),
    num_longterm_HMFO=('long_term_180', lambda x: (((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & 
    ((FULL_INPUT.loc[x.index, 'Hydromorphone'] == 1) | (FULL_INPUT.loc[x.index, 'Methadone'] == 1) | 
    (FULL_INPUT.loc[x.index, 'Fentanyl'] == 1) | (FULL_INPUT.loc[x.index, 'Oxymorphone'] == 1))).sum())),
    num_longterm_Hydrocodone=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'Hydrocodone'] == 1)).sum()),
    num_longterm_medicaid=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'Medicaid'] == 1)).sum()),
    num_longterm_prescribers3=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'num_prescribers_past180'] >= 3)).sum()),
    num_unique_patients=('patient_id', pd.Series.nunique),
    num_patients_longterm=('patient_id', lambda x: FULL_INPUT.loc[x.index].groupby('patient_id')['long_term_180'].max().gt(0).sum()),
    num_patients_longterm_benzo=('patient_id', lambda x: (FULL_INPUT.loc[x.index].groupby('patient_id')[['long_term_180', 'concurrent_benzo']].max().all(axis=1).sum()))
    ).reset_index()
)

total_row = pd.DataFrame([{
    'county': 'TOTAL',
    'num_prescriptions': len(FULL_INPUT),
    'num_longterm': FULL_INPUT['long_term_180'].sum(),
    # 'num_benzo': FULL_INPUT['concurrent_benzo'].sum(),
    'num_longterm_benzo': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['concurrent_benzo'] == 1)).sum(),
    'num_longterm_age30': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['age'] >= 30)).sum(),
    'num_longterm_mme30': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['concurrent_MME'] >= 30)).sum(),
    'num_longterm_mme75': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['concurrent_MME'] >= 75)).sum(),
    'num_longterm_mme100': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['concurrent_MME'] >= 100)).sum(),
    'num_longterm_dailymme100': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['daily_dose'] >= 100)).sum(),
    'num_longterm_HMFO': ((FULL_INPUT['long_term_180'] == 1) & ((FULL_INPUT['Hydromorphone'] == 1) | (FULL_INPUT['Methadone'] == 1) | (FULL_INPUT['Fentanyl'] == 1) | (FULL_INPUT['Oxymorphone'] == 1))).sum(),
    'num_longterm_Hydrocodone': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['Hydrocodone'] == 1)).sum(),
    'num_longterm_medicaid': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['Medicaid'] == 1)).sum(),
    'num_longterm_prescribers3': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['num_prescribers_past180'] >= 3)).sum(),
    'num_unique_patients': FULL_INPUT['patient_id'].nunique(),
    'num_patients_longterm': FULL_INPUT.groupby('patient_id')['long_term_180'].max().gt(0).sum(),
    'num_patients_longterm_benzo': (FULL_INPUT.groupby('patient_id')[['long_term_180', 'concurrent_benzo']].max().all(axis=1).sum())
}])

summary = pd.concat([summary, total_row], ignore_index=True)
summary["prec_benzo_longterm"] = (summary["num_longterm_benzo"] / summary["num_longterm"]).round(3)
summary["prec_age_longterm"] = (summary["num_longterm_age30"] / summary["num_longterm"]).round(3)
summary["prec_mme30_longterm"] = (summary["num_longterm_mme30"] / summary["num_longterm"]).round(3)
summary["prec_mme75_longterm"] = (summary["num_longterm_mme75"] / summary["num_longterm"]).round(3)
summary["prec_mme100_longterm"] = (summary["num_longterm_dailymme100"] / summary["num_longterm"]).round(3)
summary["prec_dailymme100_longterm"] = (summary["num_longterm_mme100"] / summary["num_longterm"]).round(3)
summary["prec_HMFO_longterm"] = (summary["num_longterm_HMFO"] / summary["num_longterm"]).round(3)
summary["prec_Hydrocodone_longterm"] = (summary["num_longterm_Hydrocodone"] / summary["num_longterm"]).round(3)
summary["prec_medicaid_longterm"] = (summary["num_longterm_medicaid"] / summary["num_longterm"]).round(3)
summary["prec_prescribers3_longterm"] = (summary["num_longterm_prescribers3"] / summary["num_longterm"]).round(3)

summary.to_csv(f"{resultdir}county_summary_{year}.csv", index=False)
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

# ========== LONG TERM PRESCRIPTIONS ==========

# summary = (FULL_INPUT.groupby("county").agg(
#     num_prescriptions=('county', 'size'),
#     num_longterm=('long_term_180', 'sum'),
#     num_longterm_prior=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'num_prior_prescriptions'] >= 1)).sum()),
#     num_longterm_days=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'days_supply'] >= 10)).sum()),
#     num_longterm_dailymme=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'daily_dose'] >= 90)).sum()),
#     num_longterm_HMFO=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & 
#     ((FULL_INPUT.loc[x.index, 'Hydromorphone'] == 1) | (FULL_INPUT.loc[x.index, 'Methadone'] == 1) | 
#     (FULL_INPUT.loc[x.index, 'Fentanyl'] == 1) | (FULL_INPUT.loc[x.index, 'Oxymorphone'] == 1))).sum()),
#     num_longterm_longacting=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'long_acting'] == 1)).sum()),
#     num_longterm_topprescriber=('long_term_180', lambda x: ((FULL_INPUT.loc[x.index, 'long_term_180'] == 1) & (FULL_INPUT.loc[x.index, 'prescriber_yr_avg_days_above75'] == 1)).sum())
#     ).reset_index()
#     )

# total_row = pd.DataFrame([{
#     'county': 'California',
#     'num_prescriptions': len(FULL_INPUT),
#     'num_longterm': FULL_INPUT['long_term_180'].sum(),
#     'num_longterm_prior': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['num_prior_prescriptions'] >= 1)).sum(),
#     'num_longterm_days': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['days_supply'] >= 10)).sum(),
#     'num_longterm_dailymme': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['daily_dose'] >= 90)).sum(),
#     'num_longterm_HMFO': ((FULL_INPUT['long_term_180'] == 1) & 
#                           ((FULL_INPUT['Hydromorphone'] == 1) | (FULL_INPUT['Methadone'] == 1) | 
#                            (FULL_INPUT['Fentanyl'] == 1) | (FULL_INPUT['Oxymorphone'] == 1))).sum(),
#     'num_longterm_longacting': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['long_acting'] == 1)).sum(),
#     'num_longterm_topprescriber': ((FULL_INPUT['long_term_180'] == 1) & (FULL_INPUT['prescriber_yr_avg_days_above75'] == 1)).sum()
# }])

# summary = pd.concat([summary, total_row], ignore_index=True)
# summary['presc_longterm_prior'] = (summary['num_longterm_prior'] / summary['num_longterm']).round(3)
# summary['presc_longterm_days'] = (summary['num_longterm_days'] / summary['num_longterm']).round(3)
# summary['presc_longterm_dailymme'] = (summary['num_longterm_dailymme'] / summary['num_longterm']).round(3)
# summary['presc_longterm_HMFO'] = (summary['num_longterm_HMFO'] / summary['num_longterm']).round(3)
# summary['presc_longterm_longacting'] = (summary['num_longterm_longacting'] / summary['num_longterm']).round(3)
# summary['presc_longterm_topprescriber'] = (summary['num_longterm_topprescriber'] / summary['num_longterm']).round(3)
# summary.to_csv(f"{resultdir}county_summary_{year}.csv", index=False)

# ==========================================

summary = (FULL_INPUT.groupby("county").agg(
    num_prescriptions=('county', 'size'),
    num_longterm=('long_term_180', 'sum'),
    num_prior=('long_term_180', lambda x: (FULL_INPUT.loc[x.index, 'num_prior_prescriptions'] >= 1).sum()),
    num_days=('long_term_180', lambda x: (FULL_INPUT.loc[x.index, 'days_supply'] >= 10).sum()),
    num_dailymme=('long_term_180', lambda x: (FULL_INPUT.loc[x.index, 'daily_dose'] >= 90).sum()),
    num_HMFO=('long_term_180', lambda x: (
        (FULL_INPUT.loc[x.index, 'Hydromorphone'] == 1) | 
        (FULL_INPUT.loc[x.index, 'Methadone'] == 1) | 
        (FULL_INPUT.loc[x.index, 'Fentanyl'] == 1) | 
        (FULL_INPUT.loc[x.index, 'Oxymorphone'] == 1)
    ).sum()),
    num_longacting=('long_term_180', lambda x: (FULL_INPUT.loc[x.index, 'long_acting'] == 1).sum()),
    num_topprescriber=('long_term_180', lambda x: (FULL_INPUT.loc[x.index, 'prescriber_yr_avg_days_above75'] == 1).sum())
).reset_index())

total_row = pd.DataFrame([{
    'county': 'California',
    'num_prescriptions': len(FULL_INPUT),
    'num': FULL_INPUT['long_term_180'].sum(),
    'num_prior': (FULL_INPUT['num_prior_prescriptions'] >= 1).sum(),
    'num_days': (FULL_INPUT['days_supply'] >= 10).sum(),
    'num_dailymme': (FULL_INPUT['daily_dose'] >= 90).sum(),
    'num_HMFO': (
        (FULL_INPUT['Hydromorphone'] == 1) | 
        (FULL_INPUT['Methadone'] == 1) | 
        (FULL_INPUT['Fentanyl'] == 1) | 
        (FULL_INPUT['Oxymorphone'] == 1)
    ).sum(),
    'num_longacting': (FULL_INPUT['long_acting'] == 1).sum(),
    'num_topprescriber': (FULL_INPUT['prescriber_yr_avg_days_above75'] == 1).sum()
}])

summary = pd.concat([summary, total_row], ignore_index=True)
summary['presc_prior'] = (summary['num_prior'] / summary['num_prescriptions']).round(3)
summary['presc_days'] = (summary['num_days'] / summary['num_prescriptions']).round(3)
summary['presc_dailymme'] = (summary['num_dailymme'] / summary['num_prescriptions']).round(3)
summary['presc_HMFO'] = (summary['num_HMFO'] / summary['num_prescriptions']).round(3)
summary['presc_longacting'] = (summary['num_longacting'] / summary['num_prescriptions']).round(3)
summary['presc_topprescriber'] = (summary['num_topprescriber'] / summary['num_prescriptions']).round(3)
summary.to_csv(f"{resultdir}county_summary_{year}.csv", index=False)
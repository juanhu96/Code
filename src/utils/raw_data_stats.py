# Raw data description

import pandas as pd
import numpy as np
basedir = "/export/storage_cures/CURES/"
datadir = '/export/storage_cures/CURES/Processed/'

year = 2018

if False:
    FULL_CURRENT = pd.read_csv(f"{basedir}RX_{year}.csv")
    print(FULL_CURRENT.columns.to_list())

if False:    
    df = pd.read_csv(f'{datadir}/Stumps/FULL_{year}_Explore_STUMPS_median_1.csv', delimiter=",")
    print(df.columns.to_list())

if True:
    FULL_INPUT_2018 = pd.read_csv(f"{datadir}FULL_OPIOID_2018_INPUT.csv")
    FULL_INPUT_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
    FULL_INPUT = pd.concat([FULL_INPUT_2018, FULL_INPUT_2019], ignore_index=True)

    # get the unique number of paitent_ids
    unique_patient_ids = FULL_INPUT['patient_id'].nunique()
    unique_patient_ids_2018 = FULL_INPUT_2018['patient_id'].nunique()
    unique_patient_ids_2019 = FULL_INPUT_2019['patient_id'].nunique()

    print(f"Unique patient IDs in FULL_INPUT: {unique_patient_ids}")
    print(f"Unique patient IDs in FULL_INPUT_2018: {unique_patient_ids_2018}")
    print(f"Unique patient IDs in FULL_INPUT_2019: {unique_patient_ids_2019}")

    # get the overlap of patient_ids between FULL_INPUT_2018 and FULL_INPUT_2019
    overlap_patient_ids = set(FULL_INPUT_2018['patient_id']).intersection(set(FULL_INPUT_2019['patient_id']))
    print(f"Number of overlapping patient IDs between FULL_INPUT_2018 and FULL_INPUT_2019: {len(overlap_patient_ids)}")

    # get the prescription corresponding to the overlapping patient_ids
    overlap_prescriptions = FULL_INPUT[FULL_INPUT['patient_id'].isin(overlap_patient_ids)]
    overlap_prescriptions = overlap_prescriptions.sort_values(by=['patient_id', 'date_filled'])
    columns_of_interest = ['patient_id', 'date_filled']
    overlap_prescriptions = overlap_prescriptions[columns_of_interest]
    
    print(overlap_prescriptions.head(50))
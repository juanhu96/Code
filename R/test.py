#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2024
Compute spatial effect

@author: Jingyuan Hu
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

pd.set_option('display.max_columns', None) # show all columns

datadir = "/export/storage_cures/CURES/Processed/"
exportdir = "/export/storage_cures/CURES/Processed/Patient_zip/"
year = 2018

FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_ATLEASTTWO_1_FEATURE.csv")
# FULL['patient_zip'] = FULL['patient_zip'].astype(str)

# FULL = FULL[(FULL['patient_zip'] == '90018') & (FULL['date_filled'] == '11/26/2018')]

PATIENT_ZIP = FULL.groupby(['patient_zip', 'date_filled']).agg(
    day_prescriptions=('patient_zip', 'count'), 
    day_patients=('patient_zip', pd.Series.nunique)
).reset_index()

prob_list = np.linspace(0.0, 1.0, num=21, endpoint=False)
patient_zip_list = PATIENT_ZIP['patient_zip'].quantile(prob_list).round().astype(int).values
print(patient_zip_list)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 9 2022

@author: Jingyuan Hu
"""

import os
# os.chdir('/Users/jingyuanhu/Desktop/Research/Interpretable Opioid')
import csv
import time
import random
import numpy as np
import pandas as pd

import utils.stumps as stumps


os.chdir('/mnt/phd/jihu/opioid')

year = 2019
scenario = '' # default empty, first prescription etc.

###############################################################################
###############################################################################
###############################################################################

FULL = pd.read_csv('Data/FULL_' + str(year) + scenario + '_LONGTERM_INPUT.csv', delimiter = ",", 
                    dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                          'num_prescribers': int, 'num_pharmacies': int,
                          'concurrent_benzo': int, 'consecutive_days': int})

FULL = FULL.fillna(0)
    
###############################################################################
###############################################################################
###############################################################################


## Drop the columns that start with alert
FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
FULL = FULL.drop(columns = ['switch_payment', 'drug_payment'])

x_all = FULL[['age', 'quantity', 'quantity_per_day', 'total_dose', 
              'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
              'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
              'concurrent_benzo_same', 'concurrent_benzo_diff', 
              'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
              'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
              'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]

cutoffs = []
for column_name in x_all.columns:
    if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
        cutoffs.append([n for n in range(0, 10)])
    elif column_name == 'concurrent_benzo' or column_name == 'concurrent_benzo_same' or \
        column_name == 'concurrent_benzo_diff' or column_name == 'num_presc':
        cutoffs.append([0, 1, 2, 3, 4, 5, 10])
    elif column_name == 'consecutive_days' or column_name == 'age' or column_name == 'concurrent_methadone_MME' or \
        column_name == 'days_diff':
        cutoffs.append([n for n in range(0, 90) if n % 10 == 0])
    elif column_name == 'dose_diff' or column_name == 'concurrent_MME_diff':
        cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
    else:
        cutoffs.append([n for n in range(0, 200) if n % 10 == 0])

## Divide into 20 folds
N = 20
FULL_splited = np.array_split(FULL, N)
for i in range(N):
    FULL_fold = FULL_splited[i]
    x = FULL_fold[['age', 'quantity', 'quantity_per_day', 'total_dose', 
                   'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                   'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                   'concurrent_benzo_same', 'concurrent_benzo_diff', 
                   'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                   'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                   'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
    
    x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
    # bind the data with the ones that was not for create_stumps
    x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'quantity_per_day', 'total_dose', 
                                               'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                               'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                                               'concurrent_benzo_same', 'concurrent_benzo_diff', 
                                               'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                                               'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                               'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME'])]
    
    new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
    print(new_data.shape)
    new_data.to_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', header=True, index=False)
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Stumps-related helper functions
"""

import sys 
import csv
import numpy as np
import pandas as pd
import utils.stumps as stumps
import os
from multiprocessing import Pool

pd.set_option('display.max_columns', None) # show all columns

year = sys.argv[1]
first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])

def main(year, first, upto180, cutoff='greater'):

    if cutoff == 'greater':
    
        LTOUR_features = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo']
        LTOUR_features_cts = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo']

        focal_features = ['days_supply', 'daily_dose',
                          'Hydrocodone', 'Oxycodone', 'Codeine', 'Morphine', 'HMFO', 'long_acting',
                          'CashCredit', 'Medicare', 'Medicaid', 'Other', 'CommercialIns', 'MilitaryIns', 'WorkersComp', 'IndianNation']
        focal_features_cts = ['days_supply', 'daily_dose']

        history_features = ['num_prior_prescriptions', 'diff_MME', 'diff_days',
                            'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment']
        history_features_cts = ['num_prior_prescriptions', 'diff_MME', 'diff_days']

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
        
        feature_list = LTOUR_features + ['patient_gender'] + focal_features + history_features
        cts_feature_list = LTOUR_features_cts + focal_features_cts + history_features_cts
            
        create_stumps(year, first, upto180, feature_list, cts_feature_list, percentile_list)

    else: raise KeyError("Cutoff type undefined.\n")



def create_stumps(year, 
                  first, 
                  upto180, 
                  feature_list, 
                  cts_feature_list, 
                  percentile_list,
                  datadir="/export/storage_cures/CURES/Processed/",
                  output_columns=False):
    
    print(f"Start creating stumps for year {year} with features {feature_list} and percentiles for {percentile_list}")

    if first:
        file_suffix = "_FIRST_INPUT"
    elif upto180: # up to first long_term_180
        file_suffix = "_UPTOFIRST_INPUT"
    else: # up to first long_term
        file_suffix = "_INPUT"

    file_path = f'{datadir}FULL_OPIOID_{year}{file_suffix}.csv'

    FULL = pd.read_csv(file_path, delimiter=",", dtype={'concurrent_MME': float, 
                                                        'concurrent_methadone_MME': float,
                                                        'num_prescribers_past180': int,
                                                        'num_pharmacies_past180': int,
                                                        'concurrent_benzo': int,
                                                        'consecutive_days': int})

    if output_columns: print(FULL.columns.values.tolist())
    
    FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)
    precentile_features = [col for col in FULL.columns if any(col.startswith(f'{prefix}_above') for prefix in percentile_list)]
    feature_list += precentile_features
    FULL = FULL[feature_list]

    nan_cols = FULL.columns[FULL.isna().any()]
    if len(nan_cols) > 0: 
        cols_with_na = nan_cols.tolist()
        print(f"Columns with NaNs: {cols_with_na}") 
        FULL = FULL.dropna(subset=cols_with_na)

    FULL = FULL.astype(int)

    cutoffs = []
    for column_name in FULL.columns:
        if column_name == 'concurrent_MME' or column_name == 'daily_dose':
            cutoffs.append([25, 50, 75, 90])
        elif column_name == 'num_prescribers_past180' or column_name == 'num_pharmacies_past180':
            cutoffs.append([n for n in range(2, 7)])
        elif column_name == 'days_supply':
            cutoffs.append([3, 5, 7, 10, 14])
        elif column_name == 'concurrent_benzo' or column_name == 'num_prior_prescriptions':
            cutoffs.append([1])
        elif column_name == 'diff_MME' or column_name == 'diff_quantity' or column_name == 'diff_days':
            cutoffs.append([1]) # greater or not
        else:
            pass

    N = 20 # number of splits/cores
    FULL_splited = np.array_split(FULL, N)
    args = [(FULL_splited[i], i, cts_feature_list, cutoffs, datadir, year, first, upto180) for i in range(N)]

    if not os.path.exists(f'{datadir}/Stumps/'): os.makedirs(f'{datadir}/Stumps/')
    with Pool(N) as pool: pool.map(process_fold, args)
    print("Stumps done!\n")
    
    return     


def process_fold(args):
    
    FULL_fold, i, cts_feature_list, cutoffs, datadir, year, first, upto180 = args # unpack

    x = FULL_fold[cts_feature_list]
    x_stumps = create_stumps_helper(x.values, x.columns, cutoffs)
        
    x_rest = FULL_fold[FULL_fold.columns.drop(cts_feature_list)]
        
    new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis=1)

    if first:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"

    file_path = f'{datadir}/Stumps/FULL_{year}{file_suffix}{i}.csv'
    new_data.to_csv(file_path, header=True, index=False)

    print(f'Processed fold {i} and saved.')


def create_stumps_helper(data, columns, cutpoints):
    
    """
    @parameters:
    
    - data: featres; np.array
    - columns: feature names
    - cutpoints: cut off points used to create stumps
    
    """
    
    ## data dimension
    final_data = []
    final_names = []
    n, p = data.shape[0], data.shape[1]
    
    ## loop through features
    for i in range(len(columns)):
        ## subset feature
        feature = columns[i]
        feature_values = data[:,i]
        cutoff = cutpoints[i]
        cutoff_length = len(cutoff)
        names = []
        
        ## create stumps
        ### if the variable is binary, then set the cutoff point value to be 1.
        
        stumps = np.zeros([n, cutoff_length])
        for k in range(cutoff_length):
            for j in range(n):
                if feature_values[j] >= cutoff[k]: stumps[j,k] = 1
            names.append(feature + str(cutoff[k]))
        
        ## store stumps
        final_data.append(stumps)
        final_names.append(names)
        
        ## post process
        new_data = pd.DataFrame(final_data[0], columns=final_names[0])
        for s in range(len(final_data)-1):
            a = pd.DataFrame(final_data[s+1], columns=final_names[s+1])
            new_data = pd.concat([new_data, a], axis=1)
    
    return new_data


if __name__ == "__main__":
    main(year, first, upto180)
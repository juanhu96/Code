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
case = sys.argv[2]
first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])

def main(year, case, first, upto180, cutoff='greater'):

    if cutoff == 'greater':
        
    # ['patient_id', 'patient_gender', 'patient_zip', 'quantity', 'days_supply', 'date_filled', 
    #  'daily_dose', 'total_dose', 'max_dose', 'age', 'concurrent_MME', 'concurrent_methadone_MME',
    #  'num_prescribers_past180', 'num_pharmacies_past180', 'consecutive_days', 'concurrent_benzo', 
    #  'concurrent_benzo_same', 'concurrent_benzo_diff', 'days_to_long_term', 'long_term_180', 
    #  'num_prior_prescriptions', 'num_prior_prescriptions_past180', 'num_prior_prescriptions_past90', 
    #  'num_prior_prescriptions_past30', 'switch_drug', 'switch_payment', 'ever_switch_drug', 
    #  'ever_switch_payment', 'dose_diff', 'concurrent_MME_diff', 'quantity_diff', 'days_diff', 
    #  'avgMME_past180', 'avgDays_past180', 'avgMME_past90', 'avgDays_past90', 'avgMME_past30', 
    #  'avgDays_past30', 'HMFO', 'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
    #  'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME', 'gap', 
    #  'num_prior_prescriptions_benzo_past180', 'num_prior_prescriptions_benzo_past90', 
    #  'num_prior_prescriptions_benzo_past30', 'patient_HPIQuartile', 'prescriber_HPIQuartile', 
    #  'pharmacy_HPIQuartile', 'zip_pop', 'zip_pop_density', 'median_household_income', 
    #  'family_poverty_pct', 'unemployment_pct', 'patient_zip_yr_num_prescriptions', 
    #  'patient_zip_yr_num_patients', 'patient_zip_yr_num_pharmacies', 'patient_zip_yr_avg_MME', 
    #  'patient_zip_yr_avg_days', 'patient_zip_yr_avg_quantity', 'patient_zip_yr_num_prescriptions_quartile', 
    #  'patient_zip_yr_num_patients_quartile', 'patient_zip_yr_num_pharmacies_quartile', 
    #  'patient_zip_yr_avg_MME_quartile', 'patient_zip_yr_avg_days_quartile', 'patient_zip_yr_avg_quantity_quartile', 
    #  'prescriber_yr_num_prescriptions', 'prescriber_yr_num_patients', 'prescriber_yr_num_pharmacies', 
    #  'prescriber_yr_avg_MME', 'prescriber_yr_avg_days', 'prescriber_yr_avg_quantity', 
    #  'prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 
    #  'prescriber_yr_num_pharmacies_quartile', 'prescriber_yr_avg_MME_quartile', 'prescriber_yr_avg_days_quartile', 
    #  'prescriber_yr_avg_quantity_quartile', 'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 
    #  'pharmacy_yr_num_prescribers', 'pharmacy_yr_avg_MME', 'pharmacy_yr_avg_days', 'pharmacy_yr_avg_quantity', 
    #  'pharmacy_yr_num_prescriptions_quartile', 'pharmacy_yr_num_patients_quartile', 
    #  'pharmacy_yr_num_prescribers_quartile', 'pharmacy_yr_avg_MME_quartile', 'pharmacy_yr_avg_days_quartile', 
    #  'pharmacy_yr_avg_quantity_quartile', 'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'Hydromorphone', 
    #  'Methadone', 'Fentanyl', 'Oxymorphone', 'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit', 
    #  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation', 'zip_pop_density_quartile', 
    #  'median_household_income_quartile', 'family_poverty_pct_quartile', 'unemployment_pct_quartile', 
    #  'patient_zip_yr_num_prescriptions_per_pop', 'patient_zip_yr_num_patients_per_pop', 
    #  'patient_zip_yr_num_prescriptions_per_pop_quartile', 'patient_zip_yr_num_patients_per_pop_quartile']

        LTOUR_feature_list = ['concurrent_MME', 'num_prescribers_past180',
                              'num_pharmacies_past180', 'concurrent_benzo',
                              'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO', 'long_acting', # new feature
                              'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                              'num_prior_prescriptions', 'avgDays_past180', 'diff_MME', 'diff_quantity', 'diff_days',
                              'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment']

        LTOUR_stumps_feature_list = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo',
                                     'num_prior_prescriptions', 'avgDays_past180', 'diff_MME', 'diff_quantity', 'diff_days']

        if case == 'LTOUR':

            quartile_list = []
            create_stumps(year, case, first, upto180, feature_list, LTOUR_stumps_feature_list, quartile_list)
        
        elif case == 'Explore':
            
            LTOUR_feature_list.remove('avgDays_past180')
            LTOUR_stumps_feature_list.remove('avgDays_past180')

            quartile_list = [# 'patient_HPIQuartile', 'prescriber_HPIQuartile', 'pharmacy_HPIQuartile',
                             'patient_zip_yr_num_prescriptions_quartile', 'patient_zip_yr_num_patients_quartile', 
                             'patient_zip_yr_num_pharmacies_quartile', 'patient_zip_yr_avg_MME_quartile', 
                             'patient_zip_yr_avg_days_quartile', 'patient_zip_yr_avg_quantity_quartile', 
                             'patient_zip_yr_num_prescriptions_per_pop_quartile', 'patient_zip_yr_num_patients_per_pop_quartile',
                             'prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 
                             'prescriber_yr_num_pharmacies_quartile', 'prescriber_yr_avg_MME_quartile', 
                             'prescriber_yr_avg_days_quartile', 'prescriber_yr_avg_quantity_quartile',
                             'pharmacy_yr_num_prescriptions_quartile', 'pharmacy_yr_num_patients_quartile', 
                             'pharmacy_yr_num_prescribers_quartile', 'pharmacy_yr_avg_MME_quartile', 
                             'pharmacy_yr_avg_days_quartile', 'pharmacy_yr_avg_quantity_quartile',
                             'zip_pop_density_quartile', 'median_household_income_quartile', 
                             'family_poverty_pct_quartile', 'unemployment_pct_quartile'] # need to encode

            cts_list = ['age', 'days_supply', 'daily_dose', 'patient_zip_yr_avg_days', 'patient_zip_yr_avg_quantity', 'patient_zip_yr_avg_MME'] # need to create stumps

            feature_list = LTOUR_feature_list + quartile_list + cts_list + ['patient_gender']
            stumps_feature_list = LTOUR_stumps_feature_list + cts_list
            
            create_stumps(year, case, first, upto180, feature_list, stumps_feature_list, quartile_list)

        else: 
            
            raise ValueError("Feature case undefined.\n")

    elif cutoff == 'interval': raise Exception("Interval type incomplete.\n")
    else: raise KeyError("Cutoff type undefined.\n")



def create_stumps(year, case, first, upto180, feature_list, stumps_feature_list, quartile_list,
                  datadir="/export/storage_cures/CURES/Processed/",
                  output_columns=True, median=True):
    
    print(f"Start creating stumps for year {year} with features {feature_list}\n")

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
                                                        'consecutive_days': int}) # .fillna(0)

    if output_columns: print(FULL.columns.values.tolist())

    FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)
    FULL = FULL[feature_list]
    
    # print(FULL['patient_zip_yr_num_prescriptions_quartile'].value_counts(), # zip in the bottom quartile have less prescriptions
    #       FULL['prescriber_yr_avg_days_quartile'].value_counts(), 
    #       FULL['unemployment_pct_quartile'].value_counts())

    if quartile_list: 
        
        if median:
            print('Note: encoding quartile values to binary...\n')
            for col in quartile_list:
                FULL[col] = FULL[col].replace({1: 0, 2: 0, 3: 1, 4: 1})
            FULL = FULL.dropna(subset=quartile_list)
            FULL = pd.get_dummies(FULL, columns=quartile_list)
        else:
            FULL = FULL.dropna(subset=quartile_list) # drop NA rows
            FULL = pd.get_dummies(FULL, columns=quartile_list)

    cutoffs = []
    for column_name in FULL.columns:
        if column_name == 'concurrent_MME' or column_name == 'concurrent_methadone_MME' or \
            column_name == 'patient_zip_yr_avg_MME' or column_name == 'patient_zip_yr_avg_quantity':
            cutoffs.append([10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300])
        elif column_name == 'daily_dose':
            cutoffs.append([3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100])
        elif column_name == 'num_prescribers_past180' or column_name == 'num_pharmacies_past180':
            cutoffs.append([n for n in range(2, 7)])
        elif column_name == 'concurrent_benzo':
            cutoffs.append([1])
        elif column_name == 'consecutive_days':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30, 60, 90])
        elif column_name == 'quantity':
            cutoffs.append([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300])
        elif column_name == 'num_prior_prescriptions':
            cutoffs.append([1]) # prior prescription or not
        elif column_name == 'avgDays_past180' or column_name == 'patient_zip_yr_avg_days':
            cutoffs.append([3, 5, 7, 10, 14, 21, 25, 30, 60])
        elif column_name == 'diff_MME' or column_name == 'diff_quantity' or column_name == 'diff_days':
            cutoffs.append([1]) # greater or not
        elif column_name == 'age':
            cutoffs.append([20, 30, 40, 50, 60, 70, 80])
        elif column_name == 'days_supply':
            cutoffs.append([3, 5, 7, 10, 14, 21, 30])
        else:
            pass

    N = 20 # number of splits/cores
    FULL_splited = np.array_split(FULL, N)
    args = [(FULL_splited[i], i, stumps_feature_list, cutoffs, datadir, year, case, first, upto180, median) for i in range(N)]

    if not os.path.exists(f'{datadir}/Stumps/'): os.makedirs(f'{datadir}/Stumps/')

    with Pool(N) as pool:
        pool.map(process_fold, args) # requires process_fold be global and single argument

    print("Stumps done!\n")
    
    return     


def process_fold(args):
    
    FULL_fold, i, stumps_feature_list, cutoffs, datadir, year, case, first, upto180, median = args # unpack

    x = FULL_fold[stumps_feature_list]
    x_stumps = create_stumps_helper(x.values, x.columns, cutoffs)
        
    x_rest = FULL_fold[FULL_fold.columns.drop(stumps_feature_list)]
        
    new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis=1)

    if first:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"

    if median: file_suffix += 'median_'
    file_path = f'{datadir}/Stumps/FULL_{year}_{case}{file_suffix}{i}.csv'
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
    main(year, case, first, upto180)
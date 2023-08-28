#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Stumps-related helper functions
"""

import csv
import numpy as np
import pandas as pd
import utils.stumps as stumps


feature_list = ['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
'num_prescribers', 'num_pharmacies', 'concurrent_benzo', 'consecutive_days', 'num_presc', 
'MME_diff', 'days_diff', 'quantity_diff', 'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME', 'avgMME', 'avgDays']



def create_stumps(year, scenario='', feature_list=feature_list, datadir='/mnt/phd/jihu/opioid/Data/'):
    
    '''
    year = 2019
    scenario = '' # default empty, FIRST, SECOND, THIRD
    '''
        
    FULL = pd.read_csv(f'{datadir}FULL_{str(year)}{scenario}_LONGTERM_INPUT_UPTOFIRST.csv', delimiter = ",",\
    dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float, 'num_prescribers': int,\
    'num_pharmacies': int, 'concurrent_benzo': int, 'consecutive_days': int}).fillna(0)

    
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))] # drop columns start with alert
    FULL = FULL.drop(columns = ['drug_payment'])
    x_all = FULL[feature_list]

    cutoffs = []
    for column_name in x_all.columns:
        if column_name == 'age':
            cutoffs.append([18, 25, 30, 40, 50, 60, 70, 80, 90])
        elif column_name == 'quantity':
            cutoffs.append([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300])
        elif column_name == 'days_supply':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        elif column_name == 'concurrent_MME' or column_name == 'concurrent_methadone_MME':
            cutoffs.append([10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300])
        elif column_name == 'num_prescribers' or column_name == 'num_pharmacies':
            cutoffs.append([n for n in range(1, 11)])
        elif column_name == 'concurrent_benzo':
            cutoffs.append([1])
        elif column_name == 'consecutive_days':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30, 60, 90])
        elif column_name == 'num_presc':
            cutoffs.append([1, 2, 3, 4, 5, 6, 10, 15])
        elif column_name == 'MME_diff' or column_name == 'quantity_diff':
            cutoffs.append([10, 25, 50, 75, 100, 150])
        elif column_name == 'days_diff':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        else:
            cutoffs.append([10, 20, 30, 40, 50, 75, 100, 200, 300]) # interaction MMEs
        

    ## Divide into 20 folds
    N = 20
    FULL_splited = np.array_split(FULL, N)
    for i in range(N):

        FULL_fold = FULL_splited[i]        
        x = FULL_fold[feature_list]
        x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
        x_rest = FULL_fold[FULL_fold.columns.drop(feature_list)]
        
        new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
        print(new_data.shape)
        new_data.to_csv(f'{datadir}FULL_{str(year)}{scenario}_STUMPS_UPTOFIRST{str(i)}.csv', header=True, index=False)
        
        

###############################################################################
###############################################################################
###############################################################################

    

def create_intervals(year, scenario='flexible', feature_list=feature_list, datadir='/mnt/phd/jihu/opioid/Data/'):
    
    '''
    Create intervals stumps for the dataset
    For this we also need to edit stumps.create_stumps as well
    
    Parameters
    ----------
    year
    scenario: basic feature (flexible) / full
    '''

    FULL = pd.read_csv(f'{datadir}FULL_{str(year)}_LONGTERM_INPUT.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int}).fillna(0)
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
    FULL = FULL.drop(columns = ['drug_payment'])
    
    
    if scenario == 'flexible':
        x_all = FULL[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
        
        cutoffs_i = []
        for column_name in ['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_i.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_i.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_i.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_i.append([n for n in range(0, 200) if n % 10 == 0])
        
        cutoffs_s = []
        for column_name in ['num_prescribers', 'num_pharmacies', 'concurrent_benzo']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_s.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_s.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_s.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_s.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):
            FULL_fold = FULL_splited[i]
            x = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                          'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
            
            x_i = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']]
            x_s = FULL_fold[['num_prescribers', 'num_pharmacies', 'concurrent_benzo']]
            
            x_intervals = stumps.create_intervals(x_i.values, x_i.columns, cutoffs_i)
            x_stumps = stumps.create_stumps(x_s.values, x_s.columns, cutoffs_s)
            
            new_data = pd.concat([x_intervals.reset_index(drop=True), x_stumps.reset_index(drop=True)], axis = 1)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)  
    

    elif scenario == 'full':
        x_all = FULL[feature_list]
        
        cutoffs = []
        for column_name in x_all.columns:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo' or column_name == 'concurrent_benzo_same' or \
                column_name == 'concurrent_benzo_diff' or column_name == 'num_presc':
                cutoffs.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME' or \
                column_name == 'days_diff':
                cutoffs.append([n for n in range(0, 90) if n % 10 == 0])
            elif column_name == 'dose_diff' or column_name == 'concurrent_MME_diff':
                cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
            elif column_name == 'age':
                cutoffs.append([n for n in range(20, 80) if n % 10 == 0])
            else:
                cutoffs.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):

            FULL_fold = FULL_splited[i]
            x = FULL_fold[feature_list]
            x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
            x_rest = FULL_fold[FULL_fold.columns.drop(feature_list)]

            new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
            print(new_data.shape)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)         
    
    else:
        print('Scenario cannot be identified')



    



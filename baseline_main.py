#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Baseline comparisons
"""

import time
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import utils.baseline_functions as base

def baseline_main(year:int,
                  first:bool,
                  upto180:bool,
                  model:str,
                  class_weight=None, # unbalanced
                  case:str='Explore',
                  setting_tag:str='',
                  datadir:str='/export/storage_cures/CURES/Processed/',
                  exportdir:str='/export/storage_cures/CURES/Results/',):

    print(f"Baseline test with model {model}\n")

    # ===========================================================================================
    
    if first:
        file_suffix = "_FIRST_INPUT"
    elif upto180:
        file_suffix = "_UPTOFIRST_INPUT"
    else:
        file_suffix = "_INPUT"

    file_path = f'{datadir}FULL_OPIOID_{year}{file_suffix}.csv'

    FULL = pd.read_csv(file_path, delimiter=",", dtype={'concurrent_MME': float, 
                                                        'concurrent_methadone_MME': float,
                                                        'num_prescribers_past180': int,
                                                        'num_pharmacies_past180': int,
                                                        'concurrent_benzo': int,
                                                        'consecutive_days': int})#.fillna(0)
    
    quartile_list = ['patient_HPIQuartile', 'prescriber_HPIQuartile', 'pharmacy_HPIQuartile',
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
                     'family_poverty_pct_quartile', 'unemployment_pct_quartile']
    FULL = FULL.dropna(subset=quartile_list) # drop NA rows

    y = FULL['long_term_180'].values
    
    # ===========================================================================================

    if first:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"

    data_frames = []
    for i in range(20):
        file_path = f'{datadir}/Stumps/FULL_{year}_{case}{file_suffix}{i}.csv'
        df = pd.read_csv(file_path, delimiter=",")
        data_frames.append(df)

    FULL_STUMPS = pd.concat(data_frames, ignore_index=True)
    print(f'Finished importing STUMPS, with shape {FULL_STUMPS.shape}\n')

    # ===========================================================================================

    base_feature_list = ['concurrent_MME', 
                         'num_prescribers_past180',
                         'num_pharmacies_past180', 'concurrent_benzo',
                         'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO',
                         'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                         'num_prior_prescriptions', 
                         'diff_MME', # 'diff_quantity', 
                         'diff_days',
                         'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment']
    
    drug_payment = [['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'], ['Medicaid', 'Medicare', 'CashCredit']]

    spatial_features_set = ['patient_HPIQuartile', 'patient_zip_avg_days', 'patient_zip_avg_MME', 
                            # 'patient_zip_yr_num_prescriptions_quartile', 'patient_zip_yr_num_patients_quartile', 
                            'patient_zip_yr_num_prescriptions_per_pop_quartile', 'patient_zip_yr_num_patients_per_pop_quartile',
                            'prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 'prescriber_yr_num_pharmacies_quartile', 
                            'prescriber_yr_avg_MME_quartile', 'prescriber_yr_avg_days_quartile',
                            'pharmacy_yr_num_prescriptions_quartile', 'pharmacy_yr_num_patients_quartile', 'pharmacy_yr_num_prescribers_quartile',
                            'pharmacy_yr_avg_MME_quartile', 'pharmacy_yr_avg_days_quartile',
                            'age', 'patient_gender', 'zip_pop_density_quartile', 'median_household_income_quartile', 'family_poverty_pct_quartile', 'unemployment_pct_quartile']
    
    features_to_keep = base_feature_list + spatial_features_set

    filtered_columns = [col for col in FULL_STUMPS.columns if any(col.startswith(feature) for feature in features_to_keep)]
    
    columns_to_drop = ['CommercialIns', 'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation', 'age20']
    columns_to_drop += [f'num_pharmacies_past180{i}' for i in range(4, 11)]
    columns_to_drop += [f'num_prescribers_past180{i}' for i in range(4, 11)]
    filtered_columns = [feature for feature in filtered_columns if feature not in columns_to_drop]


    # TODO: DO WE WANT THE INPUT FOR BASELINE MODELS ALSO STUMPS?

    FULL_STUMPS = FULL_STUMPS[filtered_columns]

    FULL_STUMPS['(Intercept)'] = 1
    intercept = FULL_STUMPS.pop('(Intercept)')
    FULL_STUMPS.insert(0, '(Intercept)', intercept)
    x = FULL_STUMPS

    print(FULL_STUMPS.columns.tolist()) # so that we know the order of features

    # ===========================================================================================
    
    start = time.time()
    
    # results = pd.DataFrame()

    if model == 'DecisionTree':
        depth = [5,7]
        min_samples = [10,20]
        impurity = [0.001,0.01,0.1]
        summary = base.DecisionTree(X=x, Y=y, depth=depth, min_samples=min_samples, impurity=impurity, class_weight=class_weight, seed=42)

    if model == 'L2':
        c = [1e-15, 1e-10, 1e-5, 1e-1, 10]
        summary = base.Logistic(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

    if model == 'L1':
        c = [1e-15, 1e-10, 1e-5, 1e-1, 10]
        summary = base.Lasso(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

    if model == 'SVM':
        # c = np.linspace(1e-6,1e-2,5).tolist()
        c = [1e-15, 1e-10, 1e-5, 1e-1]
        summary = base.LinearSVM(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

    if model == 'RandomForest':
        depth = [3,4,5,6]
        n_estimators = [50,100,200]
        impurity = [0.001,0.01]
        summary = base.RF(X=x, Y=y, depth=depth, estimators=n_estimators, impurity=impurity, class_weight=class_weight, seed=42)

    if model == 'XGB':
        depth = [4,5,6]
        n_estimators =  [50,100,150]
        gamma = [5,10]
        child_weight = [5,10]
        summary = base.XGB(X=x, Y=y, depth=depth, estimators=n_estimators, gamma=gamma, child_weight=child_weight, class_weight=class_weight, seed=42)
    
    if model == 'DNN':
        base.DNN(X=x, Y=y)
        
    # balanced = {"Accuracy": str(round(np.mean(summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(summary['holdout_test_accuracy']), 4)) + ")",
    #     "Recall": str(round(np.mean(summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(summary['holdout_test_recall']), 4)) + ")",
    #     "Precision": str(round(np.mean(summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(summary['holdout_test_precision']), 4)) + ")",
    #     "ROC AUC": str(round(np.mean(summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(summary['holdout_test_roc_auc']), 4)) + ")",
    #     "PR AUC": str(round(np.mean(summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(summary['holdout_test_pr_auc']), 4)) + ")",
    #     "Brier": str(round(np.mean(summary['holdout_test_brier']), 4)) + " (" + str(round(np.std(summary['holdout_test_brier']), 4)) + ")",
    #     "F2": str(round(np.mean(summary['holdout_test_f2']), 4)) + " (" + str(round(np.std(summary['holdout_test_f2']), 4)) + ")",
    #     "Calibration Error": str(round(np.mean(summary['holdout_calibration_error']), 4)) + " (" + str(round(np.std(summary['holdout_calibration_error']), 4)) + ")"}
    # balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=[model])

    # results = pd.concat([results, balanced], axis=1)

    # results = results.T
    # results.to_csv(f"{exportdir}Result/baseline_results{setting_tag}.csv")

    end = time.time()
    print(str(round(end - start,1)) + ' seconds')
    
    
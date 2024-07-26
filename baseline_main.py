#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Baseline comparisons
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import utils.baseline_functions as base
from utils.model_selection import compute_calibration
from risk_test import compute_fairness


def baseline_main(year:int,
                  first:bool,
                  upto180:bool,
                  model:str,
                  setting_tag:str):
    
    best_model, filtered_columns = baseline_train(year, first, upto180, model, setting_tag=setting_tag)
    baseline_test(best_model, filtered_columns, year, first, upto180, model, setting_tag=setting_tag)
    
    return


def baseline_train(year:int,
                   first:bool,
                   upto180:bool,
                   model:str,
                   class_weight=None, # unbalanced
                   case:str='Explore',
                   setting_tag:str='',
                   output_columns:bool=False,
                   sample:bool=False,
                   datadir:str='/export/storage_cures/CURES/Processed/',
                   exportdir:str='/export/storage_cures/CURES/Results/'):

    print(f"Baseline train with model {model}\n")

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

    FULL_STUMPS = FULL_STUMPS[filtered_columns]

    FULL_STUMPS['(Intercept)'] = 1
    intercept = FULL_STUMPS.pop('(Intercept)')
    FULL_STUMPS.insert(0, '(Intercept)', intercept)
    x = FULL_STUMPS

    if output_columns: print(FULL_STUMPS.columns.tolist()) # so that we know the order of features

    if sample:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=10000/len(y), random_state=42)
        for train_index, test_index in sss.split(x, y):
            x_sample, y_sample = x.iloc[test_index], y[test_index] # x: dataframe y: np array

        x = x_sample
        y = y_sample

    # ===========================================================================================
    
    start = time.time()
    
    # results = pd.DataFrame()

    if model == 'DecisionTree':
        best_model = base.DecisionTree(X=x, 
                                       Y=y, 
                                       depth=[4, 6], 
                                       min_samples=[10, 50], 
                                       impurity=[0.001, 0.01, 0.1], 
                                       class_weight=class_weight, 
                                       seed=42)

    elif model == 'L2':
        best_model = base.Logistic(X=x, 
                                   Y=y, 
                                   C=[1e-15, 1e-10, 1e-5, 1e-1, 10], 
                                   class_weight=class_weight, 
                                   seed=42)

    elif model == 'L1':
        best_model = base.Lasso(X=x, 
                                Y=y, 
                                C=[1e-15, 1e-10, 1e-5, 1e-1, 10], 
                                class_weight=class_weight, 
                                seed=42)

    elif model == 'SVM':
        best_model = base.LinearSVM(X=x, 
                                    Y=y, 
                                    C=[1e-15, 1e-5, 1e-1], 
                                    class_weight=class_weight, 
                                    seed=42)

    elif model == 'RandomForest':
        best_model = base.RF(X=x, 
                             Y=y, 
                             depth=[4, 6], 
                             estimators=[50, 150], 
                             impurity=[0.001, 0.01], 
                             class_weight=class_weight, 
                             seed=42)

    elif model == 'XGB':
        best_model = base.XGB(X=x, 
                              Y=y, 
                              depth=[4, 6], 
                              estimators=[50, 150], 
                              gamma=[5, 10], 
                              child_weight=[5, 10], 
                              class_weight=class_weight, 
                              seed=42)

    elif model == 'NN':
        best_model = base.NeuralNetwork(X=x, 
                                        Y=y, 
                                        alpha=[0.0001, 0.001], 
                                        batch_size=[32], 
                                        learning_rate_init=[0.001], 
                                        seed=42)

    end = time.time()
    print(str(round(end - start,1)) + ' seconds')

    return best_model, filtered_columns


# ================================================================================
# ================================================================================
# ================================================================================


def baseline_test(best_model,
                  filtered_columns,
                  year:int,
                  first:bool,
                  upto180:bool,
                  model:str,
                  class_weight=None, # unbalanced
                  case:str='Explore',
                  setting_tag:str='',
                  output_columns:bool=False,
                  sample:bool=False,
                  datadir:str='/export/storage_cures/CURES/Processed/',
                  exportdir:str='/export/storage_cures/CURES/Results/'):

    ### OUT-SAMPLE TEST ###

    print(f"Baseline test with model {model}\n")

    # ===========================================================================================
    
    if first:
        file_suffix = "_FIRST_INPUT"
    elif upto180:
        file_suffix = "_UPTOFIRST_INPUT"
    else:
        file_suffix = "_INPUT"

    test_file_path = f'{datadir}FULL_OPIOID_{year+1}{file_suffix}.csv'

    FULL_TEST = pd.read_csv(test_file_path, delimiter=",", dtype={'concurrent_MME': float, 
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
    
    FULL_TEST = FULL_TEST.dropna(subset=quartile_list) # drop NA rows
    y_test = FULL_TEST['long_term_180'].values

    # ================================================================================

    if first:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"

    data_frames = []
    for i in range(20):
        test_file_path = f'{datadir}/Stumps/FULL_{year+1}_{case}{file_suffix}{i}.csv'
        df = pd.read_csv(test_file_path, delimiter=",")
        data_frames.append(df)

    FULL_STUMPS_TEST = pd.concat(data_frames, ignore_index=True)
    print(f'Finished importing TEST STUMPS, with shape {FULL_STUMPS_TEST.shape}\n')
    
    FULL_STUMPS_TEST = FULL_STUMPS_TEST[filtered_columns]

    FULL_STUMPS_TEST['(Intercept)'] = 1
    intercept = FULL_STUMPS_TEST.pop('(Intercept)')
    FULL_STUMPS_TEST.insert(0, '(Intercept)', intercept)
    x_test = FULL_STUMPS_TEST

    # ================================================================================

    prob = best_model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5)

    test_results = {'test_accuracy': accuracy_score(y_test, pred),
            'test_recall': recall_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            'test_roc_auc': roc_auc_score(y_test, prob),
            'test_pr_auc': average_precision_score(y_test, prob),
            "test_calibration_error": compute_calibration(y_test, prob, pred)}

    print(test_results)

    compute_fairness(x_test, y_test, prob, pred, setting_tag)

    # get the test results by n th prescriptions
    FULL_TEST['num_prescriptions'] = FULL_TEST['num_prior_prescriptions'] + 1
    FULL_TEST['prod'], FULL_TEST['pred'] = prob, pred
    test_results_by_prescriptions = FULL_TEST.groupby('num_prescriptions').apply(lambda x: {'test_accuracy': accuracy_score(x['long_term_180'], x['pred']),
                                                                                           'test_recall': recall_score(x['long_term_180'], x['pred']),
                                                                                           'test_precision': precision_score(x['long_term_180'], x['pred']),
                                                                                           'test_roc_auc': roc_auc_score(x['long_term_180'], x['prob']),
                                                                                           'test_pr_auc': average_precision_score(x['long_term_180'], x['prob']),
                                                                                           'test_calibration_error': compute_calibration(x['long_term_180'], x['prob'], x['pred'])}).to_dict()
    print(test_results_by_prescriptions)

    return
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Baseline comparisons
"""

import sys 
import time
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
import utils.baseline_functions as base
from utils.model_selection import compute_calibration, compute_fairness, compute_patient, compute_nth_presc, compute_MME_presc, print_results
from multiprocessing import Pool
from scipy import special
import pickle
import re

model = sys.argv[1]
first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])
sample = any(['sample' in arg for arg in sys.argv])
median =  any(['median' in arg for arg in sys.argv])
bracket = any(['bracket' in arg for arg in sys.argv])

setting_tag = f'_{model}'
setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_sample" if sample else ""
setting_tag += f"_median" if median else ""
setting_tag += f"_bracket" if bracket else ""


def baseline_main(model:str,
                  setting_tag:str,
                  first:bool=False,
                  upto180:bool=False,
                  sample:bool=False,
                  median:bool=False,
                  bracket:bool=False,
                  year:int=2018):

    best_model, filtered_columns = baseline_train(year, first, upto180, model, setting_tag=setting_tag, sample=sample, median=median, bracket=bracket)
    baseline_test(best_model, filtered_columns, year, first, upto180, model, setting_tag=setting_tag, sample=sample, median=median, bracket=bracket)
    
    return


def baseline_train(year:int,
                   first:bool,
                   upto180:bool,
                   model:str,
                   class_weight=None, # None/"balanced"
                   case:str='Explore',
                   setting_tag:str='',
                   output_columns:bool=False,
                   sample:bool=False,
                   median:bool=False,
                   bracket:bool=False,
                   datadir:str='/export/storage_cures/CURES/Processed/',
                   exportdir:str='/export/storage_cures/CURES/Results/'):

    print("="*60, f"\nBaseline train with model {model}\n")

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

    if median: 
        file_suffix += 'median_'
        print("WARNING: Working on median stumps")

    data_frames = []
    for i in range(20):
        file_path = f'{datadir}Stumps/FULL_{year}_{case}{file_suffix}{i}.csv'
        df = pd.read_csv(file_path, delimiter=",")
        data_frames.append(df)

    FULL_STUMPS = pd.concat(data_frames, ignore_index=True)

    # ===========================================================================================

    base_feature_list = ['concurrent_MME', 'days_supply',
                         'num_prescribers_past180',
                         'num_pharmacies_past180', 'concurrent_benzo',
                         'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO', 'long_acting',
                         'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                         'num_prior_prescriptions', 
                         'diff_MME', 
                         'diff_days',
                         'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment']

    spatial_features_set = ['patient_zip_avg_days', 'patient_zip_avg_MME', # 'patient_HPIQuartile', 
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
    columns_to_drop += [f'num_pharmacies_past180{i}' for i in range(7, 11)]
    columns_to_drop += [f'num_prescribers_past180{i}' for i in range(7, 11)]
    filtered_columns = [feature for feature in filtered_columns if feature not in columns_to_drop]

    print(f"Filtered columns: {filtered_columns}")
    FULL_STUMPS = FULL_STUMPS[filtered_columns]
    
    if bracket: 
        print("WARNING: Working on bracketed stumps")
        FULL_STUMPS = reconstruct_stumps(FULL_STUMPS)

    FULL_STUMPS['(Intercept)'] = 1
    intercept = FULL_STUMPS.pop('(Intercept)')
    FULL_STUMPS.insert(0, '(Intercept)', intercept)
    x = FULL_STUMPS


    if output_columns: print(FULL_STUMPS.columns.tolist())
    if sample:
        FULL_STUMPS_sample, _, FULL_sample, _ = train_test_split(FULL_STUMPS, FULL, test_size=0.8, random_state=42)
        x_sample = FULL_STUMPS_sample
        y_sample = FULL_sample['long_term_180'].values
        print(f'WARNING: Training on sample data now of {x_sample.shape[0]} observations')
        x, y = x_sample, y_sample

    # ===========================================================================================
    
    start = time.time()
    
    if model == 'DecisionTree':
        best_model, prob, pred = base.DecisionTree(X=x, 
                                       Y=y, 
                                    #    depth=[5, 10], 
                                    #    min_samples=[5, 10], 
                                    #    impurity=[0.0001], 
                                       depth=[5, 10], 
                                       min_samples=[5, 10], 
                                       impurity=[0.001, 0.01], 
                                       class_weight=class_weight, 
                                       seed=42)

    elif model == 'L2':
        best_model, prob, pred = base.Logistic(X=x, 
                                   Y=y, 
                                   # C=[1e-15, 1e-10, 1e-5, 1e-1, 10], 
                                   C=[1e-5, 1e-4], 
                                   class_weight=class_weight, 
                                   seed=42)

    elif model == 'L1':
        best_model, prob, pred = base.Lasso(X=x, 
                                Y=y, 
                                # smaller values specify stronger regularization.
                                C=[1e-5, 1e-4], 
                                class_weight=class_weight, 
                                seed=42)

    elif model == 'LinearSVM':
        best_model, prob, pred = base.LinearSVM(X=x, 
                                    Y=y, 
                                    C=[1e-5, 1e-4, 1e-3], 
                                    class_weight=class_weight, 
                                    seed=42)

    elif model == 'RandomForest':
        best_model, prob, pred = base.RF(X=x, 
                             Y=y, 
                             depth=[5, 10], 
                             estimators=[10, 50, 100], # number of trees
                             impurity=[0.001, 0.005, 0.01], 
                             class_weight=class_weight, 
                             seed=42)

    elif model == 'XGB':
        best_model, prob, pred = base.XGB(X=x, 
                              Y=y, 
                              depth=[5, 10], 
                              estimators=[20, 50], 
                              gamma=[5, 10], 
                              child_weight=[5, 10], 
                              class_weight=class_weight, 
                              seed=42)

    elif model == 'NN':
        best_model, prob, pred = base.NeuralNetwork(X=x, 
                                        Y=y, 
                                        alpha=[0.0001, 0.01], 
                                        batch_size=[32], 
                                        learning_rate_init=[0.001], 
                                        seed=42)

    end = time.time()
    print(f'Finished training and fitting {model}: {str(round((end - start)/3600, 2))} hours')

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
                  median:bool=False,
                  bracket:bool=False,
                  fairness:bool=True,
                  patient:bool=True,
                  n_presc:bool=True,
                  MME_results:bool=True,
                  export_files:bool=True,
                  datadir:str='/export/storage_cures/CURES/Processed/',
                  exportdir:str='/export/storage_cures/CURES/Results/'):

    ### OUT-SAMPLE TEST ###
    print("="*60, f"\nBaseline test with model {model}\n")

    # ===========================================================================================
    
    if first:
        file_suffix = "_FIRST_INPUT"
    elif upto180:
        file_suffix = "_UPTOFIRST_INPUT"
    else:
        file_suffix = "_INPUT"

    test_file_path = f'{datadir}FULL_OPIOID_{year+1}{file_suffix}.csv'
    print(f"The test file path is {test_file_path}\n")


    FULL_TEST = pd.read_csv(test_file_path, delimiter=",", dtype={'concurrent_MME': float, 
                                                                  'concurrent_methadone_MME': float,
                                                                  'num_prescribers_past180': int,
                                                                  'num_pharmacies_past180': int,
                                                                  'concurrent_benzo': int,
                                                                  'consecutive_days': int})#.fillna(0)
    
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

    if median: 
        file_suffix += 'median_'
        print("WARNING: Working on median stumps")

    data_frames = []
    for i in range(20):
        test_file_path = f'{datadir}Stumps/FULL_{year+1}_{case}{file_suffix}{i}.csv'
        df = pd.read_csv(test_file_path, delimiter=",")
        data_frames.append(df)

    FULL_STUMPS_TEST = pd.concat(data_frames, ignore_index=True)

    FULL_STUMPS_TEST = FULL_STUMPS_TEST[filtered_columns]
    if bracket: 
        print("WARNING: Working on bracketed stumps")
        FULL_STUMPS_TEST = reconstruct_stumps(FULL_STUMPS_TEST)

    FULL_STUMPS_TEST['(Intercept)'] = 1
    intercept = FULL_STUMPS_TEST.pop('(Intercept)')
    FULL_STUMPS_TEST.insert(0, '(Intercept)', intercept)
    x_test = FULL_STUMPS_TEST

    if sample:
        FULL_STUMPS_TEST_sample, _, FULL_TEST_sample, _ = train_test_split(FULL_STUMPS_TEST, FULL_TEST, test_size=0.8, random_state=42)
        x_sample = FULL_STUMPS_TEST_sample
        y_sample = FULL_TEST_sample['long_term_180'].values
        print(f'WARNING: Testing on sample data now of {x_sample.shape[0]} observations')
        x_test, y_test = x_sample, y_sample

    # ================================================================================

    if model == "NN":
        scaler = StandardScaler()
        x_test = scaler.fit_transform(x_test)

    prob = best_model.predict_proba(x_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for binary classification: {optimal_threshold}")
    pred = (prob >= optimal_threshold)

    start = time.time()
    ece = round(compute_calibration(y_test, prob, pred), 3)
    test_results = {'test_accuracy': round(accuracy_score(y_test, pred), 3),
                    'test_roc_auc': round(roc_auc_score(y_test, prob), 3),
                    'test_pr_auc': round(average_precision_score(y_test, prob), 3),
                    'test_calibration_error': ece}
    print_results(test_results)

    if sample: FULL_TEST = FULL_TEST_sample
    FULL_TEST['Prob'], FULL_TEST['Pred'] = prob, pred
    FULL_TEST['long_term_180'] = FULL_TEST['long_term_180'].values
    
    if fairness: compute_fairness(FULL_TEST, y_test, prob, pred, optimal_threshold, setting_tag)
    if patient: proportions = compute_patient(FULL_TEST, setting_tag)    
    if n_presc: compute_nth_presc(FULL_TEST)
    if MME_results: test_results_by_MME = compute_MME_presc(FULL_TEST)

    if export_files: export_results(model, y_test, prob, ece, median, bracket, proportions, test_results_by_MME)

    print(f'\nFinished computing and exporting results: {str(round((time.time() - start)/3600, 2))}hours\n')

    return


def export_results(model, y_test, prob, ece, median, bracket, proportions, test_results_by_MME):

    ### ROC
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    roc_info = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}   
    filename = f'../output/baseline/{model}_roc_test_info{"_median" if median else ""}{"_bracket" if bracket else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(roc_info, f)
    print(f"ROC information for {model} saved to {filename}")

    ### Calibration
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=20)
    calibration_info = {"prob_true": prob_true, "prob_pred": prob_pred, "ece": ece}
    filename = f'../output/baseline/{model}_calibration_test_info{"_median" if median else ""}{"_bracket" if bracket else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(calibration_info, f)
    print(f"Calibration information for {model} saved to {filename}")
    
    ### Proportion
    proportion_info = {"month": [], "proportion": []}
    for month, proportion in proportions.items():
        proportion_info["month"].append(month)
        proportion_info["proportion"].append(proportion)
    filename = f'../output/baseline/{model}_proportions_test_info{"_median" if median else ""}{"_bracket" if bracket else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(proportion_info, f)
    print(f"Proportions information for {model} saved to {filename}")

    ### Recall by MME bins
    recall_by_MME_info = {"MME": [], "recall": [], "pos_ratio": []}
    for MME_bin, results in test_results_by_MME.items():
        recall_by_MME_info["MME"].append(MME_bin)
        recall_by_MME_info["recall"].append(results['test_recall'])
        recall_by_MME_info["pos_ratio"].append(results['correctly_predicted_positives_ratio'])
    filename = f'../output/baseline/{model}_recallMME_test_info{"_median" if median else ""}{"_bracket" if bracket else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(recall_by_MME_info, f)
    print(f"Recall by MME information for riskSLIM saved to {filename}")

    return


def predict_batch(args):
    model, x_batch = args
    return model.predict_proba(x_batch)[:, 1]



def reconstruct_stumps(FULL_STUMPS):

    # Copy the original DataFrame
    FULL_RECONSTRUCTED = FULL_STUMPS.copy()

    # 1. Reconstruct concurrent_MME brackets
    mme_thresholds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300]
    FULL_RECONSTRUCTED['concurrent_MME_0_10'] = (FULL_STUMPS['concurrent_MME10'] == 0).astype(int)
    cols_to_drop = ['concurrent_MME10']

    for i in range(len(mme_thresholds) - 1):
        lower = mme_thresholds[i]
        upper = mme_thresholds[i + 1]
        FULL_RECONSTRUCTED[f'concurrent_MME_{lower}_{upper}'] = (
            (FULL_STUMPS[f'concurrent_MME{lower}'] == 1) & 
            (FULL_STUMPS[f'concurrent_MME{upper}'] == 0)
        ).astype(int)
        cols_to_drop.append(f'concurrent_MME{lower}')

    FULL_RECONSTRUCTED[f'concurrent_MME_{mme_thresholds[-1]}_inf'] = (
        FULL_STUMPS[f'concurrent_MME{mme_thresholds[-1]}'] == 1
    ).astype(int)
    cols_to_drop.append(f'concurrent_MME{mme_thresholds[-1]}')


    # 2. Reconstruct age brackets similarly
    age_thresholds = [30, 40, 50, 60, 70, 80]
    FULL_RECONSTRUCTED['age_0_30'] = (FULL_STUMPS['age30'] == 0).astype(int)
    cols_to_drop.append('age30')

    for i in range(len(age_thresholds) - 1):
        lower = age_thresholds[i]
        upper = age_thresholds[i + 1]
        FULL_RECONSTRUCTED[f'age_{lower}_{upper}'] = (
            (FULL_STUMPS[f'age{lower}'] == 1) & 
            (FULL_STUMPS[f'age{upper}'] == 0)
        ).astype(int)
        cols_to_drop.append(f'age{lower}')

    FULL_RECONSTRUCTED[f'age_{age_thresholds[-1]}_inf'] = (
        FULL_STUMPS[f'age{age_thresholds[-1]}'] == 1
    ).astype(int)
    cols_to_drop.append(f'age{age_thresholds[-1]}')

    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)

    # 3. Reconstruct exact equality indicators for num_prescribers_past180
    prescriber_thresholds = [2, 3, 4, 5]
    for val in prescriber_thresholds:
        col_name = f'num_prescribers_past180{val}'
        if col_name in FULL_STUMPS.columns:
            FULL_RECONSTRUCTED[f'num_prescribers_past180_eq_{val}'] = (
                (FULL_STUMPS[col_name] == 1) & 
                (~FULL_STUMPS.get(f'num_prescribers_past180{val + 1}', pd.Series(False)).astype(bool))
            ).astype(int)

    FULL_RECONSTRUCTED['num_prescribers_past180_eq_1'] = 1 - FULL_RECONSTRUCTED['num_prescribers_past1802']
    FULL_RECONSTRUCTED.rename(columns={'num_prescribers_past1806': 'num_prescribers_past180_6_inf'}, inplace=True)

    cols_to_drop = [f'num_prescribers_past180{val}' for val in prescriber_thresholds]
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)

    # 4. Reconstruct exact equality indicators for num_pharmacies_past180
    prescriber_thresholds = [2, 3, 4, 5]
    for val in prescriber_thresholds:
        col_name = f'num_pharmacies_past180{val}'
        if col_name in FULL_STUMPS.columns:
            FULL_RECONSTRUCTED[f'num_pharmacies_past180_eq_{val}'] = (
                (FULL_STUMPS[col_name] == 1) & 
                (~FULL_STUMPS.get(f'num_pharmacies_past180{val + 1}', pd.Series(False)).astype(bool))
            ).astype(int)

    FULL_RECONSTRUCTED['num_pharmacies_past180_eq_1'] = 1 - FULL_RECONSTRUCTED['num_pharmacies_past1802']
    FULL_RECONSTRUCTED.rename(columns={'num_pharmacies_past1806': 'num_pharmacies_past180_6_inf'}, inplace=True)

    cols_to_drop = [f'num_pharmacies_past180{val}' for val in prescriber_thresholds]
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)    


    # 5. Reconstruct days supply brackets
    days_supply_thresholds = [3, 5, 7, 10, 14, 21, 30]
    FULL_RECONSTRUCTED['days_supply_0_3'] = (FULL_STUMPS['days_supply3'] == 0).astype(int)
    cols_to_drop = ['days_supply3']
    for i in range(len(days_supply_thresholds) - 1):
        lower = days_supply_thresholds[i]
        upper = days_supply_thresholds[i + 1]
        FULL_RECONSTRUCTED[f'days_supply_{lower}_{upper}'] = (
            (FULL_STUMPS[f'days_supply{lower}'] == 1) & 
            (FULL_STUMPS[f'days_supply{upper}'] == 0)
        ).astype(int)
        cols_to_drop.append(f'days_supply{lower}')
    
    FULL_RECONSTRUCTED.rename(columns={'days_supply30': 'days_supply_30_inf'}, inplace=True)
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)

    # 6. Remove quartile columns ending with _0.0, keep only those ending with _1.0
    cols_to_keep = [col for col in FULL_RECONSTRUCTED.columns 
                    if not re.search(r'quartile_0\.0$', col)]
    FULL_RECONSTRUCTED = FULL_RECONSTRUCTED[cols_to_keep]
    
    print("Final Columns:", FULL_RECONSTRUCTED.columns.tolist())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(FULL_RECONSTRUCTED.head())

    return FULL_RECONSTRUCTED




if __name__ == "__main__":
    baseline_main(model, setting_tag, first, upto180, sample, median, bracket)

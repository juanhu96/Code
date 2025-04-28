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

model = sys.argv[1]
first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])
sample = any(['sample' in arg for arg in sys.argv])
median =  any(['median' in arg for arg in sys.argv])

setting_tag = f'_{model}'
setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_sample" if sample else ""
setting_tag += f"_median" if median else ""


def baseline_main(model:str,
                  setting_tag:str,
                  first:bool=False,
                  upto180:bool=False,
                  sample:bool=False,
                  median:bool=False,
                  year:int=2018):

    best_model, filtered_columns = baseline_train(year, first, upto180, model, setting_tag=setting_tag, sample=sample, median=median)
    baseline_test(best_model, filtered_columns, year, first, upto180, model, setting_tag=setting_tag, sample=sample, median=median)
    
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

    base_feature_list = ['concurrent_MME', 
                         'num_prescribers_past180',
                         'num_pharmacies_past180', 'concurrent_benzo',
                         'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO',
                         'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                         'num_prior_prescriptions', 
                         'diff_MME', # 'diff_quantity', 
                         'diff_days',
                         'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment']

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

    if export_files: export_results(model, y_test, prob, ece, median, proportions, test_results_by_MME)

    print(f'\nFinished computing and exporting results: {str(round((time.time() - start)/3600, 2))}hours\n')

    return


def export_results(model, y_test, prob, ece, median, proportions, test_results_by_MME):

    ### ROC
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    roc_info = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}   
    filename = f'../output/baseline/{model}_roc_test_info{"_median" if median else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(roc_info, f)
    print(f"ROC information for {model} saved to {filename}")

    ### Calibration
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=20)
    calibration_info = {"prob_true": prob_true, "prob_pred": prob_pred, "ece": ece}
    filename = f'../output/baseline/{model}_calibration_test_info{"_median" if median else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(calibration_info, f)
    print(f"Calibration information for {model} saved to {filename}")
    
    ### Proportion
    proportion_info = {"month": [], "proportion": []}
    for month, proportion in proportions.items():
        proportion_info["month"].append(month)
        proportion_info["proportion"].append(proportion)
    filename = f'../output/baseline/{model}_proportions_test_info{"_median" if median else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(proportion_info, f)
    print(f"Proportions information for {model} saved to {filename}")

    ### Recall by MME bins
    recall_by_MME_info = {"MME": [], "recall": [], "pos_ratio": []}
    for MME_bin, results in test_results_by_MME.items():
        recall_by_MME_info["MME"].append(MME_bin)
        recall_by_MME_info["recall"].append(results['test_recall'])
        recall_by_MME_info["pos_ratio"].append(results['correctly_predicted_positives_ratio'])
    filename = f'../output/baseline/{model}_recallMME_test_info{"_median" if median else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(recall_by_MME_info, f)
    print(f"Recall by MME information for riskSLIM saved to {filename}")

    return


def predict_batch(args):
    model, x_batch = args
    return model.predict_proba(x_batch)[:, 1]




if __name__ == "__main__":
    baseline_main(model, setting_tag, first, upto180, sample, median)

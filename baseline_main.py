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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
import statsmodels.api as sm
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
bracket = any(['bracket' in arg for arg in sys.argv])
naive = any(['naive' in arg for arg in sys.argv]) # train on full, text on first

setting_tag = f'_{model}'
setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_sample" if sample else ""
setting_tag += f"_bracket" if bracket else ""


def baseline_main(model:str,
                  setting_tag:str,
                  first:bool=False,
                  upto180:bool=False,
                  sample:bool=False,
                  bracket:bool=False,
                  naive:bool=False,
                  year:int=2018):

    best_model, filtered_columns = baseline_train(year, first, upto180, model, setting_tag=setting_tag, sample=sample, bracket=bracket)
    # Test on all data
    print(f'Testing on all data for {model} with setting {setting_tag}')
    baseline_test(
        best_model, filtered_columns, year, first, upto180, model,
        setting_tag=setting_tag, sample=sample, bracket=bracket
    )

    # Optionally test on naive data
    if naive:
        naive_tag = f"{setting_tag}_naive"
        print(f'Testing on naive data for {model} with setting {naive_tag}')
        baseline_test(
            best_model, filtered_columns, year, first, upto180, model,
            setting_tag=naive_tag, sample=sample, bracket=bracket, naive=True
        )

    return


def baseline_train(year:int,
                   first:bool,
                   upto180:bool,
                   model:str,
                   class_weight=None, # None/"balanced"
                   setting_tag:str='',
                   output_columns:bool=False,
                   sample:bool=False,
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
    FULL = drop_na_rows(FULL)
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
        file_path = f'{datadir}Stumps/FULL_{year}{file_suffix}{i}.csv'
        df = pd.read_csv(file_path, delimiter=",")
        data_frames.append(df)

    FULL_STUMPS = pd.concat(data_frames, ignore_index=True)

    # ===========================================================================================

    base_feature_list = ['concurrent_MME', 'daily_dose', 'days_supply', 
                         'num_prescribers_past180', 'num_pharmacies_past180', 
                         'concurrent_benzo',
                         'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO', 'long_acting',
                         'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit', 'Other',
                         'num_prior_prescriptions', 
                         'diff_MME', 'diff_days',
                         'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                         'patient_gender']

    spatial_features_set = ['prescriber_yr_num_prescriptions', 'prescriber_yr_num_patients', 'prescriber_yr_num_pharmacies', 
                            'prescriber_yr_avg_MME', 'prescriber_yr_avg_days',
                            'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 'pharmacy_yr_num_prescribers',
                            'pharmacy_yr_avg_MME', 'pharmacy_yr_avg_days',
                            'zip_pop_density', 'median_household_income', 'family_poverty_pct', 'unemployment_pct']
    
    features_to_keep = base_feature_list + spatial_features_set

    filtered_columns = [col for col in FULL_STUMPS.columns if any(col.startswith(feature) for feature in features_to_keep)]
    
    columns_to_drop = ['CommercialIns', 'MilitaryIns', 'WorkersComp', 'IndianNation']
    columns_to_drop += [f'days_supply{i}' for i in [14, 21]]
    columns_to_drop += [col for col in filtered_columns if col.startswith('age')]
    columns_to_drop += [col for col in filtered_columns if col.startswith('concurrent_MME')]
    columns_to_drop += [col for col in filtered_columns if 'above55' in col or 'above60' in col or 'above65' in col or 'above70' in col or 'above80' in col]
    # columns_to_drop += [f'num_pharmacies_past180{i}' for i in range(7, 11)]
    # columns_to_drop += [f'num_prescribers_past180{i}' for i in range(7, 11)]
    
    filtered_columns = [feature for feature in filtered_columns if feature not in columns_to_drop]

    print(f"Filtered columns: {filtered_columns}")
    FULL_STUMPS = FULL_STUMPS[filtered_columns]
    
    if bracket: 
        print("WARNING: Working on bracketed stumps:")
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
                                #    C=[1e-15, 1e-10, 1e-5, 1e-1, 10], 
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

    elif model == 'Logistic':
        x = x.drop(columns=['(Intercept)'])        
        best_model = sm.Logit(y, x).fit(disp=0)
        coefficients = best_model.params
        conf = best_model.conf_int()
        conf['OR lower'] = np.exp(conf[0])
        conf['OR upper'] = np.exp(conf[1])
        odds_ratios = np.exp(coefficients)
        p_values = best_model.pvalues

        results = pd.DataFrame({
            'Feature': x.columns,
            'Coefficient': coefficients,
            'Odds Ratio': odds_ratios,
            'CI Lower': conf['OR lower'],
            'CI Upper': conf['OR upper'],
            'p-value': p_values.values
        })

        custom_order = ['num_prescribers_past180_eq_2', 'num_prescribers_past180_eq_3', 'num_prescribers_past180_eq_4', 'num_prescribers_past180_eq_5', 'num_prescribers_past180_6_inf', 
                        'num_pharmacies_past180_eq_2', 'num_pharmacies_past180_eq_3', 'num_pharmacies_past180_eq_4', 'num_pharmacies_past180_eq_5', 'num_pharmacies_past180_6_inf', 
                        'concurrent_benzo1', 'patient_gender',
                        'days_supply_3_5', 'days_supply_5_7', 'days_supply_7_10', 'days_supply_10_inf',
                        'daily_dose_25_50', 'daily_dose_50_75', 'daily_dose_75_90', 'daily_dose_90_inf',
                        'long_acting', 'Codeine', 'Oxycodone', 'Morphine', 'HMFO', 'Medicaid', 'Medicare', 'CashCredit', 'Other', 
                        'num_prior_prescriptions1', 'diff_MME1', 'diff_days1', 'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',  
                        'prescriber_yr_num_prescriptions_above50', 'prescriber_yr_num_prescriptions_above75', 'prescriber_yr_num_patients_above50', 'prescriber_yr_num_patients_above75', 'prescriber_yr_num_pharmacies_above50', 'prescriber_yr_num_pharmacies_above75', 'prescriber_yr_avg_MME_above50', 'prescriber_yr_avg_MME_above75', 'prescriber_yr_avg_days_above50', 'prescriber_yr_avg_days_above75',
                        'pharmacy_yr_num_prescriptions_above50', 'pharmacy_yr_num_prescriptions_above75', 'pharmacy_yr_num_patients_above50', 'pharmacy_yr_num_patients_above75', 'pharmacy_yr_num_prescribers_above50', 'pharmacy_yr_num_prescribers_above75', 'pharmacy_yr_avg_MME_above50', 'pharmacy_yr_avg_MME_above75', 'pharmacy_yr_avg_days_above50', 'pharmacy_yr_avg_days_above75',
                        'zip_pop_density_above50', 'zip_pop_density_above75', 'median_household_income_above50', 'median_household_income_above75', 'family_poverty_pct_above50', 'family_poverty_pct_above75', 'unemployment_pct_above50', 'unemployment_pct_above75']

        results = results.set_index('Feature').loc[custom_order].reset_index()
        results.to_csv(f'{exportdir}LogisticRegression_unregularized.csv', index=False)
        

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
                  setting_tag:str='',
                  output_columns:bool=False,
                  sample:bool=False,
                  bracket:bool=False,
                  naive:bool=False,
                  fairness:bool=False,
                  patient:bool=True,
                  n_presc:bool=True,
                  MME_results:bool=True,
                  export_files:bool=True,
                  datadir:str='/export/storage_cures/CURES/Processed/',
                  exportdir:str='/export/storage_cures/CURES/Results/'):

    ### OUT-SAMPLE TEST ###
    print("="*60, f"\nBaseline test with model {model}\n")

    # ===========================================================================================
    
    if first or naive:
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
    FULL_TEST = drop_na_rows(FULL_TEST)
    y_test = FULL_TEST['long_term_180'].values

    # ================================================================================

    if first or naive:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"

    data_frames = []
    for i in range(20):
        test_file_path = f'{datadir}Stumps/FULL_{year+1}{file_suffix}{i}.csv'
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

    if model != 'Logistic':
        prob = best_model.predict_proba(x_test)[:, 1]
    else:
        x_test = x_test.drop(columns=['(Intercept)'])
        prob = best_model.predict(x_test)
        prob = np.array(prob)

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

    if export_files: export_results(model, y_test, prob, ece, bracket, naive, proportions, test_results_by_MME)

    print(f'\nFinished computing and exporting results: {str(round((time.time() - start)/3600, 2))}hours\n')

    return


def export_results(model, y_test, prob, ece, bracket, naive, proportions, test_results_by_MME):

    ### ROC
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    roc_info = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}   
    filename = f'../output/baseline/files/{model}_roc_test_info{"_bracket" if bracket else ""}{"_naive" if naive else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(roc_info, f)
    print(f"ROC information for {model} saved to {filename}")

    ### Calibration
    prob_true, prob_pred, observations = calibration_curve(y_test, prob, n_bins=20)
    calibration_info = {"prob_true": prob_true, "prob_pred": prob_pred, "ece": ece, "observations": observations}
    filename = f'../output/baseline/files/{model}_calibration_test_info{"_bracket" if bracket else ""}{"_naive" if naive else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(calibration_info, f)
    print(f"Calibration information for {model} saved to {filename}")
    
    ### Proportion
    proportion_info = {"month": [], "proportion": []}
    for month, proportion in proportions.items():
        proportion_info["month"].append(month)
        proportion_info["proportion"].append(proportion)
    filename = f'../output/baseline/files/{model}_proportions_test_info{"_bracket" if bracket else ""}{"_naive" if naive else ""}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(proportion_info, f)
    print(f"Proportions information for {model} saved to {filename}")

    ### Recall by MME bins
    # recall_by_MME_info = {"MME": [], "recall": [], "pos_ratio": []}
    # for MME_bin, results in test_results_by_MME.items():
    #     recall_by_MME_info["MME"].append(MME_bin)
    #     recall_by_MME_info["recall"].append(results['test_recall'])
    #     recall_by_MME_info["pos_ratio"].append(results['correctly_predicted_positives_ratio'])
    # filename = f'../output/baseline/{model}_recallMME_test_info{"_bracket" if bracket else ""}.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(recall_by_MME_info, f)
    # print(f"Recall by MME information for riskSLIM saved to {filename}")

    return


def predict_batch(args):
    model, x_batch = args
    return model.predict_proba(x_batch)[:, 1]



def reconstruct_stumps(FULL_STUMPS):

    # Copy the original DataFrame
    FULL_RECONSTRUCTED = FULL_STUMPS.copy()

    # Reconstruct concurrent_MME, daily_dose brackets
    mme_thresholds = [25, 50, 75, 90] # 200, 300
    cols_to_drop = ['daily_dose10']

    for i in range(len(mme_thresholds) - 1):
        lower = mme_thresholds[i]
        upper = mme_thresholds[i + 1]
        FULL_RECONSTRUCTED[f'daily_dose_{lower}_{upper}'] = ((FULL_STUMPS[f'daily_dose{lower}'] == 1) & (FULL_STUMPS[f'daily_dose{upper}'] == 0)).astype(int)
        cols_to_drop.append(f'daily_dose{lower}')

    FULL_RECONSTRUCTED[f'daily_dose_{mme_thresholds[-1]}_inf'] = (FULL_STUMPS[f'daily_dose{mme_thresholds[-1]}'] == 1).astype(int)
    cols_to_drop.append(f'daily_dose{mme_thresholds[-1]}')

    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)

    # Reconstruct exact equality indicators for num_prescribers_past180
    prescriber_thresholds = [2, 3, 4, 5]
    for val in prescriber_thresholds:
        col_name = f'num_prescribers_past180{val}'
        if col_name in FULL_STUMPS.columns:
            FULL_RECONSTRUCTED[f'num_prescribers_past180_eq_{val}'] = (
                (FULL_STUMPS[col_name] == 1) & 
                (~FULL_STUMPS.get(f'num_prescribers_past180{val + 1}', pd.Series(False)).astype(bool))
            ).astype(int)

    # FULL_RECONSTRUCTED['num_prescribers_past180_eq_1'] = 1 - FULL_RECONSTRUCTED['num_prescribers_past1802']
    FULL_RECONSTRUCTED.rename(columns={'num_prescribers_past1806': 'num_prescribers_past180_6_inf'}, inplace=True)

    cols_to_drop = [f'num_prescribers_past180{val}' for val in prescriber_thresholds]
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)

    # Reconstruct exact equality indicators for num_pharmacies_past180
    prescriber_thresholds = [2, 3, 4, 5]
    for val in prescriber_thresholds:
        col_name = f'num_pharmacies_past180{val}'
        if col_name in FULL_STUMPS.columns:
            FULL_RECONSTRUCTED[f'num_pharmacies_past180_eq_{val}'] = (
                (FULL_STUMPS[col_name] == 1) & 
                (~FULL_STUMPS.get(f'num_pharmacies_past180{val + 1}', pd.Series(False)).astype(bool))
            ).astype(int)

    FULL_RECONSTRUCTED.rename(columns={'num_pharmacies_past1806': 'num_pharmacies_past180_6_inf'}, inplace=True)
    cols_to_drop = [f'num_pharmacies_past180{val}' for val in prescriber_thresholds]
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)    


    # Reconstruct days supply brackets
    days_supply_thresholds = [3, 5, 7, 10] # 14, 21, 30
    cols_to_drop = ['days_supply3']
    for i in range(len(days_supply_thresholds) - 1):
        lower = days_supply_thresholds[i]
        upper = days_supply_thresholds[i + 1]
        FULL_RECONSTRUCTED[f'days_supply_{lower}_{upper}'] = (
            (FULL_STUMPS[f'days_supply{lower}'] == 1) & 
            (FULL_STUMPS[f'days_supply{upper}'] == 0)
        ).astype(int)
        cols_to_drop.append(f'days_supply{lower}')
    
    FULL_RECONSTRUCTED.rename(columns={'days_supply10': 'days_supply_10_inf'}, inplace=True)
    FULL_RECONSTRUCTED.drop(columns=[col for col in cols_to_drop if col in FULL_RECONSTRUCTED.columns], inplace=True)


    # Categorical columns
    FULL_RECONSTRUCTED.drop(columns=['Hydrocodone'], inplace=True)
    print("Final Columns:", FULL_RECONSTRUCTED.columns.tolist())

    return FULL_RECONSTRUCTED


def calibration_curve(y_true, y_prob, *, pos_label=None, normalize="deprecated", n_bins=5, strategy="uniform"):
    
    y_true = column_or_1d(y_true) 
    y_prob = column_or_1d(y_prob) 
    pos_label = _check_pos_label_consistency(pos_label, y_true) 

    # TODO(1.3): Remove normalize conditional block. 
    if normalize != "deprecated": 
        warnings.warn( 
            "The normalize argument is deprecated in v1.1 and will be removed in v1.3." 
            " Explicitly normalizing y_prob will reproduce this behavior, but it is" 
            " recommended that a proper probability is used (i.e. a classifier's" 
            " `predict_proba` positive class or `decision_function` output calibrated" 
            " with `CalibratedClassifierCV`).", 
            FutureWarning, 
        ) 
        if normalize:  # Normalize predicted values into interval [0, 1] 
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min()) 

    if y_prob.min() < 0 or y_prob.max() > 1: 
        raise ValueError("y_prob has values outside [0, 1].") 

    labels = np.unique(y_true) 
    if len(labels) > 2: 
        raise ValueError( 
            f"Only binary classification is supported. Provided labels {labels}." 
        ) 
    y_true = y_true == pos_label 

    if strategy == "quantile":  # Determine bin edges by distribution of data 
        quantiles = np.linspace(0, 1, n_bins + 1) 
        bins = np.percentile(y_prob, quantiles * 100) 
    elif strategy == "uniform": 
        bins = np.linspace(0.0, 1.0, n_bins + 1) 
    else: 
        raise ValueError( 
            "Invalid entry to 'strategy' input. Strategy " 
            "must be either 'quantile' or 'uniform'." 
        ) 

    binids = np.searchsorted(bins[1:-1], y_prob) 

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins)) 
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins)) 
    bin_total = np.bincount(binids, minlength=len(bins)) 

    nonzero = bin_total != 0 
    prob_true = bin_true[nonzero] / bin_total[nonzero] 
    prob_pred = bin_sums[nonzero] / bin_total[nonzero] 

    return prob_true, prob_pred, bin_total[nonzero] # new output bin_total[nonzero]


def _check_pos_label_consistency(pos_label, y_true):
    if pos_label is None:
        # Compute classes only if pos_label is not specified:
        classes = np.unique(y_true)
        if classes.dtype.kind in "OUS" or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        ):
            classes_repr = ", ".join([repr(c) for c in classes.tolist()])
            raise ValueError(
                f"y_true takes value in {{{classes_repr}}} and pos_label is not "
                "specified: either make y_true take value in {0, 1} or "
                "{-1, 1} or pass pos_label explicitly."
            )
        pos_label = 1

    return pos_label



def drop_na_rows(FULL):

    FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)

    feature_list = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo', 
                    'patient_gender', 'days_supply', 'daily_dose',
                    'num_prior_prescriptions', 'diff_MME', 'diff_days',
                    'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                    'patient_zip_yr_avg_days', 'patient_zip_yr_avg_MME']

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
    percentile_features = [col for col in FULL.columns if any(col.startswith(f"{prefix}_above") for prefix in percentile_list)]
    feature_list_extended = feature_list + percentile_features
    FULL = FULL.dropna(subset=feature_list_extended) # drop NA rows to match the stumps

    return FULL



if __name__ == "__main__":
    baseline_main(model, setting_tag, first, upto180, sample, bracket, naive)

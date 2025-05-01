#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023

Functions for training

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

import riskslim
import utils.RiskSLIM as slim
from riskslim.utils import print_model


def risk_train(scenario:str,
               year:int,
               max_points:int,
               max_features:int,
               c:float, 
               interceptub:int,
               interceptlb:int,
               weight:str, 
               first:bool,
               upto180:bool,
               median:bool,
               feature_set,
               cutoff_set,
               essential_num,
               nodrug:bool,
               noinsurance:bool,
               county_name:str,
               stretch:bool,
               setting_tag:str,
               case:str='Explore',
               outcome:str='long_term_180', 
               datadir:str='/export/storage_cures/CURES/Processed/',
               exportdir:str='/export/storage_cures/CURES/Results/'):

    '''
    Train a riskSLIM model
    
    
    Parameters
    ----------
    year: year of the training dataset
    case: name of the training case (e.g., LTOUR)
    scenario: single/CV
    c: has to be a list when doing CV
    feature_set: set of features
    max_points: maximum point allowed per feature
    max_features: maximum number of features
    outcome: outcome to predict
    name: index for filename (when running multiple trails)
    roc: export fpr, tpr for roc visualization (only for single)
    '''

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
                                                        'consecutive_days': int})
    print(f'{file_path} imported with shape {FULL.shape}')

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

    x, constraints = import_stumps(year, case, first, upto180, median, feature_set, cutoff_set, essential_num, nodrug, noinsurance)
    y = FULL[[outcome]].to_numpy().astype('int')    
    y[y==0]= -1

    if county_name is not None: 
        zip_county = pd.read_csv(f'{datadir}/../CA/zip_county.csv', delimiter=",")
        FULL = FULL.merge(zip_county, left_on='patient_zip', right_on='zip', how='inner')
        indices = FULL.index[FULL['county'] == county_name].tolist()
        FULL = FULL.iloc[indices]
        x = x.iloc[indices]
        y = y[indices]
        print(f"Subsetting to county {county_name} with {len(y)} prescriptions.")


    if scenario == 'single':
        
        cols = x.columns.tolist()
        outer_train_sample_weight = np.repeat(1, len(y))
        outer_train_x, outer_train_y = x.values, y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': 'long_term_180',
            'sample_weights': outer_train_sample_weight
        }   
        
        start = time.time()    
        model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                                   max_coefficient=max_points, 
                                                                   max_L0_value=max_features,
                                                                   c0_value=c, 
                                                                   max_offset=interceptub,
                                                                   min_offset=interceptlb,
                                                                   class_weight=weight,
                                                                   single_cutoff=constraints['single_cutoff'],
                                                                   two_cutoffs=constraints['two_cutoffs'],
                                                                   three_cutoffs=constraints['three_cutoffs'],
                                                                   essential_cutoffs=constraints['essential_cutoffs'],
                                                                   essential_num=essential_num)
        intercept_val, rho_names, rho_values, table = print_model(model_info['solution'], new_train_data)
        print(f"Finished riskSLIM in {round(time.time() - start, 1)} seconds\n")

        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob >= 0.5)
        export_results_single(FULL, outer_train_y, outer_train_pred, outer_train_prob, filename = f'{exportdir}/Results{setting_tag}.csv')
    
    elif scenario == 'CV':
        # risk_summary = slim.risk_cv_constrain(x, y, 
        #                                       max_coefficient=max_points, 
        #                                       max_L0_value=max_features,
        #                                       c=c, 
        #                                       class_weight=weight,
        #                                       single_cutoff=constraints['single_cutoff'],
        #                                       two_cutoffs=constraints['two_cutoffs'],
        #                                       three_cutoffs=constraints['three_cutoffs'],
        #                                       essential_cutoffs=constraints['essential_cutoffs'],
        #                                       essential_num=essential_num)

        # export_results_cv(risk_summary, filename = f'{exportdir}/Results{setting_tag}.csv')
        print('CV not implemented yet.')

    else:
        raise Exception("Scenario undefined.")

    return table



def import_stumps(year, case, first, upto180, median, feature_set, cutoff_set, essential_num, nodrug, noinsurance, datadir='/export/storage_cures/CURES/Processed/'):

    N = 20
    data_frames = []

    if first:
        file_suffix = "_FIRST_STUMPS_"
    elif upto180:
        file_suffix = "_UPTOFIRST_STUMPS_"
    else:
        file_suffix = "_STUMPS_"
    
    if median: file_suffix += 'median_'

    print(f'FULL_{year}_{case}{file_suffix}')
    
    for i in range(N):
        file_path = f'{datadir}/Stumps/FULL_{year}_{case}{file_suffix}{i}.csv'
        df = pd.read_csv(file_path, delimiter=",")
        data_frames.append(df)

    STUMPS = pd.concat(data_frames, ignore_index=True)
    print(f'Finished importing STUMPS, with shape {STUMPS.shape}\n')


    # LTOUR w/p avgDays
    base_feature_list = ['concurrent_MME', 
                         'num_prescribers_past180',
                         'num_pharmacies_past180', 'concurrent_benzo',
                         'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO', 'long_acting',
                         'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation', 'Medicare_Medicaid',
                         'num_prior_prescriptions', 
                         'diff_MME', # 'diff_quantity', 
                         'diff_days',
                         'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                         'age', 'patient_gender']
    
    if nodrug:
        base_feature_list = [feature for feature in base_feature_list if feature not in ['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO']]
        drug_payment = [['Medicaid', 'Medicare', 'CashCredit', 'Medicare_Medicaid']]
        feature_drug_payment = ['Medicaid', 'Medicare', 'CashCredit', 'Medicare_Medicaid']
    else:
        drug_payment = [['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'], ['Medicaid', 'Medicare', 'CashCredit']]
        feature_drug_payment = ['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO', 'Medicaid', 'Medicare', 'CashCredit']

    if noinsurance:
        base_feature_list = [feature for feature in base_feature_list if feature not in ['Medicare_Medicaid']]
        drug_payment = [['Medicaid', 'Medicare', 'CashCredit']]
        feature_drug_payment = ['Medicaid', 'Medicare', 'CashCredit']

    # ============================================================================================
    
    # (2) Spatial demographics, intensity of opioid use in the patient zip code (pt_zip)
    features_set_2 = ['patient_zip_avg_days', 'patient_zip_avg_MME',
                    'patient_zip_yr_num_prescriptions_per_pop_quartile', 'patient_zip_yr_num_patients_per_pop_quartile']

    # (3) Provider prescription history
    features_set_3 = ['prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 'prescriber_yr_num_pharmacies_quartile', 
                      'prescriber_yr_avg_MME_quartile', 'prescriber_yr_avg_days_quartile']
    
    # (4) Pharmacy prescription history
    features_set_4 = ['pharmacy_yr_num_prescriptions_quartile', 'pharmacy_yr_num_patients_quartile', 'pharmacy_yr_num_prescribers_quartile',
                      'pharmacy_yr_avg_MME_quartile', 'pharmacy_yr_avg_days_quartile']
    
    # (5) Other spatial demographics measured at pt_zip level
    features_set_5 = ['patient_HPIQuartile', 'zip_pop_density_quartile', 'median_household_income_quartile', 'family_poverty_pct_quartile', 'unemployment_pct_quartile']

    # (6) final set of features
    features_set_6 = ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile_1.0', 'concurrent_MME40', 'age30', 'pharmacy_yr_avg_days_quartile_1.0', 'ever_switch_payment']


    # FEATURES TO KEEP
    if feature_set == '1': features_to_keep = base_feature_list
    elif feature_set == '2': features_to_keep = base_feature_list + features_set_2
    elif feature_set == '3': features_to_keep = base_feature_list + features_set_3
    elif feature_set == '4': features_to_keep = base_feature_list + features_set_4 
    elif feature_set == '5': features_to_keep = base_feature_list + features_set_5
    elif feature_set == '6': features_to_keep = feature_set_6
    elif feature_set == 'nodemo': features_to_keep = base_feature_list + features_set_2 + features_set_3 + features_set_4
    else: 
        features_to_keep = base_feature_list + features_set_2 + features_set_3 + features_set_4 + features_set_5

    # ============================================================================================
    
    # EXCLUDE SOME FEATURES
    filtered_columns = [col for col in STUMPS.columns if any(col.startswith(feature) for feature in features_to_keep)]
    
    columns_to_drop = ['CommercialIns', 'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation', 'age20']
    columns_to_drop += [f'num_pharmacies_past180{i}' for i in range(4, 11)]
    columns_to_drop += [f'num_prescribers_past180{i}' for i in range(4, 11)]
    filtered_columns = [feature for feature in filtered_columns if feature not in columns_to_drop]

    STUMPS = STUMPS[filtered_columns]

    # NOTE: create a new column called 'Medicare_medicaid' which is the sum of 'Medicare' and 'Medicaid'
    if not noinsurance: STUMPS['Medicare_Medicaid'] = STUMPS['Medicare'] + STUMPS['Medicaid']

    STUMPS['(Intercept)'] = 1
    intercept = STUMPS.pop('(Intercept)')
    STUMPS.insert(0, '(Intercept)', intercept)
    x = STUMPS

    # ============================================================================================
    
    # CUTOFF CONDITIONS  
     
    if cutoff_set == 'atmostone_feature':

        ### ONE CUTOFF PER FEATURE

        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        two_cutoff = None

    if cutoff_set == 'atmostone_group':
        
        ### ONE CUTOFF PER FEATURE
        ### AT MOST ONE CUTOFF PER GROUP

        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        single_cutoff.extend(updated_drug_payment)

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        features_sets = [features_set_2, features_set_3, features_set_4, features_set_5]
        for features_set in features_sets:
            single_cutoff += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at most one cutoff per group

        two_cutoff = None

    elif cutoff_set == 'exactlyone_group':

        ### ONE CUTOFF PER FEATURE
        ### EXACTLY ONE CUTOFF PER GROUP
        
        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        single_cutoff.extend(updated_drug_payment)

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        features_sets = [features_set_2, features_set_3, features_set_4, features_set_5]
        for features_set in features_sets:
            single_cutoff += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at most one cutoff per group
            essential_cutoffs += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at least one cutoff per group

        # essential_cutoffs += [[col for col in x if any(col.startswith(feature) for feature in feature_drug_payment)]]
        essential_cutoffs.extend(updated_drug_payment) # at least one cutoff in both drug & payment

        two_cutoff = None


    elif cutoff_set == 'atmosttwo_group':
        
        ### ONE CUTOFF PER FEATURE
        ### AT MOST TWO CUTOFFS PER GROUP

        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        two_cutoff = []
        features_sets = [features_set_2, features_set_3, features_set_4, features_set_5]
        for features_set in features_sets:
            two_cutoff += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at most two cutoffs per group
        
        two_cutoff.extend(updated_drug_payment) # at most two cutoffs in drug & payment


    elif cutoff_set == 'exactlytwo_group':
        
        ### ONE CUTOFF PER FEATURE
        ### EXACTLY TWO CUTOFFS PER GROUP
        ### essential_num = 2

        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        single_cutoff.extend(updated_drug_payment)

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        two_cutoff = []
        features_sets = [features_set_2, features_set_3, features_set_4, features_set_5]
        for features_set in features_sets:
            two_cutoff += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at most two cutoffs per group
            essential_cutoffs += [[col for col in x if any(col.startswith(feature) for feature in features_set)]] # at least essential_num cutoff per group (essential_num = 2)

        two_cutoff.extend(updated_drug_payment) # at most two cutoffs in drug & payment
        essential_cutoffs += [[col for col in x if any(col.startswith(feature) for feature in feature_drug_payment)]] # at least one cutoff in drug/payment


    elif cutoff_set == 'atleast_total':
        
        ### ONE CUTOFF PER FEATURE
        ### FOUR CONDITIONS IN ADDITION TO CORE
        ### essential_num = n

        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff per feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        single_cutoff.extend(updated_drug_payment)

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition
        
        all_features = [feature for features_set in [features_set_2, features_set_4, features_set_5, features_set_6] for feature in features_set]
        essential_cutoffs += [[col for col in x if any(col.startswith(feature) for feature in all_features)]] # at least n from noncore set

        two_cutoff = None

    else:
        # DEFAULT SETTING FOR (1), (1) + (2), ..., (1) + (6)
        single_cutoff = [[col for col in x if col.startswith(feature)] for feature in features_to_keep] # one cutoff each feature
        updated_drug_payment = [[item for item in sublist if item in features_to_keep] for sublist in drug_payment] # filter drug/payments not in features_to_keep
        single_cutoff.extend(updated_drug_payment)

        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in features_to_keep)]] # at least one condition

        # print('Quick test forcing switch payment...')
        # essential_cutoffs += [['ever_switch_payment']]
        # single_cutoff = [[col for col in x if any(col.startswith(feature) for feature in sublist)] for sublist in features_to_keep] # this is for [['A', 'C'], ['B']]
        
        two_cutoff = None

    # ============================================================================================

    def print_cutoffs(cutoffs):
        if cutoffs is not None:
            for cutoff in cutoffs:
                print(cutoff)
            print('-' * 60)

    print('Single cutoffs:')
    print_cutoffs(single_cutoff)
    print('Two cutoffs')
    print_cutoffs(two_cutoff)
    print(f'Essential cutoffs with lower bound: {essential_num}')
    print_cutoffs(essential_cutoffs)

    constraints = {'single_cutoff': single_cutoff, 'two_cutoffs': two_cutoff, 'three_cutoffs': None, 'essential_cutoffs': essential_cutoffs}

    return x, constraints



def compute_roc(outer_train_prob, outer_train_y, setting_tag, exportdir='/mnt/phd/jihu/opioid/roc/'):

    # to make it more consistent we have to manually compute fpr, tpr
    FPR_list = []
    TPR_list = []
    TN_list, FP_list, FN_list, TP_list = [],[],[],[]
    thresholds = np.arange(0, 1.1, 0.1)
    
    for threshold in thresholds:

        y_pred = (outer_train_prob >= threshold)

        TN, FP, FN, TP = confusion_matrix(outer_train_y, y_pred).ravel()   
                
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        TP_list.append(TP)
                
        FPR = FP/(FP+TN)
        TPR = TP/(TP+FN)
        FPR_list.append(FPR)
        TPR_list.append(TPR)                
                
        TN_list = np.array(TN_list)
        FP_list = np.array(FP_list)
        FN_list = np.array(FN_list)
        TP_list = np.array(TP_list)
        FPR_list = np.array(FPR_list)
        TPR_list = np.array(TPR_list)

    np.savetxt(f'{exportdir}Result/tn_{setting_tag}.csv', TN_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/fp_{setting_tag}.csv', FP_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/fn_{setting_tag}.csv', FN_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/tp_{setting_tag}.csv', TP_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/fpr_{setting_tag}.csv', FPR_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/tpr_{setting_tag}.csv', TPR_list, delimiter = ",")
    np.savetxt(f'{exportdir}Result/thresholds_{setting_tag}.csv', thresholds, delimiter = ",")

    return 




def export_results_single(x, y, y_pred, y_prob, filename):

    calibration_error = compute_calibration(x, y, y_prob, y_pred)

    train_results = {"Train accuracy": str(round(accuracy_score(y, y_pred), 3)),
                    "Train Recall": str(round(recall_score(y, y_pred), 3)),
                    "Train Precision": str(round(precision_score(y, y_pred), 3)),
                    "Train ROC AUC": str(round(roc_auc_score(y, y_prob), 3)),
                    "Train PR AUC": str(round(average_precision_score(y, y_prob), 3)),
                    "Train Calibration error": str(round(calibration_error, 3))}

    train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
    riskslim_results = train_results.T
    riskslim_results.to_csv(filename)
    print(train_results)


def export_results_cv(risk_summary, filename):

    results = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
                "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
                "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
                "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
                "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

    results = pd.DataFrame.from_dict(results, orient='index', columns=['riskSLIM'])
        
    riskslim_results = results.T
    riskslim_results.to_csv(filename)



def compute_calibration(x, y, y_prob, y_pred):
    
    num_total_presc = len(y)
    table = []
    calibration_error = 0
    
    for prob in np.unique(y_prob):
        
        y_temp = y[y_prob == prob]
        y_pred_temp = y_pred[y_prob == prob]
        
        # prescription-level results 
        accuracy = round(accuracy_score(y_temp, y_pred_temp), 4)
        TN, FP, FN, TP = confusion_matrix(y_temp, y_pred_temp, labels=[0,1]).ravel() 
        observed_risk = np.count_nonzero(y_temp == 1) / len(y_temp)
        num_presc = TN + FP + FN + TP

        # patient-level results
        x_bucket = x[y_prob == prob]
        num_patients = len(pd.unique(x_bucket['patient_id']))
        num_longterm = len(pd.unique(x_bucket[x_bucket['days_to_long_term'] > 0]['patient_id']))

        table.append({'Prob': prob, 'Num_presc': num_presc,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'Accuracy': accuracy, 'Observed Risk': observed_risk, 
        'Num_patients': num_patients, 'Num_longterm': num_longterm})

        calibration_error += abs(prob - observed_risk) * num_presc/num_total_presc
    
    return calibration_error
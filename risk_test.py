#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 2023

Iterative method on tables

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score, confusion_matrix, roc_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import utils.model_selection as model_selection


def risk_test(year, table, first, upto180, median, county_name, setting_tag, output_columns=False, datadir='/export/storage_cures/CURES/Processed/'):

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
                                                        'consecutive_days': int})# .fillna(0)
    print(f'{file_path} imported with shape {FULL.shape}')

    FULL['Medicare_Medicaid'] = FULL['Medicare'] + FULL['Medicaid']
    if output_columns: print(FULL.columns.values.tolist())

    if median:
        quartile_list = ['patient_HPIQuartile', 
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
        
        print('WARNING: Encoding quartile values to binary\n')
        for col in quartile_list:
            FULL[col] = FULL[col].replace({1: 0, 2: 0, 3: 1, 4: 1})

    if county_name is not None: 
        zip_county = pd.read_csv(f'{datadir}/../CA/zip_county.csv', delimiter=",")
        FULL = FULL.merge(zip_county, left_on='patient_zip', right_on='zip', how='inner')
        indices = FULL.index[FULL['county'] == county_name].tolist()
        FULL = FULL.iloc[indices]
        print(f"Subsetting dataset to county {county_name} with {len(indices)} prescriptions.")

    print(f"Test reults are saved with setting tag: {setting_tag}")

    results, calibration_table = test_table(FULL, intercept=table['intercept'], conditions=table['conditions'], cutoffs=table['cutoffs'], scores=table['scores'], setting_tag=setting_tag, median=median)

    df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
    print(df)
    print(calibration_table)

    return



def test_table(FULL, intercept, conditions, cutoffs, scores, setting_tag, median, 
               fairness_results=False,
               patient_results=False,
               nth_presc_results=False,
               MME_results=False,
               export_files=True,
               barplot=True,
               optimal_thresh=False,
               exportdir='/export/storage_cures/CURES/Results/'):
    
    x = FULL
    y = FULL['long_term_180'].values

    x['Prob'] = x.apply(compute_score, axis=1, args=(intercept, conditions, cutoffs, scores,))
    # x['Prob'] = x.apply(compute_or_score, axis=1, args=(intercept, conditions, cutoffs, scores,)) # OR table
    y_prob = x['Prob'].to_numpy()

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    if optimal_thresh:
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
        
    x['Pred'] = (x['Prob'] >= optimal_threshold)
    y_pred = x['Pred'].to_numpy().astype(int)

    calibration_table, calibration_error, prob_pred, prob_true = compute_calibration(x, y, y_prob, y_pred, setting_tag)
    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 3)),
               "Recall": str(round(recall_score(y, y_pred), 3)),
               "Precision": str(round(precision_score(y, y_pred), 3)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 3)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 3)),
               "Calibration error": str(round(calibration_error, 3))}

    if fairness_results: compute_fairness(x, y, y_prob, y_pred, optimal_threshold, setting_tag)
    if patient_results: proportions = compute_patient(FULL, setting_tag)
    if nth_presc_results: compute_nth_presc(FULL, x)
    if MME_results: test_results_by_MME = compute_MME_presc(FULL, x)

    if barplot: barplot_by_condition(FULL, x, conditions, cutoffs, setting_tag)

    if export_files: 

        current_directory = os.getcwd()
        print(f"Current working directory: {current_directory}")

        calibration_table.to_csv(f'{exportdir}calibration{setting_tag}.csv', index=False)

        ### ROC
        roc_info = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        filename = f'output/baseline/riskSLIM_roc_test_info{"_median" if median else ""}{setting_tag}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(roc_info, f)
        print(f"ROC information for riskSLIM saved to {filename}, threshodls: {thresholds} with optimal threshold: {optimal_threshold}")

        ### Calibration
        # n_unique = len(np.unique(y_prob))
        # prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_unique, strategy='uniform')
        calibration_info = {"prob_true": prob_true, "prob_pred": prob_pred, "ece": calibration_error}
        filename = f'output/baseline/riskSLIM_calibration_test_info{"_median" if median else ""}{setting_tag}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(calibration_info, f)
        print(f"Calibration information for riskSLIM saved to {filename}")
        
        ### Proportion
        if patient_results: 
            proportion_info = {"month": [], "proportion": []}
            for month, proportion in proportions.items():
                proportion_info["month"].append(month)
                proportion_info["proportion"].append(proportion)
            filename = f'output/baseline/riskSLIM_proportions_test_info{"_median" if median else ""}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(proportion_info, f)
            print(f"Proportions information for riskSLIM saved to {filename}")

        ### Recall by MME bins
        if MME_results: 
            recall_by_MME_info = {"MME": [], "recall": [], "pos_ratio": [], "true_pos_ratio": []}
            for MME_bin, results in test_results_by_MME.items():
                recall_by_MME_info["MME"].append(MME_bin)
                recall_by_MME_info["recall"].append(results['test_recall'])
                recall_by_MME_info["pos_ratio"].append(results['correctly_predicted_positives_ratio'])
                recall_by_MME_info["true_pos_ratio"].append(results['true_positives_ratio'])
            filename = f'output/baseline/riskSLIM_recallMME_test_info{"_median" if median else ""}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(recall_by_MME_info, f)
            print(f"Recall by MME information for riskSLIM saved to {filename}")


    print('Test done!\n')

    return results, calibration_table
    


# ========================================================================================
# ========================================================================================
# ========================================================================================



def compute_score(row, intercept, conditions, cutoffs, scores):

    score = 0

    for i in range(len(conditions)):

        condition = conditions[i]
        cutoff = cutoffs[i]
        point = scores[i]

        if isinstance(cutoff, str):
            if row[condition] == int(cutoff): score += point
        else:
            if row[condition] >= cutoff: score += point

    return 1 / (1+np.exp(-(score + intercept)))



def compute_or_score(row, intercept, conditions, cutoffs, scores):

    # tables with OR conditions

    score = 0

    for i in range(len(conditions)):

        condition_group = conditions[i]
        cutoff_group = cutoffs[i]
        point = scores[i]

        # Check if any condition in the group meets the cutoff
        group_condition_met = False
        for j in range(len(condition_group)):
            condition = condition_group[j]
            cutoff = cutoff_group[j]

            if isinstance(cutoff, str):
                if row[condition] == int(cutoff): 
                    group_condition_met = True
                    break
            else:
                if row[condition] >= cutoff: 
                    group_condition_met = True
                    break

        if group_condition_met:
            score += point

    return 1 / (1+np.exp(-(score + intercept)))


# ========================================================================================
# ========================================================================================
# ========================================================================================



def compute_patient(FULL, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):
    
    '''
    Returns a patient table with date & days
    This is to see how early an alarm detects a patient
    '''
    
    FULL['Pred'] = FULL['Pred'].astype(int)

    # TP prescriptions
    FULL_TP = FULL[(FULL['long_term_180'] == 1) & (FULL['Pred'] == 1)] # presc from TP patient 
    # print("-"*100)
    # print(f"True positives prescriptions and alert by type:\n{FULL_TP.shape} {FULL_TP['alert1'].sum()} {FULL_TP['alert2'].sum()} {FULL_TP['alert3'].sum()} {FULL_TP['alert4'].sum()} {FULL_TP['alert5'].sum()} {FULL_TP['alert6'].sum()}")
    # print("-"*100)

    # prescriptions from true positive patients
    TP_PATIENT_ID = FULL_TP['patient_id'].unique()
    FULL = FULL[FULL.patient_id.isin(TP_PATIENT_ID)]
    print("-"*100)
    print(f"Total prescriptions from true positive patients: \n {FULL.shape}")
    print("-"*100)

    if FULL_TP.shape[0] == 0:
        print("No true positive patient found!")
        return

    PATIENT_TP = FULL.groupby('patient_id').apply(lambda x: pd.Series({
        'first_presc_date': x['date_filled'].iloc[0],
        'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
        'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min(),
        'first_long_term_180_date': x.loc[x['long_term_180'] == 1, 'date_filled'].iloc[0]
        # 'first_long_term_date': x.loc[x['long_term'] == 1, 'date_filled'].iloc[0] # don't exist if upto first long_term_180
    })).reset_index()
    
    PATIENT_TP = PATIENT_TP.groupby('patient_id').agg(
        first_presc_date=('first_presc_date', 'first'),
        first_pred_date=('first_pred_date', 'first'),
        first_pred_presc=('first_pred_presc', 'first'),
        first_long_term_180_date=('first_long_term_180_date', 'first')
    ).reset_index()    

    # NOTE: we don't have first_long_term_date as we focus up to first presciption
    # PATIENT_TP['day_to_long_term'] = (pd.to_datetime(PATIENT_TP['first_long_term_date'], format='%m/%d/%Y')
    #                                    - pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')).dt.days

    # PATIENT_TP['day_to_long_term_180'] = (pd.to_datetime(PATIENT_TP['first_long_term_180_date'], format='%m/%d/%Y')
    #                                         - pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')).dt.days
    
    # main metric how long it takes the model to predict long term
    PATIENT_TP['firstpred_from_firstpresc'] = (pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')
                                                - pd.to_datetime(PATIENT_TP['first_presc_date'], format='%m/%d/%Y')).dt.days
    

    proportions = {}
    for months in [1, 2, 3]:
        within_month = (PATIENT_TP['firstpred_from_firstpresc'] <= months * 30)
        proportions[months] = round(within_month.mean() * 100, 1)
        
    print(f"Proportion of LT users detected within a month: {proportions[1]}; two months: {proportions[2]}, three months: {proportions[3]}")    

    # PATIENT_TP.to_csv(f'{datadir}PATIENT_{str(year)}_LONGTERM_{case}_output_{name}.csv')
    
    
    # FULL_FP = FULL[(FULL['long_term_180'] == 0) & (FULL['Pred'] == 1)] # presc from FP patient 
    # print(FULL_FP.shape, FULL_FP['patient_id'].unique().shape)
    # FULL_ALERT16 = FULL_FP[(FULL_FP['alert1'] == 1) | (FULL_FP['alert6'] == 1)]
    # print(FULL_ALERT16.shape, FULL_ALERT16['patient_id'].unique().shape)
    
    # FP_PATIENT_ID = FULL_FP['patient_id'].unique()
    # FULL = FULL[FULL.patient_id.isin(FP_PATIENT_ID)]

    # PATIENT_FP = FULL.groupby('patient_id').apply(lambda x: pd.Series({
    #     'first_presc_date': x['date_filled'].iloc[0],
    #     'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
    #     'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min()
    # })).reset_index()
    
    # PATIENT_FP = PATIENT_FP.groupby('patient_id').agg(
    #     first_presc_date=('first_presc_date', 'first'),
    #     first_pred_date=('first_pred_date', 'first'),
    #     first_pred_presc=('first_pred_presc', 'first'),
    # ).reset_index()    

    # PATIENT_FP['firstpred_from_firstpresc'] = (pd.to_datetime(PATIENT_FP['first_pred_date'], format='%m/%d/%Y')
    #                                             - pd.to_datetime(PATIENT_FP['first_presc_date'], format='%m/%d/%Y')).dt.days
    
    # PATIENT_FP.to_csv(f'{datadir}PATIENT_{str(year)}_LONGTERM_{case}_output_FP.csv')
    
    return proportions




def compute_calibration(x, y, y_prob, y_pred, setting_tag, plot=False, exportdir='/export/storage_cures/CURES/Results/'):

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

    table = pd.DataFrame(table)
    prob = table['Prob'].values
    observed_risk = table['Observed Risk'].values
    
    # Plot calibration curve
    if plot:
        # only for simple models
        fig, ax = plt.subplots()
        ax.plot(table['Prob'], table['Observed Risk'], marker='o', linewidth=1, label='Calibration plot')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Risk')
        ax.set_title('Calibration Plot')
        ax.legend()
        fig.savefig(f'{exportdir}/Figures/Calibration{setting_tag}.pdf', dpi=300)
        print(f"Calibration curve saved as Calibration{setting_tag}.pdf\n")
    
    return table, calibration_error, prob, observed_risk



def compute_roc(y, y_prob, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    fig.savefig(f'{exportdir}/Figures/ROC{setting_tag}.pdf', dpi=300)
    print(f"ROC curve saved as ROC{setting_tag}.pdf\n")

    # Compute confusion matrix for each threshold
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred_threshold)
        print(f"Threshold: {threshold:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print()

    return



def compute_fairness(x, y, y_prob, y_pred, optimal_threshold, setting_tag, plot=False, exportdir='/export/storage_cures/CURES/Results/'):
    
    genders = x['patient_gender'].unique()
    roc_auc_by_gender, accuracy_by_gender, calibration_by_gender = {}, {}, {}
    fig, ax = plt.subplots()

    for gender in genders: # Male: 0, Female: 1
        gender_mask = x['patient_gender'] == gender
        X_gender = x[gender_mask]
        y_true_gender = y[gender_mask]
        y_prob_gender = y_prob[gender_mask]

        # print(y_true_gender)
        # print("-"*100)
        # print(y_prob_gender)
            
        # roc
        fpr, tpr, _ = roc_curve(y_true_gender, y_prob_gender)
        roc_auc = auc(fpr, tpr)
        roc_auc_by_gender[gender] = roc_auc
            
        # accuracy
        y_pred_gender = (y_prob_gender >= optimal_threshold).astype(int)
        assert np.array_equal(y_pred_gender, y_pred[gender_mask])
        accuracy = accuracy_score(y_true_gender, y_pred_gender)
        accuracy_by_gender[gender] = accuracy
        
        # calibration
        _, calibration_error, _, _ = compute_calibration(X_gender, y_true_gender, y_prob_gender, y_pred_gender, f'{setting_tag}_gender')
        calibration_by_gender[gender] = calibration_error

        ax.plot(fpr, tpr, label=f'{gender} (AUC = {roc_auc:.3f})')

    if plot:
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve by Gender')
        ax.legend(loc='best')
        fig.savefig(f'{exportdir}/Figures/ROC{setting_tag}_gender.pdf', dpi=300)
        print(f"ROC curve saved as ROC{setting_tag}_gender.pdf\n")

    for gender in genders:
        accuracy = accuracy_by_gender.get(gender, 'N/A')
        roc_auc = roc_auc_by_gender.get(gender, 'N/A')
        calibration_error = calibration_by_gender.get(gender, 'N/A')
        print(f'{gender}: Accuracy = {accuracy:.3f}, ROC AUC = {roc_auc:.3f}, Calibration = {calibration_error:.3f}')

    return



def compute_nth_presc(FULL, x, exportdir='/export/storage_cures/CURES/Results/'):

    FULL['num_prescriptions'] = FULL['num_prior_prescriptions'] + 1
    FULL['Prob'], FULL['Pred'] = x['Prob'], x['Pred']
    test_results_by_prescriptions = FULL[FULL['num_prescriptions'] <= 3].groupby('num_prescriptions').apply(lambda x: {'test_accuracy': accuracy_score(x['long_term_180'], x['Pred']),
                                                                                                                        # 'test_recall': recall_score(x['long_term_180'], x['Pred']),
                                                                                                                        # 'test_precision': precision_score(x['long_term_180'], x['Pred']),
                                                                                                                        'test_roc_auc': roc_auc_score(x['long_term_180'], x['Prob']),
                                                                                                                        'test_pr_auc': average_precision_score(x['long_term_180'], x['Prob']),
                                                                                                                        # different from the calibration error in the calibration table
                                                                                                                        'test_calibration_error': model_selection.compute_calibration(x['long_term_180'], x['Prob'], x['Pred'])}).to_dict()

    print_results(test_results_by_prescriptions)

    return



def compute_MME_presc(FULL, x, exportdir='/export/storage_cures/CURES/Results/'):

    FULL['Prob'], FULL['Pred'] = x['Prob'], x['Pred']
    cutoffs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    bin_labels = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', 
    '[60, 70)', '[70, 80)', '[80, 90)', '[90, 100)', 'above 100']

    MME_bins = pd.cut(FULL['concurrent_MME'], bins=cutoffs, labels=bin_labels, right=False)
    FULL['MME_bins'] = MME_bins
    test_results_by_MME = FULL.groupby('MME_bins').apply(lambda x: {'test_accuracy': accuracy_score(x['long_term_180'], x['Pred']),
                                                                    'test_recall': recall_score(x['long_term_180'], x['Pred']),           
                                                                    'test_roc_auc': roc_auc_score(x['long_term_180'], x['Prob']),
                                                                    'test_pr_auc': average_precision_score(x['long_term_180'], x['Prob']),
                                                                    'test_calibration_error': model_selection.compute_calibration(x['long_term_180'], x['Prob'], x['Pred']),
                                                                    'correctly_predicted_positives_ratio': ((x['Pred'] == 1) & (x['long_term_180'] == 1)).sum() / len(x),
                                                                    'true_positives_ratio': (x['long_term_180'] == 1).sum() / len(x)}).to_dict()

    print_results(test_results_by_MME)

    return test_results_by_MME



def print_results(results):
    # print results of dict in seperate rows
    for key, value in results.items():
        print(key)
        print(value)   
    print('\n')
    return



def barplot_by_condition(FULL, x, conditions, cutoffs, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):

    FULL['Pred'] = x['Pred'].astype(int)
    true_longterm_condition = (FULL['long_term_180'] == 1)
    predicted_longterm_condition = (FULL['Pred'] == 1)
    print(true_longterm_condition.sum(), predicted_longterm_condition.sum())

    for i in range(len(conditions)):
        feature = conditions[i]
        cutoff = cutoffs[i]
        print(f"Feature: {feature}, Cutoff: {cutoff}")
        
        if isinstance(cutoff, str):
            print("Should only be for Drug payment")
            FULL[f"{feature}_binary"] = 0
            FULL.loc[FULL[feature] == int(cutoff), f"{feature}_binary"] = 1
        else:
            FULL[f"{feature}_binary"] = 0
            FULL.loc[FULL[feature] >= cutoff, f"{feature}_binary"] = 1

    # conditions.extend([f"{feature}_binary" for feature in conditions])
    conditions = [f"{feature}_binary" for feature in conditions]
    conditions.extend(['long_term_180', 'Pred', 'Prob', 'patient_zip'])

    FULL_filtered = FULL[conditions]
    FULL_filtered.rename(columns={'prescriber_yr_avg_days_quartile_binary': 'prescriber_yr_avg_days_median_binary',\
        'pharmacy_yr_avg_days_quartile_binary': 'pharmacy_yr_avg_days_median_binary',\
            'family_poverty_pct_quartile_binary': 'family_poverty_pct_median_binary',\
            'long_term_180': 'True'}, inplace=True)

    FULL_filtered.to_csv(f'{exportdir}FULL_LTOUR{setting_tag}.csv', index=False)

    return
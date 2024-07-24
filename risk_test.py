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
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score, confusion_matrix, roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


def risk_test(year, table, first, upto180, setting_tag, datadir='/export/storage_cures/CURES/Processed/', output_columns=True):

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

    if output_columns: print(FULL.columns.values.tolist())
    
    results, calibration_table = test_table(FULL, intercept=table['intercept'], conditions=table['conditions'], cutoffs=table['cutoffs'], scores=table['scores'], setting_tag=setting_tag)

    df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
    print(df)
    print(calibration_table)

    return df, calibration_table


def test_table(FULL, intercept, conditions, cutoffs, scores, setting_tag,
               output_table=False, 
               roc=True, 
               calibration=True, 
               fairness_results=True,
               patient_results=False,
               filename='', 
               datadir='/export/storage_cures/CURES/Processed/', 
               exportdir='/export/storage_cures/CURES/Results/'):
    
    x = FULL
    y = FULL['long_term_180'].values

    x['Prob'] = x.apply(compute_score, axis=1, args=(intercept, conditions, cutoffs, scores,))
    # x['Prob'] = x.apply(compute_or_score, axis=1, args=(intercept, conditions, cutoffs, scores,))
    
    # if table == 'CURES': x['Pred'] = (x['Prob'] > 0.5)
    # else: x['Pred'] = (x['Prob'] > 0.05)
    
    x['Pred'] = (x['Prob'] >= 0.5)

    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy().astype(int)

    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4))}

    if calibration: 
        calibration_table, calibration_error = compute_calibration(x, y, y_prob, y_pred, setting_tag)
        results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4)),
               "Calibration error": str(round(calibration_error, 4))}
        
        calibration_table.to_csv(f'{exportdir}calibration{setting_tag}.csv', index=False)

    if roc: compute_roc(y, y_prob, setting_tag)
    if fairness_results: compute_fairness(x, y, y_prob, y_pred, setting_tag)

    if patient_results:




    if output_table: store_predicted_table(FULL, x, filename)

    print('Test done!\n')

    return results, calibration_table # , tpr, fpr, thresholds
    


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



def store_predicted_table(SAMPLE, x, name='', datadir='/mnt/phd/jihu/opioid/Data/'):
    
    '''
    Returns a patient table with date & days
    This is to see how early an alarm detects a patient
    '''
    
    # Convert column to integer
    SAMPLE['Pred'] = x['Pred'].astype(int)

    # true positive prescriptions
    SAMPLE_TP = SAMPLE[(SAMPLE['long_term_180'] == 1) & (SAMPLE['Pred'] == 1)] # presc from TP patient 
    print("-"*100)
    print('True positives prescriptions and alert by type: \n')
    print(SAMPLE_TP.shape, SAMPLE_TP['alert1'].sum(), SAMPLE_TP['alert2'].sum(), SAMPLE_TP['alert3'].sum(), SAMPLE_TP['alert4'].sum(), SAMPLE_TP['alert5'].sum(), SAMPLE_TP['alert6'].sum())
    print("-"*100)

    # prescriptions from true positive patients
    TP_PATIENT_ID = SAMPLE_TP['patient_id'].unique()
    SAMPLE = SAMPLE[SAMPLE.patient_id.isin(TP_PATIENT_ID)]
    print("-"*100)
    print('Total prescriptions from true positive patients: \n')
    print(SAMPLE.shape)
    print("-"*100)

    PATIENT_TP = SAMPLE.groupby('patient_id').apply(lambda x: pd.Series({
        'first_presc_date': x['date_filled'].iloc[0],
        'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
        'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min(),
        'first_long_term_180_date': x.loc[x['long_term_180'] == 1, 'date_filled'].iloc[0]
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
    

    within_month = PATIENT_TP['firstpred_from_firstpresc'] < 30
    proportion = within_month.mean()
    print(f'Proportion of LT users detected within a month: {proportion}')

    # PATIENT_TP.to_csv(f'{datadir}PATIENT_{str(year)}_LONGTERM_{case}_output_{name}.csv')
    
    
    # SAMPLE_FP = SAMPLE[(SAMPLE['long_term_180'] == 0) & (SAMPLE['Pred'] == 1)] # presc from FP patient 
    # print(SAMPLE_FP.shape, SAMPLE_FP['patient_id'].unique().shape)
    # SAMPLE_ALERT16 = SAMPLE_FP[(SAMPLE_FP['alert1'] == 1) | (SAMPLE_FP['alert6'] == 1)]
    # print(SAMPLE_ALERT16.shape, SAMPLE_ALERT16['patient_id'].unique().shape)
    
    # FP_PATIENT_ID = SAMPLE_FP['patient_id'].unique()
    # SAMPLE = SAMPLE[SAMPLE.patient_id.isin(FP_PATIENT_ID)]

    # PATIENT_FP = SAMPLE.groupby('patient_id').apply(lambda x: pd.Series({
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
    
    return




def compute_calibration(x, y, y_prob, y_pred, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):

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

    # Plot calibration curve
    fig, ax = plt.subplots()
    ax.plot(table['Prob'], table['Observed Risk'], marker='o', linewidth=1, label='Calibration plot')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Risk')
    ax.set_title('Calibration Plot')
    ax.legend()
    fig.savefig(f'{exportdir}/Figures/Calibration{setting_tag}.pdf', dpi=300)
    print(f"Calibration curve saved as Calibration{setting_tag}.pdf\n")
    
    return table, calibration_error



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



def compute_fairness(x, y, y_prob, y_pred, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):
    
    genders = x['patient_gender'].astype(str).unique()
    roc_auc_by_gender, accuracy_by_gender, calibration_by_gender = {}, {}, {}
    fig, ax = plt.subplots()

    for gender in genders:
        gender_mask = x['patient_gender'] == gender
        X_gender = x[gender_mask]
        y_true_gender = y[gender_mask]
        y_prob_gender = y_prob[gender_mask]
            
        # roc
        fpr, tpr, _ = roc_curve(y_true_gender, y_prob_gender)
        roc_auc = roc_auc_score(y_true_gender, y_prob_gender)
        roc_auc_by_gender[gender] = roc_auc
            
        # accuracy
        y_pred_gender = (y_prob_gender >= 0.5).astype(int)
        assert(y_pred_gender == y_pred[gender_mask])
        accuracy = accuracy_score(y_true_gender, y_pred_gender)
        accuracy_by_gender[gender] = accuracy
        
        # calibration
        _, calibration_error = compute_calibration(X_gender, y_true_gender, y_prob_gender, y_pred_gender)
        calibration_by_gender[gender] = calibration_error

        ax.plot(fpr, tpr, label=f'{gender} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve by Gender')
    ax.legend(loc='best')
    fig.savefig(f'{exportdir}/Figures/ROC_gender{setting_tag}.pdf', dpi=300)
    print(f"ROC curve saved as ROC_gender{setting_tag}.pdf\n")

    for gender in genders:
        accuracy = accuracy_by_gender.get(gender, 'N/A')
        roc_auc = roc_auc_by_gender.get(gender, 'N/A')
        calibration_error = calibration_by_gender.get(gender, 'N/A')
        print(f'{gender}: Accuracy = {accuracy:.4f}, ROC AUC = {roc_auc:.4f}, Calibration = {calibration_error:.4f}')

    return



# def compute_roc(y, y_prob, y_pred, table, resultdir, year=2019, export_file=False):

#     print("********************* HERE *********************")

#     FPR_list = []
#     TPR_list = []
#     TN_list, FP_list, FN_list, TP_list = [],[],[],[]
#     thresholds = np.arange(0, 1.0, 0.05)

#     for threshold in thresholds:

#         y_pred = (y_prob > threshold)   
#         y_pred = y_pred.astype(int)

#         TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()   
            
#         TN_list.append(TN)
#         FP_list.append(FP)
#         FN_list.append(FN)
#         TP_list.append(TP)
            
#         FPR = FP/(FP+TN)
#         TPR = TP/(TP+FN)
#         FPR_list.append(FPR)
#         TPR_list.append(TPR)                
            
#     TN_list = np.array(TN_list)
#     FP_list = np.array(FP_list)
#     FN_list = np.array(FN_list)
#     TP_list = np.array(TP_list)
#     FPR_list = np.array(FPR_list)
#     TPR_list = np.array(TPR_list)


#     if export_file:
#         np.savetxt(f'{resultdir}{table}_tn.csv', TN_list, delimiter = ",")
#         np.savetxt(f'{resultdir}{table}_fp.csv', FP_list, delimiter = ",")
#         np.savetxt(f'{resultdir}{table}_fn.csv', FN_list, delimiter = ",")
#         np.savetxt(f'{resultdir}{table}_tp.csv', TP_list, delimiter = ",")
            
#         np.savetxt(f'{resultdir}{table}_fpr.csv', FPR_list, delimiter = ",")
#         np.savetxt(f'{resultdir}{table}_tpr.csv', TPR_list, delimiter = ",")
#         np.savetxt(f'{resultdir}{table}_thresholds.csv', thresholds, delimiter = ",")

#     return TPR_list, FPR_list

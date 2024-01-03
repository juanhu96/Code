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
average_precision_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


def test_table(SAMPLE, year, table, intercept, conditions, cutoffs, scores, output_table=False, roc=False, calibration=True, filename='', datadir='/mnt/phd/jihu/opioid/Data/', resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    '''
    Compute the performance metric given a scoring table for a given year
    For tables with the six features only
    '''

    x = SAMPLE
    y = SAMPLE['long_term_180'].values


    ### Performance
    x['Prob'] = x.apply(compute_score, axis=1, args=(intercept, conditions, cutoffs, scores,))
    
    if table == 'CURES': x['Pred'] = (x['Prob'] > 0.5)
    else: x['Pred'] = (x['Prob'] > 0.05)

    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy().astype(int)

    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    

    if output_table: store_predicted_table(year, table, SAMPLE, x, filename)
    if roc: 
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        np.savetxt(f'{resultdir}{table}_test_fpr.csv', fpr, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_test_tpr.csv', tpr, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_test_thresholds.csv', thresholds, delimiter = ",")

        TPR_list, FPR_list = compute_roc(y, y_prob, y_pred, table=table, resultdir=resultdir, year=2019, export_file=True)

    if calibration: 
        calibration_table, calibration_error = compute_calibration(x, y, y_prob, y_pred, resultdir, f'{table}_{filename}')
        results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4)),
               "Calibration error": str(round(calibration_error, 4))}
    
    print('Test done!\n')

    return results, calibration_table, tpr, fpr, thresholds
    


# ========================================================================================



def test_table_full(year, output_table=False, roc=False, calibration=False, datadir='/mnt/phd/jihu/opioid/Data/', resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    '''
    Compute the performance metric given a scoring table for a given year
    '''
    
    ### Import 
    
    SAMPLE = pd.read_csv(f'{datadir}FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    x = SAMPLE
    y = SAMPLE[['long_term_180']].to_numpy().astype('int')
    

    ### Performance
    x['Prob'] = x.apply(compute_score_full_one, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_one = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                   "Recall": str(round(recall_score(y, y_pred), 4)),
                   "Precision": str(round(precision_score(y, y_pred), 4)),
                   "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                   "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_one = pd.DataFrame.from_dict(results_one, orient='index', columns=['1'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'one')

    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_one')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_two, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_two = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                   "Recall": str(round(recall_score(y, y_pred), 4)),
                   "Precision": str(round(precision_score(y, y_pred), 4)),
                   "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                   "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_two = pd.DataFrame.from_dict(results_two, orient='index', columns=['2'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'two')

    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_two')
        
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_three, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_three = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                     "Recall": str(round(recall_score(y, y_pred), 4)),
                     "Precision": str(round(precision_score(y, y_pred), 4)),
                     "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                     "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_three = pd.DataFrame.from_dict(results_three, orient='index', columns=['3'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'three')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_three')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_four, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_four = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                    "Recall": str(round(recall_score(y, y_pred), 4)),
                    "Precision": str(round(precision_score(y, y_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                    "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_four = pd.DataFrame.from_dict(results_four, orient='index', columns=['4'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'four')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_four')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_five, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_five = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                    "Recall": str(round(recall_score(y, y_pred), 4)),
                    "Precision": str(round(precision_score(y, y_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                    "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_five = pd.DataFrame.from_dict(results_five, orient='index', columns=['5'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'five')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_five')
    
    # ========================================================================================

    results = pd.concat([results_one, results_two], axis=1)
    results = pd.concat([results, results_three], axis=1)
    results = pd.concat([results, results_four], axis=1)
    results = pd.concat([results, results_five], axis=1)
    results = results.T
    results.to_csv(f'{resultdir}results_test_{str(year)}_LTOUR.csv')
    



# ========================================================================================



def compute_score(row, intercept, conditions, cutoffs, scores):

    score = 0

    for i in range(len(conditions)):

        condition = conditions[i]
        cutoff = cutoffs[i]
        point = scores[i]
        
        if row[condition] >= cutoff: score += point

    # if score >= -intercept - 1: score = -intercept - 1 # group high-risk
    # if score >= -intercept: score = -intercept # group high-risk

    return 1 / (1+np.exp(-(score + intercept)))



# ========================================================================================




def compute_score_full_one(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -3
    
    if row['avgDays'] >= 14:
        score += 3
    if row['avgDays'] >= 25:
        score += 2
    if row['concurrent_MME'] >= 15:
        score += 1
    if row['concurrent_MME'] >= 100:
        score += 1
    if row['concurrent_benzo'] >= 1:
        score += 1
    if row['payment'] == 'Medicare':
        score += 1
    
    return 1 / (1+np.exp(-(score + intercept)))


def compute_score_full_two(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -2
    
    if row['avgDays'] >= 10:
        score += 2
    if row['avgDays'] >= 21:
        score += 2
    if row['concurrent_benzo'] >= 1:
        score += 1
    # Hydromorphone, Methadone, Fentanyl, Oxymorphone
    if row['drug'] == 'Hydromorphone' or row['drug'] == 'Methadone' or row['drug'] == 'Fentanyl' or row['drug'] == 'Oxymorphone':
        score += 1
    if row['payment'] == 'Medicare':
        score += 1
    
    
    return 1 / (1+np.exp(-(score + intercept)))


def compute_score_full_three(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -2
    
    if row['avgDays'] >= 10:
        score += 2
    if row['avgDays'] >= 21:
        score += 2
    if row['concurrent_benzo'] >= 1:
        score += 1
    # Hydromorphone, Methadone, Fentanyl, Oxymorphone
    if row['drug'] == 'Hydromorphone' or row['drug'] == 'Methadone' or row['drug'] == 'Fentanyl' or row['drug'] == 'Oxymorphone':
        score += 1
    if row['payment'] == 'MilitaryIns':
        score += 1
    
    return 1 / (1+np.exp(-(score + intercept)))


def compute_score_full_four(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -2
    
    if row['avgDays'] >= 10:
        score += 2
    if row['avgDays'] >= 25:
        score += 2
    if row['concurrent_benzo'] >= 1:
        score += 1
    # Hydromorphone, Methadone, Fentanyl, Oxymorphone
    if row['drug'] == 'Hydromorphone' or row['drug'] == 'Methadone' or row['drug'] == 'Fentanyl' or row['drug'] == 'Oxymorphone':
        score += 1
    if row['payment'] == 'Medicare':
        score += 1
    
    return 1 / (1+np.exp(-(score + intercept)))


def compute_score_full(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -7
    
    if row['avgDays'] >= 7:
        score += 3
    if row['avgDays'] >= 14:
        score += 1
    if row['HMFO'] >=1:
        score += 1
    if row['concurrent_benzo'] >= 1 or row['Medicare'] >=1:
        score += 1
    
    return 1 / (1+np.exp(-(score + intercept)))



# ========================================================================================
# ========================================================================================
# ========================================================================================



def store_predicted_table(year, case, SAMPLE, x, name='', datadir='/mnt/phd/jihu/opioid/Data/'):
    
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



def compute_roc(y, y_prob, y_pred, table, resultdir, year=2019, export_file=False):

    print("********************* HERE *********************")

    FPR_list = []
    TPR_list = []
    TN_list, FP_list, FN_list, TP_list = [],[],[],[]
    thresholds = np.arange(0, 1.0, 0.05)

    for threshold in thresholds:

        y_pred = (y_prob > threshold)   
        y_pred = y_pred.astype(int)

        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()   
            
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


    if export_file:
        np.savetxt(f'{resultdir}{table}_tn.csv', TN_list, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_fp.csv', FP_list, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_fn.csv', FN_list, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_tp.csv', TP_list, delimiter = ",")
            
        np.savetxt(f'{resultdir}{table}_fpr.csv', FPR_list, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_tpr.csv', TPR_list, delimiter = ",")
        np.savetxt(f'{resultdir}{table}_thresholds.csv', thresholds, delimiter = ",")

    return TPR_list, FPR_list



def compute_calibration(x, y, y_prob, y_pred, resultdir, filename):
    
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
    
    return table, calibration_error


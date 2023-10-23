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
average_precision_score, accuracy_score, confusion_matrix

###############################################################################
###############################################################################
###############################################################################

def test_table(year, cutoffs, scores, case, outcome='long_term_180', output_table=False, roc=False, calibration=False, datadir='/mnt/phd/jihu/opioid/Data/', resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    '''
    Compute the performance metric given a scoring table for a given year
    '''

    
    ### Import 
    SAMPLE = pd.read_csv(f'{datadir}FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    if year == 2018: SAMPLE['concurrent_methadone_MME'] = SAMPLE['concurrent_MME_methadone']
    SAMPLE = SAMPLE.fillna(0)

    x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',\
                'num_pharmacies', 'concurrent_benzo', 'consecutive_days']] 
    y = SAMPLE[outcome].values

    
    ### Performance
    x['Prob'] = x.apply(compute_score, axis=1, args=(cutoffs, scores,))
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy().astype(int)

    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results = pd.DataFrame.from_dict(results, orient='index', columns=['Test'])
    results = results.T
    results.to_csv(f'{resultdir}results_test_{str(year)}_{case}.csv')

    ### Store the predicted table
    if output_table == True:
        store_predicted_table(year=year, case=case, SAMPLE=SAMPLE, x=x)
        
    if roc == True:
        # to make it more consistent we have to manually compute fpr, tpr
        FPR_list = []
        TPR_list = []
        TN_list, FP_list, FN_list, TP_list = [],[],[],[]
        thresholds = np.arange(0, 1.1, 0.1)
        for threshold in thresholds:
            
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
                   
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tn.csv', TN_list, delimiter = ",")
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fp.csv', FP_list, delimiter = ",")
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fn.csv', FN_list, delimiter = ",")
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tp.csv', TP_list, delimiter = ",")
        
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fpr.csv', FPR_list, delimiter = ",")
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tpr.csv', TPR_list, delimiter = ",")
        np.savetxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_thresholds.csv', thresholds, delimiter = ",")
    
    if calibration == True:
        compute_calibration(y, y_prob, y_pred, resultdir)


###############################################################################
###############################################################################
###############################################################################


def test_table_full(year, outcome='long_term_180', output_table=False, datadir='/mnt/phd/jihu/opioid/Data/', resultdir='/mnt/phd/jihu/opioid/Result/'):
    
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
    y = SAMPLE[[outcome]].to_numpy().astype('int')
    
    
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
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='one')
    
    ###########################################################################    
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
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='two')
    
    ###########################################################################
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
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='three')
    
    ###########################################################################
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
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='four')
    
    ###########################################################################
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
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='five')
    
    ###########################################################################
    results = pd.concat([results_one, results_two], axis=1)
    results = pd.concat([results, results_three], axis=1)
    results = pd.concat([results, results_four], axis=1)
    results = pd.concat([results, results_five], axis=1)
    results = results.T
    results.to_csv(f'{resultdir}results_test_{str(year)}_full.csv')
    

###############################################################################
###############################################################################
###############################################################################


def test_table_full_final(year, case, outcome='long_term_180', output_table=False, roc=False):
    
    '''
    Compute the performance metric given a scoring table for a given year
    '''
    os.chdir('/mnt/phd/jihu/opioid')
    
    ### Import 
    
    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    
    # SAMPLE['concurrent_methadone_MME'] = SAMPLE['concurrent_MME_methadone']
    
    SAMPLE = SAMPLE.fillna(0)
    x = SAMPLE
    y = SAMPLE[[outcome]].to_numpy().astype('int')
    
    print(x.columns)
    
    ### Performance
    x['Prob'] = x.apply(compute_score_full_two, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_one = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                   "Recall": str(round(recall_score(y, y_pred), 4)),
                   "Precision": str(round(precision_score(y, y_pred), 4)),
                   "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                   "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_one = pd.DataFrame.from_dict(results_one, orient='index', columns=['1'])
    print(results_one)
    
    if output_table == True:
        store_predicted_table(year=year, case='full', SAMPLE=SAMPLE, x=x, name='one')

    if roc == True:
        # to make it more consistent we have to manually compute fpr, tpr
        FPR_list = []
        TPR_list = []
        TN_list, FP_list, FN_list, TP_list = [],[],[],[]
        thresholds = np.arange(0, 1.1, 0.1)
        for threshold in thresholds:
            
            ## NOTE!!
            x['Pred'] = (x['Prob'] > threshold)
            y_pred = x['Pred'].to_numpy()
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
                   
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_tn.csv', TN_list, delimiter = ",")
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_fp.csv', FP_list, delimiter = ",")
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_fn.csv', FN_list, delimiter = ",")
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_tp.csv', TP_list, delimiter = ",")
        
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_fpr.csv', FPR_list, delimiter = ",")
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_tpr.csv', TPR_list, delimiter = ",")
        np.savetxt('Result/result_' + str(year) + '_' + case + '_single_balanced_thresholds.csv', thresholds, delimiter = ",")
    




###############################################################################
###############################################################################
###############################################################################

def test_table_extra(year, outcome='long_term_180'):
    
    '''
    Compute the performance metric given a scoring table for a given year
    '''
    os.chdir('/mnt/phd/jihu/opioid')
    
    ### Import 
    
    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_MME_methadone': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    
    SAMPLE['concurrent_methadone_MME'] = SAMPLE['concurrent_MME_methadone']

    SAMPLE = SAMPLE.fillna(0)
    x = SAMPLE
    y = SAMPLE[[outcome]].to_numpy().astype('int')

    ### Performance
    x['Prob'] = x.apply(compute_score_full_extra, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results = pd.DataFrame.from_dict(results, orient='index')
    results = results.T
    results.to_csv('Result/results_test_' + str(year) + '_full_extra.csv')


###############################################################################
###############################################################################
###############################################################################

def iterative_table(year, current_cutoffs, scores, feature, new_cutoff, outcome='long_term_180'):
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ### Import 
    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)
    x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',\
                'num_pharmacies', 'concurrent_benzo', 'consecutive_days']] 
    y = SAMPLE[[outcome]].to_numpy().astype('int')
    
    
    ### Current performance
    x['Prob'] = x.apply(compute_score, axis=1, args=(current_cutoffs, scores,))
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    current_results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                       "Recall": str(round(recall_score(y, y_pred), 4)),
                       "Precision": str(round(precision_score(y, y_pred), 4)),
                       "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                       "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    current_results = pd.DataFrame.from_dict(current_results, orient='index', columns=['Current'])
    total_results = current_results.copy()
    
    ### Update cutoff for selected feature
    cutoffs = {'intercept': current_cutoffs[0], 'concurrent_MME': current_cutoffs[1],\
               'concurrent_methadone_MME': current_cutoffs[2], 'num_prescribers': current_cutoffs[3],\
                'num_pharmacies': current_cutoffs[4], 'concurrent_benzo': current_cutoffs[5],\
                    'consecutive_days': current_cutoffs[6]}
    
    # x.to_csv('Result/iterative_table_current_x.csv')
    
        
    for i in range(len(new_cutoff)):
        cutoffs[feature] = new_cutoff[i]
        new_cutoffs = list(cutoffs.values())
        
        ### Update performance
        x['Prob'] = x.apply(compute_score, axis=1, args=(new_cutoffs, scores, ))
        x['Pred'] = (x['Prob'] > 0.5)
        y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
        
        updated_results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                           "Recall": str(round(recall_score(y, y_pred), 4)),
                           "Precision": str(round(precision_score(y, y_pred), 4)),
                           "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                           "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
        
        updated_results = pd.DataFrame.from_dict(updated_results, orient='index', columns=[str(new_cutoff[i])])
        total_results = pd.concat([total_results, updated_results], axis=1)
          
    ### Merge & export
    total_results = total_results.T
    total_results.to_csv('Result/iterative_table_' + str(year) + '_' + feature + '.csv')
    
    
###############################################################################
###############################################################################
###############################################################################

def compute_score(row, cutoff, scores):
    
    '''
    cutoff is a list of cutoffs for each basic feature, with intercept
    for features that does not show up in the table, we set the cutoff to 0
    
    [0, 60, 10, 3, 0, 2, 20]
    
    '''
       
    score = 0
    intercept = scores[0]
    
    if row['concurrent_MME'] >= cutoff[1]:
        score += scores[1]
    if row['concurrent_methadone_MME'] >= cutoff[2]:
        score += scores[2]
    if row['num_prescribers'] >= cutoff[3]:
        score += scores[3]
    if row['num_pharmacies'] >= cutoff[4]:
        score += scores[4]
    if row['concurrent_benzo'] >= cutoff[5]:
        score += scores[5]
    if row['consecutive_days'] >= cutoff[6]:
        score += scores[6]
    
    return 1 / (1+np.exp(-(score + intercept)))

###############################################################################
###############################################################################
###############################################################################

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


def compute_score_full_five(row):
    
    '''
    for full features we need to implement the features & cutoffs manually    
    '''
       
    score = 0
    intercept = -2
    
    if row['avgDays'] >= 10:
        score += 2
    if row['avgDays'] >= 25:
        score += 3
    if row['concurrent_benzo'] >= 1:
        score += 1
    # Hydromorphone, Methadone, Fentanyl, Oxymorphone
    if row['drug'] == 'Hydromorphone' or row['drug'] == 'Methadone' or row['drug'] == 'Fentanyl' or row['drug'] == 'Oxymorphone':
        score += 1
    
    return 1 / (1+np.exp(-(score + intercept)))


def compute_score_full_extra(row):
    
    '''
    For the extra round when we are doing iterative method.
    '''
       
    score = 0
    intercept = -3
    
    if row['consecutive_days'] >= 20:
        score += 4
    if row['age'] >= 30:
        score += 1
    if row['num_presc'] >= 4:
        score += 2
    
    return 1 / (1+np.exp(-(score + intercept)))

###############################################################################
###############################################################################
###############################################################################

def store_predicted_table(year, case, SAMPLE, x, name='', datadir='/mnt/phd/jihu/opioid/Data/'):
    
    '''
    Returns a patient table with date & days
    This is to see how early an alarm detects a patient
    '''
    
    # Convert column to integer
    SAMPLE['Pred'] = x['Pred'].astype(int)

    # Focus on TP prescriptions (42778 prescriptions from 29296 patients in 2019)
    # SAMPLE_TP = SAMPLE[(SAMPLE['long_term_180'] == 1) & (SAMPLE['Pred'] == 1)] # presc from TP patient 
    # print(SAMPLE_TP.shape)
    # print(SAMPLE_TP['alert1'].sum(), SAMPLE_TP['alert2'].sum(), SAMPLE_TP['alert3'].sum(), SAMPLE_TP['alert4'].sum(), SAMPLE_TP['alert5'].sum(), SAMPLE_TP['alert6'].sum())
    # SAMPLE_ALERT16 = SAMPLE_TP[(SAMPLE_TP['alert1'] == 1) | (SAMPLE_TP['alert6'] == 1)]
    # print(SAMPLE_ALERT16.shape, SAMPLE_ALERT16['patient_id'].unique().shape)

    # find the unique patient id and left join with the full SAMPLE
    # SAMPLE = pd.merge(SAMPLE, PATIENT, on='patient_id', how='left') # need to think more on this
    # TP_PATIENT_ID = SAMPLE_TP['patient_id'].unique()
    # SAMPLE = SAMPLE[SAMPLE.patient_id.isin(TP_PATIENT_ID)] # 51349 prescriptions (including TP) from 29296 patients
    # print(SAMPLE.shape)

    # PATIENT_TP = SAMPLE.groupby('patient_id').apply(lambda x: pd.Series({
    #     'first_presc_date': x['date_filled'].iloc[0],
    #     'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
    #     'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min(),
    #     'first_long_term_180_date': x.loc[x['long_term_180'] == 1, 'date_filled'].iloc[0]
    # })).reset_index()
    
    # PATIENT_TP = PATIENT_TP.groupby('patient_id').agg(
    #     first_presc_date=('first_presc_date', 'first'),
    #     first_pred_date=('first_pred_date', 'first'),
    #     first_pred_presc=('first_pred_presc', 'first'),
    #     first_long_term_180_date=('first_long_term_180_date', 'first')
    # ).reset_index()    

    # PATIENT_TP['day_to_long_term'] = (pd.to_datetime(PATIENT_TP['first_long_term_date'], format='%m/%d/%Y')
    #                                    - pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')).dt.days

    # PATIENT_TP['day_to_long_term_180'] = (pd.to_datetime(PATIENT_TP['first_long_term_180_date'], format='%m/%d/%Y')
    #                                         - pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')).dt.days
    
    ## how long it takes the model to predict long term
    # PATIENT_TP['firstpred_from_firstpresc'] = (pd.to_datetime(PATIENT_TP['first_pred_date'], format='%m/%d/%Y')
    #                                             - pd.to_datetime(PATIENT_TP['first_presc_date'], format='%m/%d/%Y')).dt.days
    
    # print(PATIENT_TP.shape)
    # PATIENT_TP.to_csv(f'{datadir}PATIENT_{str(year)}_LONGTERM_{case}_output_{name}.csv')
    
    
    SAMPLE_FP = SAMPLE[(SAMPLE['long_term_180'] == 0) & (SAMPLE['Pred'] == 1)] # presc from FP patient 
    print(SAMPLE_FP.shape, SAMPLE_FP['patient_id'].unique().shape)
    SAMPLE_ALERT16 = SAMPLE_FP[(SAMPLE_FP['alert1'] == 1) | (SAMPLE_FP['alert6'] == 1)]
    print(SAMPLE_ALERT16.shape, SAMPLE_ALERT16['patient_id'].unique().shape)
    
    FP_PATIENT_ID = SAMPLE_FP['patient_id'].unique()
    SAMPLE = SAMPLE[SAMPLE.patient_id.isin(FP_PATIENT_ID)]

    PATIENT_FP = SAMPLE.groupby('patient_id').apply(lambda x: pd.Series({
        'first_presc_date': x['date_filled'].iloc[0],
        'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
        'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min()
    })).reset_index()
    
    PATIENT_FP = PATIENT_FP.groupby('patient_id').agg(
        first_presc_date=('first_presc_date', 'first'),
        first_pred_date=('first_pred_date', 'first'),
        first_pred_presc=('first_pred_presc', 'first'),
    ).reset_index()    

    PATIENT_FP['firstpred_from_firstpresc'] = (pd.to_datetime(PATIENT_FP['first_pred_date'], format='%m/%d/%Y')
                                                - pd.to_datetime(PATIENT_FP['first_presc_date'], format='%m/%d/%Y')).dt.days
    
    print(PATIENT_FP.shape)
    PATIENT_FP.to_csv(f'{datadir}PATIENT_{str(year)}_LONGTERM_{case}_output_FP.csv')
    
    



def compute_calibration(y, y_prob, y_pred, resultdir):
    
    table = []

    for prob in unique(y_prob):
        
        y_temp = y[y_prob == prob]
        y_pred_temp = y_pred[y_prob == prob]

        TN, FP, FN, TP = confusion_matrix(y_temp, y_pred_temp).ravel() 

        table.append({'Prob': prob, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP})


    table = pd.DataFrame.from_dict(table, orient='index', columns=['Test'])
    table.to_csv(f'{resultdir}calibration.csv')

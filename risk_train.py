#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 2023

Functions for training

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, brier_score_loss, fbeta_score, accuracy_score, roc_curve, confusion_matrix

import utils.stumps as stumps
import pprint
import riskslim
import utils.RiskSLIM as slim
from riskslim.utils import print_model



def risk_train(year, features, scenario, c, weight = 'balanced', constraint=True, max_points=5,
               max_features=6, outcome = 'long_term_180', alpha='', beta='', output_y=False, name='', 
               selected_feautres=None, interaction_effects=False, roc=False, workdir='/mnt/phd/jihu/opioid/'):
    
    '''
    Train a riskSLIM model
    
    
    Parameters
    ----------
    year: year of the training dataset
    features: base/flexible/full/selected
    scenario: single/nested
    c: has to be a list when doing nested CV
    weight: original/balanced/positive/positive_2/positive_4
    constraint: operational constraint (at most one cutoff)
    max_points: maximum point allowed per feature
    outcome: outcome to predict
    alpha: weight on cutoff terms
    beta: weight on exponential terms
    output_y: whether to export the predicted y
    name: index for filename (when running multiple trails)
    selected_feautres: only if features = 'selected', list of selected features
    interaction_effects: if interaction effects will be included (only for full)
    roc: export fpr, tpr for roc visualization (only for single)
    '''
    

    SAMPLE = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)


    # ================================ X INPUT ====================================


    if features == 'base':
        
        SAMPLE['(Intercept)'] = 1
        intercept = SAMPLE.pop('(Intercept)')
        SAMPLE.insert(0, '(Intercept)', intercept)
        x = SAMPLE[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']] 
        
        new_constraints = None
        new_constraints_multiple = None
        essential_constraints = None
            
    elif features == 'full':
         
        N = 20
        SAMPLE_STUMPS = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST0.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST{str(i)}.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
        

        # Drop colinear & undisired
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'days_supply', 'daily_dose',
                                                                                                                  'quantity_per_day', 'total_dose', 'dose_diff',
                                                                                                                  'concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        
        # Drop the meaningless cutoffs & quantity
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['age18', 'num_prescribers1', 'num_pharmacies1', 'consecutive_days1', 'num_presc1'])]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['quantity10', 'quantity15', 'quantity20', 'quantity25', 'quantity30',
                                                                  'quantity40', 'quantity50', 'quantity75', 'quantity100', 'quantity150',
                                                                  'quantity200', 'quantity300'])]
        
        # Interaction
        if interaction_effects == False:
            SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('Codeine_MME', 'Hydrocodone_MME',
                                                                                                                      'Oxycodone_MME', 'Morphine_MME', 
                                                                                                                      'Hydromorphone_MME', 'Methadone_MME',
                                                                                                                      'Fentanyl_MME', 'Oxymorphone_MME'))])]
            
            # other interaction
            SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['Codeine_Medicaid', 'Codeine_CommercialIns', 
                                                                      'Codeine_Medicare', 'Codeine_CashCredit', 'Codeine_MilitaryIns', 'Codeine_WorkersComp', 'Codeine_Other',
                                                                      'Codeine_IndianNation', 'Hydrocodone_Medicaid', 'Hydrocodone_CommercialIns', 'Hydrocodone_Medicare',
                                                                      'Hydrocodone_CashCredit', 'Hydrocodone_MilitaryIns', 'Hydrocodone_WorkersComp', 'Hydrocodone_Other',
                                                                      'Hydrocodone_IndianNation', 'Oxycodone_Medicaid', 'Oxycodone_CommercialIns', 'Oxycodone_Medicare',
                                                                      'Oxycodone_CashCredit', 'Oxycodone_MilitaryIns', 'Oxycodone_WorkersComp', 'Oxycodone_Other',
                                                                      'Oxycodone_IndianNation', 'Morphine_Medicaid', 'Morphine_CommercialIns', 'Morphine_Medicare',
                                                                      'Morphine_CashCredit', 'Morphine_MilitaryIns', 'Morphine_WorkersComp', 'Morphine_Other',
                                                                      'Morphine_IndianNation', 'Hydromorphone_Medicaid', 'Hydromorphone_CommercialIns', 'Hydromorphone_Medicare',
                                                                      'Hydromorphone_CashCredit', 'Hydromorphone_MilitaryIns', 'Hydromorphone_WorkersComp', 'Hydromorphone_Other',
                                                                      'Hydromorphone_IndianNation', 'Methadone_Medicaid', 'Methadone_CommercialIns', 'Methadone_Medicare',
                                                                      'Methadone_CashCredit', 'Methadone_MilitaryIns', 'Methadone_WorkersComp', 'Methadone_Other', 'Methadone_IndianNation',
                                                                      'Fentanyl_Medicaid', 'Fentanyl_CommercialIns', 'Fentanyl_Medicare', 'Fentanyl_CashCredit', 'Fentanyl_MilitaryIns',
                                                                      'Fentanyl_WorkersComp', 'Fentanyl_Other', 'Fentanyl_IndianNation', 'Oxymorphone_Medicaid', 'Oxymorphone_CommercialIns',
                                                                      'Oxymorphone_Medicare', 'Oxymorphone_CashCredit', 'Oxymorphone_MilitaryIns', 'Oxymorphone_WorkersComp', 'Oxymorphone_Other',
                                                                      'Oxymorphone_IndianNation'])]
            
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS

        
        # Constraints
        if constraint == True and interaction_effects == True:
            
            # Single cutoff
            selected_features = ['age', 'num_prescribers', 'num_pharmacies', 
                                 'concurrent_methadone_MME', 'consecutive_days',
                                 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                                 'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                 'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
            
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)]) 
                
            new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine',
                                    'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone'])         
            new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                    'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])


            # Multiple cutoffs
            new_constraints_multiple = None
    
            
            # Essential features (NOTE: not working)
            essential_constraints = []
            essential_constraints.append([col for col in x if col.startswith('concurrent_MME')])
            

        elif constraint == True and interaction_effects == False:
            # Single cutoff
            selected_features = ['age', 'num_prescribers', 'num_pharmacies', 
                                 'concurrent_methadone_MME', 'consecutive_days',
                                 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff']
            
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)])
            
            new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine',
                                    'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone'])         
            new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                    'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])
  

            # Multiple cutoffs
            new_constraints_multiple = None
            
            
            # Essential features (NOTE: not working)
            essential_constraints = []
            essential_constraints.append([col for col in x if col.startswith('concurrent_MME')])
            essential_constraints.append([col for col in x if col.startswith('ever_switch')])
                
        else:
            new_constraints = None
            new_constraints_multiple = None
            essential_constraints = None
    

    else:
        
        # ================================== LTOUR ======================================
        
        N = 20
        SAMPLE_STUMPS = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST0.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST{str(i)}.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'age', 'days_supply', 'daily_dose',
                                                                                                                  'quantity_per_day', 'total_dose', 'dose_diff',
                                                                                                                  'concurrent_benzo_same', 'concurrent_benzo_diff',
                                                                                                                  'concurrent_methadone_MME', 'avgMME',
                                                                                                                  'consecutive_days'))])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['num_prescribers1', 'num_pharmacies1', 'num_presc1'])]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['quantity10', 'quantity15', 'quantity20', 'quantity25', 'quantity30',
                                                                  'quantity40', 'quantity50', 'quantity75', 'quantity100', 'quantity150',
                                                                  'quantity200', 'quantity300'])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('Codeine_MME', 'Hydrocodone_MME',
                                                                                                                  'Oxycodone_MME', 'Morphine_MME', 
                                                                                                                  'Hydromorphone_MME', 'Methadone_MME',
                                                                                                                  'Fentanyl_MME', 'Oxymorphone_MME'))])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['Codeine_Medicaid', 'Codeine_CommercialIns', 
                                                                  'Codeine_Medicare', 'Codeine_CashCredit', 'Codeine_MilitaryIns', 'Codeine_WorkersComp', 'Codeine_Other',
                                                                  'Codeine_IndianNation', 'Hydrocodone_Medicaid', 'Hydrocodone_CommercialIns', 'Hydrocodone_Medicare',
                                                                  'Hydrocodone_CashCredit', 'Hydrocodone_MilitaryIns', 'Hydrocodone_WorkersComp', 'Hydrocodone_Other',
                                                                  'Hydrocodone_IndianNation', 'Oxycodone_Medicaid', 'Oxycodone_CommercialIns', 'Oxycodone_Medicare',
                                                                  'Oxycodone_CashCredit', 'Oxycodone_MilitaryIns', 'Oxycodone_WorkersComp', 'Oxycodone_Other',
                                                                  'Oxycodone_IndianNation', 'Morphine_Medicaid', 'Morphine_CommercialIns', 'Morphine_Medicare',
                                                                  'Morphine_CashCredit', 'Morphine_MilitaryIns', 'Morphine_WorkersComp', 'Morphine_Other',
                                                                  'Morphine_IndianNation', 'Hydromorphone_Medicaid', 'Hydromorphone_CommercialIns', 'Hydromorphone_Medicare',
                                                                  'Hydromorphone_CashCredit', 'Hydromorphone_MilitaryIns', 'Hydromorphone_WorkersComp', 'Hydromorphone_Other',
                                                                  'Hydromorphone_IndianNation', 'Methadone_Medicaid', 'Methadone_CommercialIns', 'Methadone_Medicare',
                                                                  'Methadone_CashCredit', 'Methadone_MilitaryIns', 'Methadone_WorkersComp', 'Methadone_Other', 'Methadone_IndianNation',
                                                                  'Fentanyl_Medicaid', 'Fentanyl_CommercialIns', 'Fentanyl_Medicare', 'Fentanyl_CashCredit', 'Fentanyl_MilitaryIns',
                                                                  'Fentanyl_WorkersComp', 'Fentanyl_Other', 'Fentanyl_IndianNation', 'Oxymorphone_Medicaid', 'Oxymorphone_CommercialIns',
                                                                  'Oxymorphone_Medicare', 'Oxymorphone_CashCredit', 'Oxymorphone_MilitaryIns', 'Oxymorphone_WorkersComp', 'Oxymorphone_Other',
                                                                  'Oxymorphone_IndianNation'])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone'])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['MME_diff25', 'MME_diff50', 'MME_diff75', 'MME_diff100', 'MME_diff150',
                                                                  'quantity_diff25', 'quantity_diff50', 'quantity_diff75', 'quantity_diff100', 'quantity_diff150',
                                                                  'days_diff3', 'days_diff5', 'days_diff7', 'days_diff10', 'days_diff14', 'days_diff21', 'days_diff25', 'days_diff30'])]
        

        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        

        # Single cutoff
        selected_features = ['num_prescribers', 'num_pharmacies', 'num_presc']
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x if col.startswith(feature)])
        
        new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'])         
        new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])
        

        # Multiple cutoffs
        selected_features_multiple = ['concurrent_MME', 'avgDays']
        new_constraints_multiple = []
        for feature in selected_features_multiple:
            new_constraints_multiple.append([col for col in x if col.startswith(feature)])
        

        # Essential features (NOTE: not working)
        essential_constraints = None

    
    # =============================== SCENARIOS ====================================

        
    if scenario == 'single':
        
        y = SAMPLE[[outcome]].to_numpy().astype('int')    
        y[y==0]= -1
        
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
        
        print("Start training " + str(year) + scenario + features)
        start = time.time()    
        model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                                   max_coefficient=max_points, 
                                                                   max_L0_value=max_features,
                                                                   c0_value=c, 
                                                                   max_runtime=1000, 
                                                                   max_offset=5,
                                                                   class_weight=weight,
                                                                   new_constraints=new_constraints,
                                                                   new_constraints_multiple=new_constraints_multiple,
                                                                   essential_constraints=essential_constraints)
        print_model(model_info['solution'], new_train_data)
        print(str(round(time.time() - start,1)) + ' seconds')
        
        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                    "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                    "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
        train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
        riskslim_results = train_results.T
        riskslim_results.to_csv(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}.csv')
        

        if roc == True:        
            compute_roc(outer_train_prob, outer_train_y, y_pred, year, features, scenario, weight, name)

        if output_y == True:
            np.savetxt(f'{workdir}Result/riskSLIM_y.csv', outer_train_pred, delimiter=",")
        

    elif scenario == 'nested':
        
        y = SAMPLE[[outcome]].to_numpy().astype('int')    
        y[y==0]= -1
        
        print("Start training " + str(year) + scenario + features)
        start = time.time()
        risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                     Y=y,
                                                     y_label=outcome, 
                                                     max_coef=max_points, 
                                                     max_coef_number=max_features,
                                                     new_constraints=new_constraints,
                                                     new_constraints_multiple=new_constraints_multiple,
                                                     c=c,
                                                     class_weight=weight,
                                                     seed=42)
        
        end = time.time()
        print(str(round(end - start,1)) + ' seconds')    
        
        results = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
                   "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
                   "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
                   "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
                   "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

        results = pd.DataFrame.from_dict(results, orient='index', columns=['riskSLIM'])
        
        riskslim_results = results.T
        riskslim_results.to_csv(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}.csv')
    



def compute_roc(outer_train_prob, outer_train_y, y_pred, year, features, scenario, weight, name, workdir='/mnt/phd/jihu/opioid/'):

    # to make it more consistent we have to manually compute fpr, tpr
    FPR_list = []
    TPR_list = []
    TN_list, FP_list, FN_list, TP_list = [],[],[],[]
    thresholds = np.arange(0, 1.1, 0.1)
    
    for threshold in thresholds:

        y_pred = (outer_train_prob > threshold)

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

    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_tn.csv', TN_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_fp.csv', FP_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_fn.csv', FN_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_tp.csv', TP_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_fpr.csv', FPR_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_tpr.csv', TPR_list, delimiter = ",")
    np.savetxt(f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}_thresholds.csv', thresholds, delimiter = ",")

    return 
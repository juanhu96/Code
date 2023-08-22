#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 2023

Complete version of predicting long-term 180

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
               selected_feautres=None, interaction_effects=True, roc=False):
    
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
    
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################

    if features == 'base':
        
        ### X
        SAMPLE['(Intercept)'] = 1
        intercept = SAMPLE.pop('(Intercept)')
        SAMPLE.insert(0, '(Intercept)', intercept)
        x = SAMPLE[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']] 
        
        ### Constraints (no constraints for base)
        new_constraints = None
        new_constraints_multiple = None
        essential_constraints = None
        
    elif features == 'flexible':
        
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
            
        basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                          'num_prescribers', 'num_pharmacies',\
                                                                              'consecutive_days', 'concurrent_benzo'))]
        SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        
        # Manually drop
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0',
                                                                  'num_prescribers0', 'num_prescribers1',
                                                                  'num_pharmacies0','num_pharmacies1',
                                                                  'concurrent_benzo0', 'consecutive_days0'])]
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        if constraint == True:
            selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                 'num_pharmacies', 'concurrent_benzo', 'consecutive_days']
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)])
        else:
            new_constraints = None
            
    elif features == 'full':
         
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
        
        # Drop colinear & undisired
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'days_supply', 'daily_dose',
                                                                                                                  'quantity_per_day', 'total_dose', 'dose_diff',
                                                                                                                  'concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        
        ## Drop the meaningless cutoffs & quantity
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['age18', 'num_prescribers1', 'num_pharmacies1', 'consecutive_days1', 'num_presc1'])]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['quantity10', 'quantity15', 'quantity20', 'quantity25', 'quantity30',
                                                                  'quantity40', 'quantity50', 'quantity75', 'quantity100', 'quantity150',
                                                                  'quantity200', 'quantity300'])]
        
        ## Drop age to see if it picks up concurrent MME
        # SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('age'))])]
        
        
        ### Interaction
        if interaction_effects == False:
            SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('Codeine_MME', 'Hydrocodone_MME',
                                                                                                                      'Oxycodone_MME', 'Morphine_MME', 
                                                                                                                      'Hydromorphone_MME', 'Methadone_MME',
                                                                                                                      'Fentanyl_MME', 'Oxymorphone_MME'))])]
            
            ## Drop the other interaction
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
        # print(SAMPLE_STUMPS.columns)
        
        ### Constraints
        if constraint == True and interaction_effects == True:
            # Single cutoff
            selected_features = ['age', 'num_prescribers', 'num_pharmacies', 
                                 'concurrent_methadone_MME', 'consecutive_days',
                                 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                                 'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                 'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
            
            # selected_features = ['age', 'num_prescribers', 'num_pharmacies', 'concurrent_methadone_MME',
            #                       'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
            #                       'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
            #                       'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
            
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)]) 
                
            new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine',
                                    'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone'])         
            new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                    'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])
            # new_constraints.append(['Methadone', 'concurrent_methadone_MME'])
            
            new_constraints_multiple = None
            
            # Multiple cutoffs
            # selected_features_multiple = ['consecutive_days']
            # new_constraints_multiple = []
            # for feature in selected_features_multiple:
            #     new_constraints_multiple.append([col for col in x if col.startswith(feature)])
            
            essential_constraints = []
            essential_constraints.append([col for col in x if col.startswith('concurrent_MME')])
            
        elif constraint == True and interaction_effects == False:
            # Single cutoff
            selected_features = ['age', 'num_prescribers', 'num_pharmacies', 
                                 'concurrent_methadone_MME', 'consecutive_days',
                                 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff']
            
            # selected_features = ['age', 'num_prescribers', 'num_pharmacies', 'concurrent_methadone_MME',
            #                       'num_presc', 'MME_diff', 'days_diff', 'quantity_diff']
            
            
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)])
            
            new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine',
                                    'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone'])         
            new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                    'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])
            # new_constraints.append(['Methadone', 'concurrent_methadone_MME'])            
            
            new_constraints_multiple = None
            
            # Multiple cutoffs
            # selected_features_multiple = ['consecutive_days']
            # new_constraints_multiple = []
            # for feature in selected_features_multiple:
            #     new_constraints_multiple.append([col for col in x if col.startswith(feature)])
                
            essential_constraints = []
            essential_constraints.append([col for col in x if col.startswith('concurrent_MME')])
            essential_constraints.append([col for col in x if col.startswith('ever_switch')])
                
        else:
            new_constraints = None
            new_constraints_multiple = None
            essential_constraints = None
        
    
    elif features == 'selected':
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
            
        # SAMPLE_STUMPS = SAMPLE_STUMPS[['avgDays10', 'avgDays21', 'avgDays25', 'concurrent_benzo1', 'Medicare', 'HMFO', 'WorkersComp']]
        
        basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('avgDays', 'avgMME', 'concurrent_MME', 
                                                                          'concurrent_benzo1', 'Medicare', 'HMFO', 'WorkersComp'))]
        SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]
        
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        print(SAMPLE_STUMPS.columns)
        
        new_constraints = None
        new_constraints_multiple = []
        new_constraints_multiple.append([col for col in x if col.startswith(('avgDays', 'avgMME', 'concurrent_MME'))])
        # new_constraints_multiple = None
        essential_constraints = None
    
    else: ### special, withtout consecutive_days, MME
    
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(i) + '.csv', delimiter = ",")
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
        print(SAMPLE_STUMPS.columns)
        
        selected_features = ['num_prescribers', 'num_pharmacies', 'num_presc']
               
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x if col.startswith(feature)])
        
        new_constraints.append(['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'])         
        new_constraints.append(['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])
        
        # Multiple cutoffs
        # new_constraints_multiple = None
        selected_features_multiple = ['concurrent_MME', 'avgDays']
        new_constraints_multiple = []
        for feature in selected_features_multiple:
            new_constraints_multiple.append([col for col in x if col.startswith(feature)])
        
        
        # essential_constraints = []
        # essential_constraints.append([col for col in x if col.startswith('concurrent_MME')])
        essential_constraints = None
        
    ###############################################################################
    ################################## Scenario ###################################
    ###############################################################################
        
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
                                                                   max_offset=100,
                                                                   class_weight=weight,
                                                                   new_constraints=new_constraints,
                                                                   new_constraints_multiple=new_constraints_multiple,
                                                                   essential_constraints=essential_constraints)
        print_model(model_info['solution'], new_train_data)
        print(str(round(time.time() - start,1)) + ' seconds')
        
        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                    "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                    "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
        train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
        
        riskslim_results = train_results.T
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/Explore/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '.csv')
        
        if roc == True:        
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
                       
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_tn.csv', TN_list, delimiter = ",")
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_fp.csv', FP_list, delimiter = ",")
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_fn.csv', FN_list, delimiter = ",")
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_tp.csv', TP_list, delimiter = ",")
            
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_fpr.csv', FPR_list, delimiter = ",")
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_tpr.csv', TPR_list, delimiter = ",")
            np.savetxt('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '_thresholds.csv', thresholds, delimiter = ",")
            
        if output_y == True:
            np.savetxt("Result/riskSLIM_y.csv", outer_train_pred, delimiter=",")
        
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
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + name + '.csv')
    
    
###############################################################################
###############################################################################
###############################################################################     
    
def risk_train_two_stage(year, candidate_features, c, weight = 'balanced', outcome = 'long_term_180'):
    
    '''
    c has to be a list when doing nested CV
    Train the model using only candidate features from first round
    '''
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################
        
    ### X
    N = 20
    SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
    for i in range(1, N):
        TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
        SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['age0', 'concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                              'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                              'num_prescribers1','num_pharmacies1', 'num_presc0', 'num_presc1', 
                                                              'concurrent_benzo_same0', 'concurrent_benzo_diff0', 
                                                              'Codeine_MME0', 'Hydrocodone_MME0', 'Oxycodone_MME0', 'Morphine_MME0', 
                                                              'Hydromorphone_MME0', 'Methadone_MME0', 'Fentanyl_MME0', 'Oxymorphone_MME0'])]
    '''
    # An alternative automatic way of doing this
    cols_to_drop = SAMPLE_STUMPS.columns[SAMPLE_STUMPS.all()] # identify columns with all 1s
    SAMPLE_STUMPS = SAMPLE_STUMPS.drop(cols_to_drop, axis=1) # drop columns with all 1s     
    '''
    
    # avoid colinearity
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'opioid_days', 'quantity', 'quantity_per_day', 'total_dose'))])]
    
    # convert list to a tuple of strings
    candidate_features_stumps = [col for col in SAMPLE_STUMPS if col.startswith(tuple(str(x) for x in candidate_features))]
    SAMPLE_STUMPS = SAMPLE_STUMPS[candidate_features_stumps]
    
    SAMPLE_STUMPS['(Intercept)'] = 1
    intercept = SAMPLE_STUMPS.pop('(Intercept)')
    SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    x = SAMPLE_STUMPS
    
    ### Constraints (not same as candidate features, e.g. no switch)
    # selected_features = ['age', 'concurrent_MME',  'consecutive_days', 
    #                      'concurrent_benzo_same', 'num_presc', 'dose_diff']
    selected_features = ['age', 'concurrent_MME',  'consecutive_days', 
                         'concurrent_benzo_same', 'num_presc']
    new_constraints = []
    for feature in selected_features:
        new_constraints.append([col for col in x if col.startswith(feature)])     
        
    ###############################################################################
    ################################# Two-stage ###################################
    ###############################################################################
        
    y = SAMPLE[[outcome]].to_numpy().astype('int')    
    y[y==0]= -1
    
    print("Start training second round ...")
    start = time.time()
    risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                 Y=y,
                                                 y_label=outcome, 
                                                 max_coef=5, 
                                                 max_coef_number=6,
                                                 new_constraints=new_constraints,
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
    os.chdir('/mnt/phd/jihu/opioid')
    riskslim_results.to_csv('Result/result_secondround_' + str(year) + '.csv')
  
###############################################################################
###############################################################################
###############################################################################

def risk_train_last_stage(year, candidate_features, c, weight = 'balanced', outcome = 'long_term_180', name=''):
    
    '''
    Train the model using only candidate features from previous round
    For the last round, only do a single table training
    '''

    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################
        
    ### X
    N = 20
    SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(0) + '.csv', delimiter = ",")
    for i in range(1, N):
        TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS_UPTOFIRST' + str(i) + '.csv', delimiter = ",")
        SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
    
    # selected features only
    candidate_features_stumps = [col for col in SAMPLE_STUMPS if col.startswith(tuple(str(x) for x in candidate_features))]
    SAMPLE_STUMPS = SAMPLE_STUMPS[candidate_features_stumps]
    
    SAMPLE_STUMPS['(Intercept)'] = 1
    intercept = SAMPLE_STUMPS.pop('(Intercept)')
    SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    x = SAMPLE_STUMPS
    
    ### Constraints (not same as candidate features, e.g. no switch)
    # 'age', 'consecutive_days', 'ever_switch_drug', 'ever_switch_payment'
    selected_features = ['age']
    selected_features_multiple = ['consecutive_days']
    
    new_constraints = []
    for feature in selected_features:
        new_constraints.append([col for col in x if col.startswith(feature)]) 

    new_constraints_multiple = []
    for feature in selected_features_multiple:
        new_constraints_multiple.append([col for col in x if col.startswith(feature)])     
        
    ###############################################################################
    ################################### Single ####################################
    ###############################################################################

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
    
    print("Start training last round ...")
    start = time.time()    
    model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                               max_coefficient=5, 
                                                               max_L0_value=6, 
                                                               c0_value=c, 
                                                               max_runtime=1000, 
                                                               max_offset=100,
                                                               class_weight=weight,
                                                               new_constraints=new_constraints,
                                                               new_constraints_multiple=new_constraints_multiple)
    print_model(model_info['solution'], new_train_data)
    print(str(round(time.time() - start,1)) + ' seconds')
    
    ## Results
    outer_train_x = outer_train_x[:,1:]
    outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
    outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
    outer_train_pred = (outer_train_prob > 0.5)
    
    train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
    train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
    
    riskslim_results = train_results.T
    os.chdir('/mnt/phd/jihu/opioid')
    riskslim_results.to_csv('Result/Explore/result_lastround_' + str(year) + '_' + name + '.csv')
      
    
###############################################################################
###############################################################################
###############################################################################    
    
    
def risk_train_test(train_year, test_year, features, scenario, outcome = 'long_term_180'):
    
    '''
    Train a model on the train year
    Test it on test year
    '''
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE_TRAIN = pd.read_csv('Data/FULL_' + str(train_year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE_TRAIN = SAMPLE_TRAIN.fillna(0)
    
    
    SAMPLE_TEST = pd.read_csv('Data/FULL_' + str(test_year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE_TEST = SAMPLE_TEST.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################

    if features == 'base':
        
        ### X
        SAMPLE_TRAIN['(Intercept)'] = 1
        intercept = SAMPLE_TRAIN.pop('(Intercept)')
        SAMPLE_TRAIN.insert(0, '(Intercept)', intercept)
        x_train = SAMPLE_TRAIN[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']]
        
        
        ### X test
        SAMPLE_TEST['(Intercept)'] = 1
        intercept = SAMPLE_TEST.pop('(Intercept)')
        SAMPLE_TEST.insert(0, '(Intercept)', intercept)
        x_test = SAMPLE_TEST[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']]
        
        ### Constraints (no constraints for base)
        new_constraints = None
        
        
    elif features == 'flexible':
        
        ### X train
        N = 20
        SAMPLE_TRAIN_STUMPS = pd.read_csv('Data/FULL_' + str(train_year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(train_year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_TRAIN_STUMPS = pd.concat([SAMPLE_TRAIN_STUMPS, TEMP])
            
        basic_features = [col for col in SAMPLE_TRAIN_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                          'num_prescribers', 'num_pharmacies',\
                                                                              'consecutive_days', 'concurrent_benzo'))]
        SAMPLE_TRAIN_STUMPS = SAMPLE_TRAIN_STUMPS[basic_features]
        SAMPLE_TRAIN_STUMPS = SAMPLE_TRAIN_STUMPS[SAMPLE_TRAIN_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                  'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                  'num_prescribers1','num_pharmacies1'])]
        SAMPLE_TRAIN_STUMPS = SAMPLE_TRAIN_STUMPS[SAMPLE_TRAIN_STUMPS.columns.drop([col for col in SAMPLE_TRAIN_STUMPS if col.startswith(('concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        SAMPLE_TRAIN_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_TRAIN_STUMPS.pop('(Intercept)')
        SAMPLE_TRAIN_STUMPS.insert(0, '(Intercept)', intercept)
        x_train = SAMPLE_TRAIN_STUMPS
        
        
        ### X test
        SAMPLE_TEST_STUMPS = pd.read_csv('Data/FULL_' + str(test_year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(test_year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_TEST_STUMPS = pd.concat([SAMPLE_TEST_STUMPS, TEMP])
            
        basic_features = [col for col in SAMPLE_TEST_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                          'num_prescribers', 'num_pharmacies',\
                                                                              'consecutive_days', 'concurrent_benzo'))]
        SAMPLE_TEST_STUMPS = SAMPLE_TEST_STUMPS[basic_features]
        SAMPLE_TEST_STUMPS = SAMPLE_TEST_STUMPS[SAMPLE_TEST_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                  'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                  'num_prescribers1','num_pharmacies1'])]
        SAMPLE_TEST_STUMPS = SAMPLE_TEST_STUMPS[SAMPLE_TEST_STUMPS.columns.drop([col for col in SAMPLE_TEST_STUMPS if col.startswith(('concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        SAMPLE_TEST_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_TEST_STUMPS.pop('(Intercept)')
        SAMPLE_TEST_STUMPS.insert(0, '(Intercept)', intercept)
        x_test = SAMPLE_TEST_STUMPS
        
        ### Constraints
        selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                             'num_pharmacies', 'concurrent_benzo', 'consecutive_days']
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x_train if col.startswith(feature)])
        
        
    elif features == 'full':
         
        ### X train
        N = 20
        SAMPLE_TRAIN_STUMPS = pd.read_csv('Data/FULL_' + str(train_year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(train_year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_TRAIN_STUMPS = pd.concat([SAMPLE_TRAIN_STUMPS, TEMP])

        SAMPLE_TRAIN_STUMPS = SAMPLE_TRAIN_STUMPS[SAMPLE_TRAIN_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                                    'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                                    'num_prescribers1','num_pharmacies1', 'num_presc0', 'num_presc1', 
                                                                                    'concurrent_benzo_same0', 'concurrent_benzo_diff0', 
                                                                                    'Codeine_MME0', 'Hydrocodone_MME0', 'Oxycodone_MME0', 'Morphine_MME0', 
                                                                                    'Hydromorphone_MME0', 'Methadone_MME0', 'Fentanyl_MME0', 'Oxymorphone_MME0'])]

        SAMPLE_TRAIN_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_TRAIN_STUMPS.pop('(Intercept)')
        SAMPLE_TRAIN_STUMPS.insert(0, '(Intercept)', intercept)
        x_train = SAMPLE_TRAIN_STUMPS
        
        
        ### X test
        SAMPLE_TEST_STUMPS = pd.read_csv('Data/FULL_' + str(test_year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(test_year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_TEST_STUMPS = pd.concat([SAMPLE_TEST_STUMPS, TEMP])

        SAMPLE_TEST_STUMPS = SAMPLE_TEST_STUMPS[SAMPLE_TEST_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                                 'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                                 'num_prescribers1','num_pharmacies1', 'num_presc0', 'num_presc1', 
                                                                                 'concurrent_benzo_same0', 'concurrent_benzo_diff0', 
                                                                                 'Codeine_MME0', 'Hydrocodone_MME0', 'Oxycodone_MME0', 'Morphine_MME0', 
                                                                                 'Hydromorphone_MME0', 'Methadone_MME0', 'Fentanyl_MME0', 'Oxymorphone_MME0'])]

        SAMPLE_TEST_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_TEST_STUMPS.pop('(Intercept)')
        SAMPLE_TEST_STUMPS.insert(0, '(Intercept)', intercept)
        x_test = SAMPLE_TEST_STUMPS
        
        
        ### Constraints
        selected_features = ['age', 'quantity', 'quantity_per_day', 'total_dose', 
                             'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                             'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                             'concurrent_benzo_same', 'concurrent_benzo_diff', 
                             'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                             'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                             'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x_train if col.startswith(feature)]) 
         
        
    ###############################################################################
    ################################## Scenario ###################################
    ###############################################################################
        
    if scenario == 'single':
        
        y_train = SAMPLE_TRAIN[[outcome]].to_numpy().astype('int')    
        y_train[y_train==0]= -1
        
        cols = x_train.columns.tolist() 
        outer_train_sample_weight = np.repeat(1, len(y_train))
        outer_train_x, outer_train_y = x_train.values, y_train.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': 'long_term_180',
            'sample_weights': outer_train_sample_weight
        }
        
        print("Start training " + str(train_year) + scenario + features)
        start = time.time()    
        model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                                   max_coefficient=5, 
                                                                   max_L0_value=6, 
                                                                   c0_value=1e-4, 
                                                                   max_runtime=1000, 
                                                                   max_offset=100,
                                                                   class_weight='balanced',
                                                                   new_constraints=new_constraints)
        print_model(model_info['solution'], new_train_data)
        print(str(round(time.time() - start,1)) + ' seconds')
        
        ## Train Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                    "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                    "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
        train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
        
        
        ## Test Results
        y_test = SAMPLE_TEST[[outcome]].to_numpy().astype('int')    
        y_test[y_test==0]= -1
        outer_test_x, outer_test_y = x_test.values, y_test.reshape(-1,1)
        
        outer_test_x = outer_test_x[:,1:]
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        outer_test_prob = slim.riskslim_prediction(outer_test_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_pred = (outer_test_prob > 0.5)
        
        test_results = {"Accuracy": str(round(accuracy_score(outer_test_y, outer_test_pred), 4)),
                    "Recall": str(round(recall_score(outer_test_y, outer_test_pred), 4)),
                    "Precision": str(round(precision_score(outer_test_y, outer_test_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_test_y, outer_test_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_test_y, outer_test_prob), 4))}
        test_results = pd.DataFrame.from_dict(test_results, orient='index', columns=['Test'])
        
        results = pd.concat([train_results, test_results], axis = 1)
        riskslim_results = results.T
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/result_train_' + str(train_year) + '_test_' + str(test_year) + '_' + features + '_' + scenario + '.csv')
        
    
    ### Need to reimplement so that it can do train & test
    # elif scenario == 'nested':
        
    #     y = SAMPLE_TRAIN[[outcome]].to_numpy().astype('int')    
    #     y[y==0]= -1
        
    #     print("Start training " + str(train_year) + scenario + features)
    #     start = time.time()
    #     risk_summary = slim.risk_nested_cv_constrain(X=x,
    #                                                  Y=y,
    #                                                  y_label=outcome, 
    #                                                  max_coef=5, 
    #                                                  max_coef_number=6,
    #                                                  new_constraints=new_constraints,
    #                                                  c=[1e-5, 1e-4, 1e-3],
    #                                                  class_weight='balanced',
    #                                                  seed=42)
        
    #     end = time.time()
    #     print(str(round(end - start,1)) + ' seconds')    
        
    #     results = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
    #                "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
    #                "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
    #                "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
    #                "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

    #     results = pd.DataFrame.from_dict(results, orient='index', columns=['riskSLIM'])
        
    #     riskslim_results = results.T
    #     os.chdir('/mnt/phd/jihu/opioid')
    #     riskslim_results.to_csv('Result/result_' + str(train_year) + '_' + features + '_' + scenario + '.csv')    
        
        
    
###############################################################################
###############################################################################
###############################################################################

def risk_train_patient(year, features, scenario, case = None, outcome = 'long_term_ever'):
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE = pd.read_csv('Data/PATIENT_' + str(year) + '_LONGTERM.csv', delimiter = ",")
    SAMPLE = SAMPLE.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################

    if features == 'base':
        '''
        ### X
        SAMPLE['(Intercept)'] = 1
        intercept = SAMPLE.pop('(Intercept)')
        SAMPLE.insert(0, '(Intercept)', intercept)
        x = SAMPLE[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']] 
        
        ### Constraints (no constraints for base)
        new_constraints = None
        '''
        
        ### SAMPLE only contains patient id and long term ever
        
        
    elif features == 'flexible':
        
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/PATIENT_' + str(year) + '_' + case + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/PATIENT_' + str(year) + '_' + case + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
            
        basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                          'num_prescribers', 'num_pharmacies',\
                                                                              'consecutive_days', 'concurrent_benzo'))]
        SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                  'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                  'num_prescribers1','num_pharmacies1'])]
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                             'num_pharmacies', 'concurrent_benzo', 'consecutive_days']
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x if col.startswith(feature)])
        
    
    elif features == 'full':
         
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/PATIENT_' + str(year) + '_' + case + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/PATIENT_' + str(year) + '_' + case + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                  'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                  'num_prescribers1','num_pharmacies1'])]
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(list(SAMPLE_STUMPS.filter(regex='long_term_ever')))]

        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        selected_features = ['age', 'quantity', 'quantity_per_day', 'total_dose', 'daily_dose', 'days_supply',
                             'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                             'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                             'concurrent_benzo_same', 'concurrent_benzo_diff', 'opioid_days',
                             'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 
                             'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone',
                             'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                             'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                             'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'num_alert',
                             'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                             'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                             'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
        new_constraints = []
        for feature in selected_features:
            new_constraints.append([col for col in x if col.startswith(feature)]) 
         
        
    ###############################################################################
    ################################## Scenario ###################################
    ###############################################################################
        
    if scenario == 'single':
        
        y = SAMPLE[[outcome]].to_numpy().astype('int')    
        y[y==0]= -1
        
        print(np.count_nonzero(y == 1))
        
        cols = x.columns.tolist() 
        outer_train_sample_weight = np.repeat(1, len(y))
        outer_train_x, outer_train_y = x.values, y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': outcome,
            'sample_weights': outer_train_sample_weight
        }
        
        print("Start training " + str(year) + scenario + features + case)
        start = time.time()    
        model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                                   max_coefficient=5, 
                                                                   max_L0_value=6, 
                                                                   c0_value=1e-4, 
                                                                   max_runtime=1000, 
                                                                   max_offset=100,
                                                                   class_weight='balanced',
                                                                   new_constraints=new_constraints)
        print_model(model_info['solution'], new_train_data)
        print(str(round(time.time() - start,1)) + ' seconds')
        
        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                    "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                    "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
        train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
        
        riskslim_results = train_results.T
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + case + '.csv')
        
        
        
    elif scenario == 'nested':
        
        y = SAMPLE[[outcome]].to_numpy().astype('int')    
        y[y==0]= -1
        
        print("Start training " + str(year) + scenario + features + case)
        start = time.time()
        risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                     Y=y,
                                                     y_label=outcome, 
                                                     max_coef=5, 
                                                     max_coef_number=6,
                                                     new_constraints=new_constraints,
                                                     c=[1e-5, 1e-3],
                                                     class_weight='balanced',
                                                     seed=38)
        
        end = time.time()
        print(str(round(end - start,1)) + ' seconds')    
        
        results = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
                   "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
                   "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
                   "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
                   "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

        results = pd.DataFrame.from_dict(results, orient='index', columns=['riskSLIM'])
        
        riskslim_results = results.T
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + case + '.csv')    
    
    


###############################################################################
###############################################################################
###############################################################################



def risk_train_intervals(year, features, scenario, c, L0, weight = 'balanced', constraint=True, outcome = 'long_term_180', alpha='', beta='', output_y=False, name='', selected_feautres=None):
    
    '''
    Train a riskSLIM model
    
    
    Parameters
    ----------
    year: year of the training dataset
    features: base/flexible/full/selected
    scenario: single/nested
    c: has to be a list when doing nested CV
    L0: maximum number of features to selected
    weight: original/balanced/positive/positive_2/positive_4
    constraint: operational constraint (at most one cutoff)
    outcome: outcome to predict
    alpha: weight on cutoff terms
    beta: weight on exponential terms
    output_y: whether to export the predicted y
    name: index for filename (when running multiple trails)
    selected_feautres: only if features = 'selected', list of selected features
    '''
    
    
    os.chdir('/mnt/phd/jihu/opioid')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################

    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    ###############################################################################
    ################################## X input ####################################
    ###############################################################################

    if features == 'base':
        
        ### X
        SAMPLE['(Intercept)'] = 1
        intercept = SAMPLE.pop('(Intercept)')
        SAMPLE.insert(0, '(Intercept)', intercept)
        x = SAMPLE[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']] 
        
        ### Constraints (no constraints for base)
        new_constraints = None
        
        
    elif features == 'flexible':
        
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + 'flexible_INTERVALS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + 'flexible_INTERVALS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
            
        basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                          'num_prescribers', 'num_pharmacies',\
                                                                              'consecutive_days', 'concurrent_benzo'))]
        SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('concurrent_benzo_same', 'concurrent_benzo_diff'))])]
        
        # Manually drop
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['num_prescribers0','num_pharmacies0', 'concurrent_benzo0',\
                                                                  'num_prescribers1','num_pharmacies1'])]
        '''
        # An alternative automatic way of doing this
        # This gets trick when testing, because the columns with all 1 might not match
        cols_to_drop = SAMPLE_STUMPS.columns[SAMPLE_STUMPS.all()] # identify columns with all 1s
        SAMPLE_STUMPS = SAMPLE_STUMPS.drop(cols_to_drop, axis=1) # drop columns with all 1s     
        '''
        
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        if constraint == True:
            selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                 'num_pharmacies', 'concurrent_benzo', 'consecutive_days']
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)])
        else:
            new_constraints = None
            
    elif features == 'full':
         
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
        
        # age 20 instead of age 0
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['age20', 'concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                  'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                  'num_prescribers1','num_pharmacies1', 'num_presc0', 'num_presc1', 
                                                                  'concurrent_benzo_same0', 'concurrent_benzo_diff0', 
                                                                  'Codeine_MME0', 'Hydrocodone_MME0', 'Oxycodone_MME0', 'Morphine_MME0', 
                                                                  'Hydromorphone_MME0', 'Methadone_MME0', 'Fentanyl_MME0', 'Oxymorphone_MME0'])]
        '''
        # An alternative automatic way of doing this
        cols_to_drop = SAMPLE_STUMPS.columns[SAMPLE_STUMPS.all()] # identify columns with all 1s
        SAMPLE_STUMPS = SAMPLE_STUMPS.drop(cols_to_drop, axis=1) # drop columns with all 1s     
        '''
        
        SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'opioid_days','quantity', 'quantity_per_day', 'total_dose'))])]
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        if constraint == True:
            selected_features = ['age', 'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                 'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                                 'concurrent_benzo_same', 'concurrent_benzo_diff', 
                                 'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                                 'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                 'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']
            new_constraints = []
            for feature in selected_features:
                new_constraints.append([col for col in x if col.startswith(feature)]) 
        else:
            new_constraints = None
        
    elif features == 'selected':
        ### X
        N = 20
        SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
        for i in range(1, N):
            TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
            SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])
        
        # convert list to a tuple of strings
        candidate_features_stumps = [col for col in SAMPLE_STUMPS if col.startswith(tuple(str(x) for x in selected_feautres))]
        SAMPLE_STUMPS = SAMPLE_STUMPS[candidate_features_stumps]
        SAMPLE_STUMPS['(Intercept)'] = 1
        intercept = SAMPLE_STUMPS.pop('(Intercept)')
        SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
        x = SAMPLE_STUMPS
        
        ### Constraints
        ### TODO: how to automatically identify which feature need to add constraint?
        new_constraints = None
        
    ###############################################################################
    ################################## Scenario ###################################
    ###############################################################################
        
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
                                                                   max_coefficient=5, 
                                                                   max_L0_value=L0, 
                                                                   c0_value=c, 
                                                                   max_runtime=1000, 
                                                                   max_offset=100,
                                                                   class_weight=weight,
                                                                   new_constraints=new_constraints)
        print_model(model_info['solution'], new_train_data)
        print(str(round(time.time() - start,1)) + ' seconds')
        
        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                    "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                    "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                    "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
        train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
        
        riskslim_results = train_results.T
        os.chdir('/mnt/phd/jihu/opioid')
        # riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + alpha + '_' + beta + '_exp' + name + '.csv')
        riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_temp' + name + '.csv')
        
        # fpr, tpr, thresholds = roc_curve(outer_train_y, outer_train_prob)
        # np.savetxt('Result/riskSLIM_' + str(year) + '_' + features + '_' + scenario + '_fpr.csv', fpr, delimiter = ",")
        # np.savetxt('Result/riskSLIM_' + str(year) + '_' + features + '_' + scenario + '_tpr.csv', tpr, delimiter = ",")
        # np.savetxt('Result/riskSLIM_' + str(year) + '_' + features + '_' + scenario + '_thresholds.csv', thresholds, delimiter = ",")
        
        if output_y == True:
            np.savetxt("Result/riskSLIM_y.csv", outer_train_pred, delimiter=",")
        
    elif scenario == 'nested':
        
        y = SAMPLE[[outcome]].to_numpy().astype('int')    
        y[y==0]= -1
        
        print("Start training " + str(year) + scenario + features)
        start = time.time()
        risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                     Y=y,
                                                     y_label=outcome, 
                                                     max_coef=5, 
                                                     max_coef_number=L0,
                                                     new_constraints=new_constraints,
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
        os.chdir('/mnt/phd/jihu/opioid')
        riskslim_results.to_csv('Result/result_' + str(year) + '_' + features + '_' + scenario + '_' + weight + 'intervals_L08_' + name + '.csv')
 





    
###############################################################################
###############################################################################
###############################################################################

    
def create_stumps(year, scenario = ''):
    
    '''
    year = 2019
    scenario = '' # default empty, FIRST, SECOND, THIRD
    '''
        
    os.chdir('/mnt/phd/jihu/opioid')

    ###############################################################################
    ###############################################################################
    ###############################################################################

    FULL = pd.read_csv('Data/FULL_' + str(year) + scenario + '_LONGTERM_INPUT.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int})

    FULL = FULL.fillna(0)
        
    ###############################################################################
    ###############################################################################
    ###############################################################################


    ## Drop the columns that start with alert
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
    FULL = FULL.drop(columns = ['drug_payment'])

    # x_all = FULL[['age', 'quantity', 'days_supply', 'quantity_per_day', 'daily_dose', 'total_dose',
    #               'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
    #               'num_pharmacies', 'concurrent_benzo', 'concurrent_benzo_same', 'concurrent_benzo_diff', 
    #               'consecutive_days', 'num_presc', 'dose_diff', 'MME_diff', 'days_diff', 'quantity_diff',
    #               'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
    #               'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
    
    x_all = FULL[['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                  'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                  'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                  'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                  'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]

    cutoffs = []
    # case1 = ['num_prescribers', 'num_pharmacies', 'concurrent_benzo', 'concurrent_benzo_same', 'concurrent_benzo_diff', 'num_presc']
    # case2 = ['days_supply', 'concurrent_methadone_MME', 'consecutive_days', 'dose_diff', 'MME_diff', 'days_diff']
    # for column_name in x_all.columns:
    #     if column_name in case1:
    #         cutoffs.append([n for n in range(1, 11)])
    #     elif column_name in case2:
    #         cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
    #     elif column_name == 'age':
    #         cutoffs.append([n for n in range(20, 70) if n % 10 == 0])
    #     else:
    #         cutoffs.append([n for n in range(0, 210) if n % 10 == 0])
    
    for column_name in x_all.columns:
        if column_name == 'age':
            cutoffs.append([18, 25, 30, 40, 50, 60, 70, 80, 90])
        elif column_name == 'quantity':
            cutoffs.append([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300])
        elif column_name == 'days_supply':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        elif column_name == 'concurrent_MME' or column_name == 'concurrent_methadone_MME':
            cutoffs.append([10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300])
        elif column_name == 'num_prescribers' or column_name == 'num_pharmacies':
            cutoffs.append([n for n in range(1, 11)])
        elif column_name == 'concurrent_benzo':
            cutoffs.append([1])
        elif column_name == 'consecutive_days':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30, 60, 90])
        elif column_name == 'num_presc':
            cutoffs.append([1, 2, 3, 4, 5, 6, 10, 15])
        elif column_name == 'MME_diff' or column_name == 'quantity_diff':
            cutoffs.append([10, 25, 50, 75, 100, 150])
        elif column_name == 'days_diff':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        else:
            cutoffs.append([10, 20, 30, 40, 50, 75, 100, 200, 300]) # interaction MMEs
        

    ## Divide into 20 folds
    N = 20
    FULL_splited = np.array_split(FULL, N)
    for i in range(N):
        FULL_fold = FULL_splited[i]
        # x = FULL_fold[['age', 'quantity', 'days_supply', 'quantity_per_day', 'daily_dose', 'total_dose',
        #                'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
        #                'num_pharmacies', 'concurrent_benzo', 'concurrent_benzo_same', 'concurrent_benzo_diff', 
        #                'consecutive_days', 'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
        #                'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
        #                'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
        
        
        x = FULL_fold[['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                       'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                       'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                       'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                       'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
        
        
        x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
        # bind the data with the ones that was not for create_stumps
        # x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'days_supply', 'quantity_per_day', 'daily_dose', 'total_dose',
        #                                            'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
        #                                            'num_pharmacies', 'concurrent_benzo', 'concurrent_benzo_same', 'concurrent_benzo_diff', 
        #                                            'consecutive_days', 'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
        #                                            'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
        #                                            'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME'])]
        
        x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                                                   'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                                                   'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                                                   'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                                   'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME'])]
        
        new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
        print(new_data.shape)
        new_data.to_csv('Data/FULL_' + str(year) + scenario + '_STUMPS' + str(i) + '.csv', header=True, index=False)
        
        
        
###############################################################################
###############################################################################
###############################################################################
def create_stumps_uptofirst(year, scenario = ''):
    
    '''
    year = 2019
    '''
        
    os.chdir('/mnt/phd/jihu/opioid')

    ###############################################################################
    ###############################################################################
    ###############################################################################

    FULL = pd.read_csv('Data/FULL_' + str(year) + scenario + '_LONGTERM_INPUT_UPTOFIRST.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int})

    FULL = FULL.fillna(0)
        
    ###############################################################################
    ###############################################################################
    ###############################################################################


    ## Drop the columns that start with alert
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
    FULL = FULL.drop(columns = ['drug_payment'])
    
    x_all = FULL[['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                  'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                  'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                  'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                  'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME',
                  'avgMME', 'avgDays']]

    cutoffs = []
    
    for column_name in x_all.columns:
        if column_name == 'age':
            cutoffs.append([18, 25, 30, 40, 50, 60, 70, 80, 90])
        elif column_name == 'quantity':
            cutoffs.append([10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300])
        elif column_name == 'days_supply':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        elif column_name == 'concurrent_MME' or column_name == 'concurrent_methadone_MME':
            cutoffs.append([10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300])
        elif column_name == 'num_prescribers' or column_name == 'num_pharmacies':
            cutoffs.append([n for n in range(1, 11)])
        elif column_name == 'concurrent_benzo':
            cutoffs.append([1])
        elif column_name == 'consecutive_days':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30, 60, 90])
        elif column_name == 'num_presc':
            cutoffs.append([1, 2, 3, 4, 5, 6, 10, 15])
        elif column_name == 'MME_diff' or column_name == 'quantity_diff':
            cutoffs.append([10, 25, 50, 75, 100, 150])
        elif column_name == 'days_diff':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30])
        elif column_name == 'avgMME':
            cutoffs.append([15, 20, 25, 30, 35, 40, 50, 70, 90])
        elif column_name == 'avgDays':
            cutoffs.append([1, 3, 5, 7, 10, 14, 21, 25, 30, 60, 90])
        else:
            cutoffs.append([10, 20, 30, 40, 50, 75, 100, 200, 300]) # interaction MMEs
        

    ## Divide into 20 folds
    N = 20
    FULL_splited = np.array_split(FULL, N)
    for i in range(N):
        FULL_fold = FULL_splited[i]      
        x = FULL_fold[['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                       'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                       'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                       'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                       'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME',
                       'avgMME', 'avgDays']]
        
        
        x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)       
        x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'days_supply', 'concurrent_MME', 'concurrent_methadone_MME', 
                                                   'num_prescribers', 'num_pharmacies', 'concurrent_benzo',
                                                   'consecutive_days', 'num_presc', 'MME_diff', 'days_diff', 'quantity_diff',
                                                   'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                                   'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME',
                                                   'avgMME', 'avgDays'])]
        
        new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
        print(new_data.shape)
        new_data.to_csv('Data/FULL_' + str(year) + scenario + '_STUMPS_UPTOFIRST' + str(i) + '.csv', header=True, index=False)
        
###############################################################################
###############################################################################
###############################################################################        
        
def create_stumps_patient(year, scenario = ''):
    
    '''
    year = 2019
    scenario = '' # default empty, FIRST, SECOND, THIRD
    
    For SECOND, THIRD use this function instead -- need to drop patient id
    
    '''
        
    os.chdir('/mnt/phd/jihu/opioid')

    ###############################################################################
    ###############################################################################
    ###############################################################################

    FULL = pd.read_csv('Data/PATIENT_' + str(year) + '_' + scenario + '_LONGTERM_INPUT.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int})

    FULL = FULL.fillna(0)
    print(FULL.shape)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################


    ## Drop the columns that start with alert
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='patient_id')))]

    x_all = FULL[['age', 'quantity', 'quantity_per_day', 'total_dose', 'daily_dose', 'days_supply',
                  'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                  'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                  'concurrent_benzo_same', 'concurrent_benzo_diff', 'opioid_days',
                  'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 
                  'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone',
                  'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                  'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                  'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'num_alert',
                  'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                  'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                  'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]

    cutoffs = []
    for column_name in x_all.columns:
        if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
            cutoffs.append([n for n in range(0, 10)])
        elif column_name == 'concurrent_benzo' or column_name == 'concurrent_benzo_same' or \
            column_name == 'concurrent_benzo_diff' or column_name == 'num_presc' or column_name == 'num_alert':
            cutoffs.append([0, 1, 2, 3, 4, 5, 10])
        elif column_name == 'consecutive_days' or column_name == 'age' or column_name == 'concurrent_methadone_MME' or \
            column_name == 'days_diff' or column_name == 'days_supply':
            cutoffs.append([n for n in range(0, 90) if n % 10 == 0])
        elif column_name == 'dose_diff' or column_name == 'MME_diff':
            cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
        elif column_name == 'Codeine' or column_name == 'Hydrocodone' or column_name == 'Oxycodone' or \
            column_name == 'Morphine' or column_name == 'Hydromorphone' or column_name == 'Methadone' or \
                column_name == 'Fentanyl' or column_name == 'Oxymorphone':
                    cutoffs.append([0, 1, 2, 3])
        elif column_name == 'Medicaid' or column_name == 'CommercialIns' or column_name == 'Medicare' or \
            column_name == 'CashCredit' or column_name == 'MilitaryIns' or column_name == 'WorkersComp' or \
                column_name == 'Other' or column_name == 'IndianNation':
                    cutoffs.append([0, 1, 2, 3])
        elif column_name == 'alert1' or column_name == 'alert2' or column_name == 'alert3' or \
            column_name == 'alert4' or column_name == 'alert5' or column_name == 'alert6':
                cutoffs.append([0, 1, 2, 3])                    
        else:
            cutoffs.append([n for n in range(0, 200) if n % 10 == 0])

    ## Divide into 20 folds
    N = 20
    FULL_splited = np.array_split(FULL, N)
    for i in range(N):
        # print(i)
        FULL_fold = FULL_splited[i]
        x = FULL_fold[['age', 'quantity', 'quantity_per_day', 'total_dose', 'daily_dose', 'days_supply',
                       'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                       'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                       'concurrent_benzo_same', 'concurrent_benzo_diff', 'opioid_days',
                       'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 
                       'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone',
                       'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                       'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                       'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'num_alert',
                       'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                       'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                       'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
        
        x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
        # bind the data with the ones that was not for create_stumps
        x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'quantity_per_day', 'total_dose', 'daily_dose', 'days_supply',
                                                   'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                                   'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                                                   'concurrent_benzo_same', 'concurrent_benzo_diff', 'opioid_days',
                                                   'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 
                                                   'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone',
                                                   'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
                                                   'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation',
                                                   'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'num_alert',
                                                   'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                                                   'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                                   'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME'])]
        
        new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
        print(new_data.shape)
        new_data.to_csv('Data/PATIENT_' + str(year) + '_' + scenario + '_STUMPS' + str(i) + '.csv', header=True, index=False)




###############################################################################
###############################################################################
###############################################################################

    
def create_intervals(year, scenario='flexible'):
    
    '''
    Create intervals stumps for the dataset
    For this we also need to edit stumps.create_stumps as well
    
    Parameters
    ----------
    year
    scenario: basic feature (flexible) / full
    '''
        
    os.chdir('/mnt/phd/jihu/opioid')

    ###############################################################################
    ###############################################################################
    ###############################################################################

    FULL = pd.read_csv('Data/FULL_' + str(year) + '_LONGTERM_INPUT.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int})

    FULL = FULL.fillna(0)
        
    ## Drop the columns that start with alert
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
    FULL = FULL.drop(columns = ['drug_payment'])
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    if scenario == 'flexible':
        x_all = FULL[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
        
        cutoffs_i = []
        for column_name in ['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_i.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_i.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_i.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_i.append([n for n in range(0, 200) if n % 10 == 0])
        
        cutoffs_s = []
        for column_name in ['num_prescribers', 'num_pharmacies', 'concurrent_benzo']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_s.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_s.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_s.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_s.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):
            FULL_fold = FULL_splited[i]
            x = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                          'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
            
            x_i = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']]
            x_s = FULL_fold[['num_prescribers', 'num_pharmacies', 'concurrent_benzo']]
            
            x_intervals = stumps.create_intervals(x_i.values, x_i.columns, cutoffs_i)
            x_stumps = stumps.create_stumps(x_s.values, x_s.columns, cutoffs_s)
            
            new_data = pd.concat([x_intervals.reset_index(drop=True), x_stumps.reset_index(drop=True)], axis = 1)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)  
    
    '''
    elif scenario == 'full':
        x_all = FULL[['age', 'quantity', 'quantity_per_day', 'total_dose', 
                      'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                      'concurrent_benzo_same', 'concurrent_benzo_diff', 
                      'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                      'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                      'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
        
        cutoffs = []
        for column_name in x_all.columns:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo' or column_name == 'concurrent_benzo_same' or \
                column_name == 'concurrent_benzo_diff' or column_name == 'num_presc':
                cutoffs.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME' or \
                column_name == 'days_diff':
                cutoffs.append([n for n in range(0, 90) if n % 10 == 0])
            elif column_name == 'dose_diff' or column_name == 'concurrent_MME_diff':
                cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
            elif column_name == 'age':
                cutoffs.append([n for n in range(20, 80) if n % 10 == 0])
            else:
                cutoffs.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):
            FULL_fold = FULL_splited[i]
            x = FULL_fold[['age', 'quantity', 'quantity_per_day', 'total_dose', 
                           'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                           'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                           'concurrent_benzo_same', 'concurrent_benzo_diff', 
                           'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                           'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                           'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME']]
            
            x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
            # bind the data with the ones that was not for create_stumps
            x_rest = FULL_fold[FULL_fold.columns.drop(['age', 'quantity', 'quantity_per_day', 'total_dose', 
                                                       'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                                                       'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                                                       'concurrent_benzo_same', 'concurrent_benzo_diff', 
                                                       'num_presc', 'dose_diff', 'MME_diff', 'days_diff',
                                                       'Codeine_MME', 'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 
                                                       'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 'Oxymorphone_MME'])]
            
            new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
            print(new_data.shape)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)         
    
    else:
        print('Scenario cannot be identified')

    '''


    







  
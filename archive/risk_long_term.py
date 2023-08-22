#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 3 2023

@Author: Jingyuan Hu
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
average_precision_score, brier_score_loss, fbeta_score, accuracy_score

import utils.stumps as stumps
import pprint
import riskslim
import utils.RiskSLIM as slim
from riskslim.utils import print_model

os.chdir('/mnt/phd/jihu/opioid')


###########################################################################

## Y
SAMPLE = pd.read_csv('Data/FULL_2018_INPUT_LONGTERM_NOILLICIT.csv', delimiter = ",")
SAMPLE = SAMPLE[['long_term_180']]
SAMPLE = SAMPLE.fillna(0)

## X
N = 20
SAMPLE_STUMPS = pd.read_csv('Data/FULL_2018_ALERT_STUMPS' + str(0) + '_LONGTERM_NOILLICIT.csv', delimiter = ",")
for i in range(1, N):
    TEMP = pd.read_csv('Data/FULL_2018_ALERT_STUMPS' + str(i) + '_LONGTERM_NOILLICIT.csv', delimiter = ",")
    SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

## Keep the basic feautres only
basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                  'num_prescribers', 'num_pharmacies',\
                                                                      'consecutive_days', 'concurrent_benzo'))]
SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]

## Drop the dummy cutoffs
SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                          'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                          'num_prescribers1','num_pharmacies1'])]

SAMPLE_STUMPS['(Intercept)'] = 1
intercept = SAMPLE_STUMPS.pop('(Intercept)')
SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
x = SAMPLE_STUMPS

### Regular CV with operational constraints
## Operational constraints
# selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
#                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
#                      'dose_diff', 'MME_diff', 'days_diff']
# selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
#                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days']

# new_constraints = []
# for feature in selected_features:
#     new_constraints.append([col for col in x if col.startswith(feature)])

## Balanced riskSLIM
outcome_list = ['long_term_180']
for outcome in outcome_list:
    y = SAMPLE[[outcome]].to_numpy().astype('int')    
    y[y==0]= -1
    
    ###########################################################################
    ############### Use all data to create a single output table ##############
    ###########################################################################
    
    '''
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
    
    print("Start training single table for " + str(outcome))
    start = time.time()
    
    model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                               max_coefficient=5, 
                                                               max_L0_value=6, 
                                                               c0_value=1e-3, 
                                                               max_runtime=1000, 
                                                               max_offset=100,
                                                               class_weight='balanced',
                                                               new_constraints=new_constraints)
    print_model(model_info['solution'], new_train_data)
    print(str(round(time.time() - start,1)) + ' seconds')
    
    ## change data format
    outer_train_x = outer_train_x[:,1:]
    outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
    
    ## train results
    outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
    outer_train_pred = (outer_train_prob > 0.5)
    
    balanced = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}

    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Long term'])

    riskslim_results_type = balanced.T
    riskslim_results_type.to_csv('../../../Result/result_longterm_single.csv')
    '''
    
    ###########################################################################
    ####################### Nested Cross Validation  ##########################
    ###########################################################################
    
    print("Start training nested " + str(outcome))
    start = time.time()
    
    risk_summary_balanced = slim.risk_nested_cv_constrain(X=x,
                                                          Y=y,
                                                          y_label='long_term_180', 
                                                          max_coef=10, 
                                                          max_coef_number=20,
                                                          # new_constraints=new_constraints,
                                                          c=[1e-4, 1e-3], # greater regularization term
                                                          class_weight = 'balanced',
                                                          seed=42)
    
    end = time.time()
    print(str(round(end - start,1)) + ' seconds')

    balanced = {"Accuracy": str(round(np.mean(risk_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(risk_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(risk_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(risk_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(risk_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_pr_auc']), 4)) + ")"}

    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Long term'])
    riskslim_results_type = balanced.T
    riskslim_results_type.to_csv('../../../Result/Longterm/result_basicfeature_flexible_noconstraint.csv')
    
    ###########################################################################
    #################### Nested Cross Validation (Bag) ########################
    ###########################################################################
    '''
    print("Start training nested " + str(outcome))
    start = time.time()
    
    risk_summary_balanced = slim.risk_nested_cv_constrain_bagging(X=x,
                                                                  Y=y,
                                                                  y_label='long_term_180', 
                                                                  max_coef=5, 
                                                                  max_coef_number=6,
                                                                  new_constraints=new_constraints,
                                                                  c=[1e-4], # greater regularization term
                                                                  class_weight = 'balanced',
                                                                  seed=42)
    
    end = time.time()
    print(str(round(end - start,1)) + ' seconds')

    balanced = {"Accuracy": str(round(np.mean(risk_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(risk_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(risk_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(risk_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(risk_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Accuracy (Bag)": str(round(risk_summary_balanced['bag_accuracy'], 4)),
               "Recall (Bag)": str(round(risk_summary_balanced['bag_recall'], 4)),
               "Precision (Bag)": str(round(risk_summary_balanced['bag_precision'], 4)),
               "ROC AUC (Bag)": str(round(risk_summary_balanced['bag_roc_auc'], 4)),
               "PR AUC (Bag)": str(round(risk_summary_balanced['bag_pr_auc'], 4))}

    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Long term'])
    riskslim_results_type = balanced.T
    riskslim_results_type.to_csv('../../../Result/result_longterm_nested_bag_newtest.csv')
    '''

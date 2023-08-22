#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 3 2023

@Author: Jingyuan Hu

All features, under different weights
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
average_precision_score, brier_score_loss, fbeta_score, accuracy_score, roc_curve

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

SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                          'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                          'num_prescribers1','num_pharmacies1'])]
SAMPLE_STUMPS['(Intercept)'] = 1
intercept = SAMPLE_STUMPS.pop('(Intercept)')
SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
x = SAMPLE_STUMPS

## Operational constraints
selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                     'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
                     'dose_diff', 'MME_diff', 'days_diff']
new_constraints = []
for feature in selected_features:
    new_constraints.append([col for col in x if col.startswith(feature)])


###########################################################################
############### Use all data to create a single output table ##############
###########################################################################

'''
weight_list = ['original', 'balanced', 'positive', 'positive_2', 'positive_4']

for weight in weight_list:
    
    ## Y
    y = SAMPLE[['long_term_180']].to_numpy().astype('int')    
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
    
    print("Start training single table under weigths " + weight)
    start = time.time()    
    model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                               max_coefficient=5, 
                                                               max_L0_value=6, 
                                                               c0_value=1e-3, 
                                                               max_runtime=1000, 
                                                               max_offset=100,
                                                               class_weight=weight,
                                                               new_constraints=new_constraints)
    print_model(model_info['solution'], new_train_data)
    print(str(round(time.time() - start,1)) + ' seconds')

    ## change data format
    outer_train_x = outer_train_x[:,1:]
    outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0

    outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
    outer_train_pred = (outer_train_prob > 0.5)

    original = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
                "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
                "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
                "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
                "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}

    original = pd.DataFrame.from_dict(original, orient='index', columns=[weight])
    riskslim_results_type = original.T
    riskslim_results_type.to_csv('../../../Result/Longterm/result_longterm_' + weight + '.csv')
    
    ## Save FPR, TPR for ROC plot
    fpr, tpr, _ = roc_curve(outer_train_y, outer_train_prob)
    
    np.savetxt('../../../Result/Longterm/ROC/' + weight + '_fpr.csv', fpr, delimiter = ",")
    np.savetxt('../../../Result/Longterm/ROC/' + weight + '_tpr.csv', tpr, delimiter = ",")
'''

###########################################################################
##################### Use 2019 data as test data ##########################
###########################################################################

## Y
SAMPLE_2019 = pd.read_csv('Data/FULL_2019_INPUT_LONGTERM.csv', delimiter = ",")
SAMPLE_2019 = SAMPLE_2019[['long_term_180']]
SAMPLE_2019 = SAMPLE_2019.fillna(0)
y_2019 = SAMPLE_2019[['long_term_180']].to_numpy().astype('int')

## X
N = 20
SAMPLE_STUMPS_2019 = pd.read_csv('Data/FULL_2019_ALERT_STUMPS' + str(0) + '_LONGTERM.csv', delimiter = ",")
for i in range(1, N):
    TEMP = pd.read_csv('Data/FULL_2019_ALERT_STUMPS' + str(i) + '_LONGTERM.csv', delimiter = ",")
    SAMPLE_STUMPS_2019 = pd.concat([SAMPLE_STUMPS_2019, TEMP])

SAMPLE_STUMPS_2019 = SAMPLE_STUMPS_2019[SAMPLE_STUMPS_2019.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                         'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                         'num_prescribers1','num_pharmacies1'])]
SAMPLE_STUMPS_2019['(Intercept)'] = 1
intercept = SAMPLE_STUMPS_2019.pop('(Intercept)')
SAMPLE_STUMPS_2019.insert(0, '(Intercept)', intercept)
x_2019 = SAMPLE_STUMPS_2019

###########################################################################
    
y = SAMPLE[['long_term_180']].to_numpy().astype('int')    
y[y==0]= -1

cols = x.columns.tolist() 
outer_train_sample_weight = np.repeat(1, len(y))
outer_train_x, outer_train_y = x.values, y.reshape(-1,1)
outer_test_x, outer_test_y = x_2019.values, y_2019.reshape(-1,1)
new_train_data = {
    'X': outer_train_x,
    'Y': outer_train_y,
    'variable_names': cols,
    'outcome_name': 'long_term_180',
    'sample_weights': outer_train_sample_weight
}

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

## change data format

outer_train_x = outer_train_x[:,1:]
outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
outer_train_pred = (outer_train_prob > 0.5)

outer_test_x = outer_test_x[:,1:]
outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
outer_test_prob = slim.riskslim_prediction(outer_test_x, np.array(cols), model_info).reshape(-1,1)
outer_test_pred = (outer_test_prob > 0.5)

train_results = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
            "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
            "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
            "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
            "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['2018'])

test_results = {"Accuracy": str(round(accuracy_score(outer_test_y, outer_test_pred), 4)),
            "Recall": str(round(recall_score(outer_test_y, outer_test_pred), 4)),
            "Precision": str(round(precision_score(outer_test_y, outer_test_pred), 4)),
            "ROC AUC": str(round(roc_auc_score(outer_test_y, outer_test_prob), 4)),
            "PR AUC": str(round(average_precision_score(outer_test_y, outer_test_prob), 4))}
test_results = pd.DataFrame.from_dict(test_results, orient='index', columns=['2019'])


results = pd.concat([train_results, test_results], axis = 1)
results = results.T
results.to_csv('../../../Result/Longterm/train_test20182019.csv')

    

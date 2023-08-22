#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 4 2023

@Author: Jingyuan Hu

Test the model using 2019 data
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
##################### Use 2018 data as train data #########################
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

###########################################################################
## Keep the basic feautres only
basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                  'num_prescribers', 'num_pharmacies',\
                                                                      'consecutive_days', 'concurrent_benzo'))]
SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]

###########################################################################

SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                          'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                          'num_prescribers1','num_pharmacies1'])]
SAMPLE_STUMPS['(Intercept)'] = 1
intercept = SAMPLE_STUMPS.pop('(Intercept)')
SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
x = SAMPLE_STUMPS

## Operational constraints
selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days']

# selected_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
#                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
#                      'dose_diff', 'MME_diff', 'days_diff']
new_constraints = []
for feature in selected_features:
    new_constraints.append([col for col in x if col.startswith(feature)])


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

###########################################################################
SAMPLE_STUMPS_2019 = SAMPLE_STUMPS_2019[basic_features]
###########################################################################

SAMPLE_STUMPS_2019 = SAMPLE_STUMPS_2019[SAMPLE_STUMPS_2019.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                                         'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                                         'num_prescribers1','num_pharmacies1'])]
SAMPLE_STUMPS_2019['(Intercept)'] = 1
intercept = SAMPLE_STUMPS_2019.pop('(Intercept)')
SAMPLE_STUMPS_2019.insert(0, '(Intercept)', intercept)
x_2019 = SAMPLE_STUMPS_2019

###########################################################################
###########################################################################
###########################################################################


'''
### 2018
SAMPLE = pd.read_csv('Data/FULL_2018_INPUT_LONGTERM_NOILLICIT.csv', delimiter = ",", 
                     dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                            'num_prescribers': int, 'num_pharmacies': int,
                            'concurrent_benzo': int, 'consecutive_days': int,
                            'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
SAMPLE = SAMPLE.fillna(0)
SAMPLE['(Intercept)'] = 1
intercept = SAMPLE.pop('(Intercept)')
SAMPLE.insert(0, '(Intercept)', intercept)

x = SAMPLE[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']]


### 2019
SAMPLE_2019 = pd.read_csv('Data/FULL_2019_INPUT_LONGTERM.csv', delimiter = ",", 
                     dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                            'num_prescribers': int, 'num_pharmacies': int,
                            'concurrent_benzo': int, 'consecutive_days': int,
                            'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
SAMPLE_2019 = SAMPLE_2019.fillna(0)
SAMPLE_2019['(Intercept)'] = 1
intercept = SAMPLE_2019.pop('(Intercept)')
SAMPLE_2019.insert(0, '(Intercept)', intercept)

x_2019 = SAMPLE_2019[['(Intercept)', 'alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6']]
'''

###########################################################################
###########################################################################
###########################################################################


y = SAMPLE[['long_term_180']].to_numpy().astype('int')    
y[y==0]= -1

y_2019 = SAMPLE_2019[['long_term_180']].to_numpy().astype('int')
y_2019[y_2019==0]= -1


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
                                                            c0_value=1e-5, 
                                                            max_runtime=1000, 
                                                            max_offset=100,
                                                            class_weight='balanced',
                                                            new_constraints=new_constraints)
# model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
#                                                            max_coefficient=5, 
#                                                            max_L0_value=6, 
#                                                            c0_value=1e-4, 
#                                                            max_runtime=1000, 
#                                                            max_offset=100,
#                                                            class_weight='balanced')
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
results.to_csv('../../../Result/Longterm/train_test_flexible_new.csv')

    
###########################################################################
###########################################################################
###########################################################################
'''
print("Start training 2018")

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

outer_train_x = outer_train_x[:,1:]
outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
outer_train_pred = (outer_train_prob > 0.5)

results_2018 = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
            "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
            "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
            "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
            "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
results_2018 = pd.DataFrame.from_dict(results_2018, orient='index', columns=['2018'])
'''
###########################################################################

print("Start training 2019")

y_2019 = SAMPLE_2019[['long_term_180']].to_numpy().astype('int')
y_2019[y_2019==0]= -1

cols = x_2019.columns.tolist() 
outer_train_sample_weight = np.repeat(1, len(y_2019))
outer_train_x, outer_train_y = x_2019.values, y_2019.reshape(-1,1)
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
                                                            c0_value=1e-5, 
                                                            max_runtime=1000, 
                                                            max_offset=100,
                                                            class_weight='balanced',
                                                            new_constraints=new_constraints)

print_model(model_info['solution'], new_train_data)
print(str(round(time.time() - start,1)) + ' seconds')

outer_train_x = outer_train_x[:,1:]
outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
outer_train_pred = (outer_train_prob > 0.5)

results_2019 = {"Accuracy": str(round(accuracy_score(outer_train_y, outer_train_pred), 4)),
            "Recall": str(round(recall_score(outer_train_y, outer_train_pred), 4)),
            "Precision": str(round(precision_score(outer_train_y, outer_train_pred), 4)),
            "ROC AUC": str(round(roc_auc_score(outer_train_y, outer_train_prob), 4)),
            "PR AUC": str(round(average_precision_score(outer_train_y, outer_train_prob), 4))}
results_2019 = pd.DataFrame.from_dict(results_2019, orient='index', columns=['2019'])

# results = pd.concat([results_2018, results_2019], axis = 1)
results = results_2019
results = results.T
# results.to_csv('../../../Result/Longterm/2018_2019_new.csv')
results.to_csv('../../../Result/Longterm/2019_new.csv')



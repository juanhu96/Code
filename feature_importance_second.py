#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 23 2022

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
average_precision_score, brier_score_loss, fbeta_score, accuracy_score

import utils.stumps as stumps
import pprint
import riskslim
import utils.RiskSLIM as slim
from riskslim.utils import print_model

os.chdir('/mnt/phd/jihu/opioid')
year = 2018
outcome = 'long_term_180'

###############################################################################
###############################################################################

## Y
SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM.csv', delimiter = ",", 
                     dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                            'num_prescribers': int, 'num_pharmacies': int,
                            'concurrent_benzo': int, 'consecutive_days': int,
                            'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
SAMPLE = SAMPLE.fillna(0)

## X
N = 20
SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(0) + '.csv', delimiter = ",")
for i in range(1, N):
    TEMP = pd.read_csv('Data/FULL_' + str(year) + '_STUMPS' + str(i) + '.csv', delimiter = ",")
    SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

## Keep the basic feautres only
basic_features = [col for col in SAMPLE_STUMPS if col.startswith(('concurrent_MME', 'concurrent_methadone_MME',\
                                                                  'num_prescribers', 'num_pharmacies',\
                                                                      'consecutive_days', 'concurrent_benzo'))]
SAMPLE_STUMPS = SAMPLE_STUMPS[basic_features]
SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                          'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                          'num_prescribers1','num_pharmacies1'])]
SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('concurrent_benzo_same', 'concurrent_benzo_diff'))])]
SAMPLE_STUMPS['(Intercept)'] = 1
intercept = SAMPLE_STUMPS.pop('(Intercept)')
SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
x = SAMPLE_STUMPS

###############################################################################
########################### Feature Importance ################################
###############################################################################    

selected_features = ['num_prescribers', 'num_pharmacies']

for feature in selected_features:
    # throw away one feature at a time
    print('Dropping feature ' + feature)
    
    x = SAMPLE_STUMPS.copy() # avoid modifying the original dataframe
    x = x[x.columns.drop([col for col in x if col.startswith(feature)])] # drop the feature
    
    y = SAMPLE[[outcome]].to_numpy().astype('int')    
    y[y==0]= -1
    
    all_features = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                    'num_pharmacies', 'concurrent_benzo', 'consecutive_days']
    all_features.remove(feature)
    new_constraints = []
    for temp in all_features:
        new_constraints.append([col for col in x if col.startswith(temp)])
           
    print("Start training without ...")
    start = time.time()
    risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                  Y=y,
                                                  y_label=outcome, 
                                                  max_coef=5, 
                                                  max_coef_number=6,
                                                  new_constraints=new_constraints,
                                                  c=[1e-4],
                                                  class_weight='balanced',
                                                  seed=42)

    end = time.time()
    print(str(round(end - start,1)) + ' seconds')    

    results_feature = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
                        "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
                        "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
                        "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
                        "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

    results_feature = pd.DataFrame.from_dict(results_feature, orient='index', columns=[feature])
    results_feature = results_feature.T
    os.chdir('/mnt/phd/jihu/opioid')
    results_feature.to_csv('Result/feature_importance_' + feature + '.csv')




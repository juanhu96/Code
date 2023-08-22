#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 7 2022

@author: Jingyuan Hu
Uses the current alert only
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

########################################
################## 31 ##################
########################################

# y = SAMPLE.to_numpy()[:,[-4]].astype('int') # alert31 within
# y = SAMPLE.to_numpy()[:,[-7]].astype('int') # alert31 exact
y = SAMPLE[['long_term_180']].to_numpy().astype('int')    
y[y==0]= -1

## Balanced riskSLIM
print("Start training ...")
start = time.time()
risk_summary_balanced = slim.risk_nested_cv_constrain(X=x,
                                                      Y=y,
                                                      y_label='long_term_180', 
                                                      max_coef=6, 
                                                      max_coef_number=5,
                                                      c=[1e-5, 1e-4, 1e-3],
                                                      class_weight = 'balanced',
                                                      seed=42)
end = time.time()
print(str(round(end - start,1)) + ' seconds')

### Balanced only
balanced = {"Accuracy": str(round(np.mean(risk_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_accuracy']), 4)) + ")",
            "Recall": str(round(np.mean(risk_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_recall']), 4)) + ")",
            "Precision": str(round(np.mean(risk_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_precision']), 4)) + ")",
            "ROC AUC": str(round(np.mean(risk_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
            "PR AUC": str(round(np.mean(risk_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
            "Brier": str(round(np.mean(risk_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_brier']), 4)) + ")",
            "F2": str(round(np.mean(risk_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_f2']), 4)) + ")"}

balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Risk SLIM'])
riskslim_results = balanced.T
riskslim_results.to_csv('../../../Result/Longterm/result_base.csv')

'''
########################################
################## 62 ##################
########################################

# y = SAMPLE.to_numpy()[:,[-3]].astype('int') # alert62 within
y = SAMPLE.to_numpy()[:,[-6]].astype('int') # alert62 exact
y[y==0]= -1

## Balanced riskSLIM
print("Start training base 62")
start = time.time()
risk_summary_balanced = slim.risk_cv_constrain(X=x,
                                                Y=y,
                                                y_label='high_risk', 
                                                max_coef=6, 
                                                max_coef_number=5,
                                                c=1e-6,
                                                class_weight = 'balanced',
                                                seed=42)
end = time.time()
print(str(round(end - start,1)) + ' seconds')

### Balanced only
balanced = {"Accuracy": str(round(np.mean(risk_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_accuracy']), 4)) + ")",
            "Recall": str(round(np.mean(risk_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_recall']), 4)) + ")",
            "Precision": str(round(np.mean(risk_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_precision']), 4)) + ")",
            "ROC AUC": str(round(np.mean(risk_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
            "PR AUC": str(round(np.mean(risk_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
            "Brier": str(round(np.mean(risk_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_brier']), 4)) + ")",
            "F2": str(round(np.mean(risk_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_f2']), 4)) + ")"}

balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Risk SLIM'])
riskslim_results = balanced.T
riskslim_results.to_csv('../../../Result/result_base_62exact.csv')



########################################
################## 93 ##################
########################################

# y = SAMPLE.to_numpy()[:,[-2]].astype('int') # alert93 within
y = SAMPLE.to_numpy()[:,[-5]].astype('int') # alert93 exact
y[y==0]= -1

## Balanced riskSLIM
print("Start training base 93")
start = time.time()
risk_summary_balanced = slim.risk_cv_constrain(X=x,
                                                Y=y,
                                                y_label='high_risk', 
                                                max_coef=6, 
                                                max_coef_number=5,
                                                c=1e-6,
                                                class_weight = 'balanced',
                                                seed=42)
end = time.time()
print(str(round(end - start,1)) + ' seconds')

### Balanced only
balanced = {"Accuracy": str(round(np.mean(risk_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_accuracy']), 4)) + ")",
            "Recall": str(round(np.mean(risk_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_recall']), 4)) + ")",
            "Precision": str(round(np.mean(risk_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_precision']), 4)) + ")",
            "ROC AUC": str(round(np.mean(risk_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
            "PR AUC": str(round(np.mean(risk_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
            "Brier": str(round(np.mean(risk_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_brier']), 4)) + ")",
            "F2": str(round(np.mean(risk_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(risk_summary_balanced['holdout_test_f2']), 4)) + ")"}

balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Risk SLIM'])
riskslim_results = balanced.T
riskslim_results.to_csv('../../../Result/result_base_93exact.csv')


'''



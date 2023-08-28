#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:57:11 2022

@author: jingyuanhu
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
import utils.baseline_functions as base
os.chdir('/mnt/phd/jihu/opioid')

with parallel_backend('threading', n_jobs=40):
    SAMPLE = pd.read_csv('Data/FULL_2018_INPUT_LONGTERM_NOILLICIT.csv', delimiter = ",")
    
    # x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
    #             'num_pharmacies', 'concurrent_benzo', 'consecutive_days',
    #             'dose_diff', 'MME_diff', 'days_diff']]
    # x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
    #             'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
    
    
    
    ###########################################################################
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
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['concurrent_MME0', 'concurrent_methadone_MME0', 'num_prescribers0',
                                                              'num_pharmacies0', 'concurrent_benzo0', 'consecutive_days0',
                                                              'num_prescribers1','num_pharmacies1'])]

    SAMPLE_STUMPS['(Intercept)'] = 1
    intercept = SAMPLE_STUMPS.pop('(Intercept)')
    SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    x = SAMPLE_STUMPS
    ###########################################################################
    
    y = SAMPLE['long_term_180'].values
    
    start = time.time()
    
    # Decision Tree
    '''
    depth = [3,4,5,6]
    min_samples = [5,10]
    impurity = [0.001,0.01,0.1]
    dt_summary_balanced = base.DecisionTree(X=x, Y=y, 
                                   depth=depth, 
                                   min_samples=min_samples, 
                                   impurity=impurity,
                                   class_weight="balanced",
                                   seed=42)
    
    
    
    # L2 logistic
    c = np.linspace(1e-4,1,5).tolist()
    logistic_summary_balanced = base.Logistic(X=x, Y=y, C=c, class_weight="balanced", seed=42)
    '''
    
    # L1 logistic
    c = np.linspace(1e-5,1e-3,3).tolist()
    lasso_summary_balanced = base.Lasso(X=x, Y=y, C=c, class_weight="balanced", seed=42)
    
    '''
    # LinearSVM
    c = np.linspace(1e-4,1,5).tolist()
    svm_summary_balanced = base.LinearSVM(X=x, Y=y, C=c, class_weight="balanced", seed=42)
    
    # Random Forest 
    depth = [3,4,5,6]
    n_estimators = [50,100,200]
    impurity = [0.001,0.01]
    rf_summary_balanced = base.RF(X=x, Y=y,
                                  depth=depth,
                                  estimators=n_estimators,
                                  impurity=impurity,
                                  class_weight="balanced",
                                  seed=42)
    
    # XGBoost
    depth = [3,4,5,6]
    n_estimators =  [50,100]
    gamma = [5,10,15]
    child_weight = [5,10,15]
    xgb_summary_balanced = base.XGB(X=x, Y=y,
                                    depth=depth,
                                    estimators=n_estimators,
                                    gamma=gamma,
                                    child_weight=child_weight,
                                    class_weight="balanced",
                                    seed=42)
    
    end = time.time()
    print(str(round(end - start,1)) + ' seconds')
    
    
    # DT
    balanced = {"Accuracy": str(round(np.mean(dt_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(dt_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(dt_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(dt_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(dt_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(dt_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(dt_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(dt_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Decision Tree'])
    dt_results = balanced
    '''    
    
    # L2
    balanced = {"Accuracy": str(round(np.mean(logistic_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(logistic_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(logistic_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(logistic_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(logistic_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(logistic_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(logistic_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(logistic_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Logistic (L2)'])
    logistic_results = balanced
    
    # L1
    balanced = {"Accuracy": str(round(np.mean(lasso_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(lasso_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(lasso_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(lasso_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(lasso_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(lasso_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(lasso_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(lasso_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Logistic (L1)'])
    lasso_results = balanced
    
    '''
    # SVM
    balanced = {"Accuracy": str(round(np.mean(svm_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(svm_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(svm_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(svm_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(svm_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(svm_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(svm_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(svm_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['SVM'])
    svm_results = balanced
    
    # RF
    balanced = {"Accuracy": str(round(np.mean(rf_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(rf_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(rf_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(rf_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(rf_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(rf_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(rf_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(rf_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['Random Forest'])
    rf_results = balanced
    
    # XGB
    balanced = {"Accuracy": str(round(np.mean(xgb_summary_balanced['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_accuracy']), 4)) + ")",
               "Recall": str(round(np.mean(xgb_summary_balanced['holdout_test_recall']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_recall']), 4)) + ")",
               "Precision": str(round(np.mean(xgb_summary_balanced['holdout_test_precision']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_precision']), 4)) + ")",
               "ROC AUC": str(round(np.mean(xgb_summary_balanced['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_roc_auc']), 4)) + ")",
               "PR AUC": str(round(np.mean(xgb_summary_balanced['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_pr_auc']), 4)) + ")",
               "Brier": str(round(np.mean(xgb_summary_balanced['holdout_test_brier']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_brier']), 4)) + ")",
               "F2": str(round(np.mean(xgb_summary_balanced['holdout_test_f2']), 4)) + " (" + str(round(np.std(xgb_summary_balanced['holdout_test_f2']), 4)) + ")"}
    
    balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=['XGB'])
    xgb_results = balanced
    
    results = pd.concat([dt_results, logistic_results], axis=1)
    results = pd.concat([results, lasso_results], axis=1)
    results = pd.concat([results, svm_results], axis=1)
    results = pd.concat([results, rf_results], axis=1)
    results = pd.concat([results, xgb_results], axis=1)
    results = results.T
    results.to_csv("Result/Longterm/baseline_models.csv")
    '''
    
    results = pd.concat([logistic_results, lasso_results], axis = 1)
    results = results.T
    results.to_csv("Result/Longterm/logistic_results.csv")

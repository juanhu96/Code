#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 22 2023
Baseline comparisons
"""

import time
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import utils.baseline_functions as base

def baseline_main(year, Model_list=['Decision Tree', 'Logistic (L2)', 'Logistic (L1)', 'SVM', 'Random Forest', 'XGB'], class_weight=None, workdir='/mnt/phd/jihu/opioid/'):

    SAMPLE = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    N = 20
    SAMPLE_STUMPS = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST0.csv', delimiter = ",")
    for i in range(1, N):
        TEMP = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_STUMPS_UPTOFIRST{str(i)}.csv', delimiter = ",")
        SAMPLE_STUMPS = pd.concat([SAMPLE_STUMPS, TEMP])

    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('consecutive_days', 'concurrent_methadone_MME', 'quantity'))])]
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['avgDays60'])]
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([f'num_prescribers{i}' for i in range(4, 11)])]
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([f'num_pharmacies{i}' for i in range(4, 11)])]
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop(['CommercialIns', 'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation'])]

    SAMPLE_STUMPS['(Intercept)'] = 1
    intercept = SAMPLE_STUMPS.pop('(Intercept)')
    SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    x = SAMPLE_STUMPS
    y = SAMPLE['long_term_180'].values

    print(SAMPLE_STUMPS.columns.tolist()) # so that we know the order of features
    
    start = time.time()
    
    results = pd.DataFrame()
    for model in Model_list:
        
        if model == 'Decision Tree':

            depth = [3,4,5,6]
            min_samples = [5,10,20]
            impurity = [0.001,0.01,0.1]
            summary = base.DecisionTree(X=x, Y=y, depth=depth, min_samples=min_samples, impurity=impurity, class_weight=class_weight, seed=42)

        if model == 'Logistic (L2)':

            # c = np.linspace(1e-5,1,5).tolist()
            c = [1e-5, 1e-3, 1e-1, 10]
            summary = base.Logistic(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

        if model == 'Logistic (L1)':

            # c = np.linspace(1e-5,1,5).tolist()
            c = [1e-5, 1e-3, 1e-1, 10]
            summary = base.Lasso(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

        if model == 'SVM':

            c = np.linspace(1e-6,1e-2,5).tolist()
            summary = base.LinearSVM(X=x, Y=y, C=c, class_weight=class_weight, seed=42)

        if model == 'Random Forest':
            depth = [3,4,5,6]
            n_estimators = [50,100,200]
            impurity = [0.001,0.01]
            summary = base.RF(X=x, Y=y, depth=depth, estimators=n_estimators, impurity=impurity, class_weight=class_weight, seed=42)

        if model == 'XGB':
            depth = [4,5,6]
            n_estimators =  [50,100]
            gamma = [5,10]
            child_weight = [5,10]
            summary = base.XGB(X=x, Y=y, depth=depth, estimators=n_estimators, gamma=gamma, child_weight=child_weight, class_weight=class_weight, seed=42)

        
        balanced = {"Accuracy": str(round(np.mean(summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(summary['holdout_test_accuracy']), 4)) + ")",
        "Recall": str(round(np.mean(summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(summary['holdout_test_recall']), 4)) + ")",
        "Precision": str(round(np.mean(summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(summary['holdout_test_precision']), 4)) + ")",
        "ROC AUC": str(round(np.mean(summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(summary['holdout_test_roc_auc']), 4)) + ")",
        "PR AUC": str(round(np.mean(summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(summary['holdout_test_pr_auc']), 4)) + ")",
        "Brier": str(round(np.mean(summary['holdout_test_brier']), 4)) + " (" + str(round(np.std(summary['holdout_test_brier']), 4)) + ")",
        "F2": str(round(np.mean(summary['holdout_test_f2']), 4)) + " (" + str(round(np.std(summary['holdout_test_f2']), 4)) + ")"}
        
        balanced = pd.DataFrame.from_dict(balanced, orient='index', columns=[model])
        results = pd.concat([results, balanced], axis=1)

    results = results.T
    results.to_csv(f"{workdir}Result/baseline_results_SVM.csv")

    end = time.time()
    print(str(round(end - start,1)) + ' seconds')
    
    
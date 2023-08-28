#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 19 2023

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

import statsmodels.api as sm

os.chdir('/mnt/phd/jihu/opioid')
year = 2018

with parallel_backend('threading', n_jobs=40):
    
    SAMPLE = pd.read_csv('Data/FULL_' + str(year) +'_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)
    
    ###########################################################################
    
    SAMPLE_STUMPS = pd.read_csv('Data/FULL_' + str(year) + '_LONGTERM_INPUT_UPTOFIRST.csv', delimiter = ",")
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('long_term_180', 'days_supply', 'daily_dose', 'quantity',
                                                                                                              'quantity_per_day', 'total_dose', 'dose_diff', 'drug_payment',
                                                                                                              'concurrent_benzo_same', 'concurrent_benzo_diff'))])]
    
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
    
    SAMPLE_STUMPS = SAMPLE_STUMPS[SAMPLE_STUMPS.columns.drop([col for col in SAMPLE_STUMPS if col.startswith(('Codeine_MME', 'Hydrocodone_MME',
                                                                                                              'Oxycodone_MME', 'Morphine_MME', 
                                                                                                              'Hydromorphone_MME', 'Methadone_MME',
                                                                                                              'Fentanyl_MME', 'Oxymorphone_MME'))])]
        
    SAMPLE_STUMPS['(Intercept)'] = 1
    intercept = SAMPLE_STUMPS.pop('(Intercept)')
    SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    x = SAMPLE_STUMPS
    
    print(SAMPLE_STUMPS.columns.tolist()) # so that we know the order of features
    
    ###########################################################################
    
    y = SAMPLE['long_term_180'].values
    
    '''
    # L2 logistic
    c = np.linspace(1e-5,1e-3,5).tolist()
    logistic_summary_balanced = base.Logistic(X=x, Y=y, C=c, class_weight="balanced", seed=42)
    
    # L1 logistic
    c = np.linspace(1e-5,1e-3,5).tolist()
    lasso_summary_balanced = base.Lasso(X=x, Y=y, C=c, class_weight="balanced", seed=42)
    
    ###########################################################################
    
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
    
    results = pd.concat([logistic_results, lasso_results], axis=1)
    results = results.T
    results.to_csv("Result/logistic_results_raw.csv")
    '''
    
    ###########################################################################
    ###########################################################################
    
    # Create a DataFrame with the input variables and the binary outcome
    
    # Add constant term to the input variables
    # X = sm.add_constant(x[['X1', 'X2']])
    
    # Fit the Linear Probability Model
    lpm_model = sm.OLS(y, x).fit()
    
    # Print the model summary
    print(lpm_model.summary())
    
    ###########################################################################
    ###########################################################################
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023

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


import seaborn as sns
import matplotlib.pyplot as plt


def risk_train(year,
            features,
            scenario, 
            c, 
            weight='balanced',  
            max_points=5,
            max_features=6, 
            outcome='long_term_180', 
            output_y=False,
            roc=False,
            workdir='/mnt/phd/jihu/opioid/',
            name=''):
    
    '''
    Train a riskSLIM model
    
    
    Parameters
    ----------
    year: year of the training dataset
    features: base/flexible/full/selected
    scenario: single/nested
    c: has to be a list when doing nested CV
    weight: original/balanced/positive/positive_2/positive_4
    max_points: maximum point allowed per feature
    outcome: outcome to predict
    output_y: whether to export the predicted y
    name: index for filename (when running multiple trails)
    roc: export fpr, tpr for roc visualization (only for single)

    '''


    # ================================ Y INPUT ====================================

    SAMPLE = pd.read_csv(f'{workdir}Data/FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for drug_category in SAMPLE['drug'].unique():
    #     subset = SAMPLE[SAMPLE['drug'] == drug_category]
    #     sns.kdeplot(subset['concurrent_MME'], ax=ax, label=drug_category, bw_adjust=2)  # Adjust bw_adjust as needed

    # ax.set_xlim(0, 300)
    # ax.set_title('Distribution of Concurrent MME Grouped by Drug')
    # ax.set_xlabel('Concurrent MME')
    # ax.set_ylabel('Density')
    # ax.legend()
    # plt.show()
    # fig.savefig(f'/mnt/phd/jihu/opioid/Result/density_kde.pdf', dpi=300)


    # fig, ax = plt.subplots(figsize=(10, 6))
    # for drug_category in SAMPLE['drug'].unique():
    #     subset = SAMPLE[SAMPLE['drug'] == drug_category]
    #     sns.kdeplot(subset['quantity'], ax=ax, label=drug_category, bw_adjust=2)  # Adjust bw_adjust as needed

    # ax.set_xlim(0, 300)
    # ax.set_title('Distribution of Quantity Grouped by Drug')
    # ax.set_xlabel('Quantity')
    # ax.set_ylabel('Density')
    # ax.legend()
    # plt.show()
    # fig.savefig(f'/mnt/phd/jihu/opioid/Result/density_kde_quantity.pdf', dpi=300)

    # sns.histplot(data=SAMPLE, x='concurrent_MME', hue='drug', element='step', stat='density', common_norm=False)
    # plt.title('Distribution of Concurrent MME Grouped by Drug')
    # plt.xlabel('Concurrent MME')
    # plt.ylabel('Density')
    # plt.show()
    # fig.savefig(f'/mnt/phd/jihu/opioid/Result/density.pdf', dpi=300)
    # print("-"*100)

    # ================================ X INPUT ====================================

        
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

    # drug_payment = [['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'], 
    # ['Medicaid', 'CommercialIns', 'Medicare', 'CashCredit', 'MilitaryIns', 'WorkersComp', 'Other', 'IndianNation']]
    drug_payment = [['Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 'HMFO'], ['Medicaid', 'Medicare', 'CashCredit']]
    
    # Single cutoff
    single_cutoff_features = [['num_prior_presc', 'num_prescribers', 'num_pharmacies'], ['concurrent_MME']]
    # single_cutoff_features = [['avgDays'], ['num_prescribers'], ['num_pharmacies'], ['concurrent_MME'], ['quantity']]
    # single_cutoff_features = [['num_prior_presc'], ['num_prescribers'], ['num_pharmacies'], ['concurrent_MME'], ['quantity'], ['concurrent_benzo']]
    if single_cutoff_features: 
        single_cutoff = [[col for col in x if any(col.startswith(feature) for feature in sublist)] for sublist in single_cutoff_features]
        single_cutoff.extend(drug_payment)
    else: 
        single_cutoff = None
          

    # Two cutoffs
    two_cutoffs_features = [['avgDays']]
    if two_cutoffs_features: 
        two_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in sublist)] for sublist in two_cutoffs_features]
        # two_cutoffs.extend(drug_payment)
    else: two_cutoffs = None


    # Three cutoffs
    three_cutoffs_features = []
    if three_cutoffs_features: three_cutoffs = [[col for col in x if col.startswith(feature)] for feature in three_cutoffs_features]
    else: three_cutoffs = None


    # essential_cutoffs
    # essential_cutoffs_feautres = [['avgDays'], ['num_prior_presc', 'num_prescribers', 'num_pharmacies'], ['concurrent_MME', 'quantity']]
    essential_cutoffs_feautres = [['avgDays']] # NONE type not iterable
    if essential_cutoffs_feautres:
        essential_cutoffs = [[col for col in x if any(col.startswith(feature) for feature in sublist)] for sublist in essential_cutoffs_feautres]
        essential_cutoffs.extend(drug_payment)
    else: essential_cutoffs = None


    # columns_to_keep = ['concurrent_MME15', 'avgDays7', 'avgDays14', 'num_prior_presc2', 
    # 'concurrent_benzo1', 'HMFO', 'Medicare', 'Medicaid', 'CashCredit', 'switch_payment']
    # SAMPLE_STUMPS = SAMPLE_STUMPS[columns_to_keep]
    
    # SAMPLE_STUMPS['(Intercept)'] = 1
    # intercept = SAMPLE_STUMPS.pop('(Intercept)')
    # SAMPLE_STUMPS.insert(0, '(Intercept)', intercept)
    # x = SAMPLE_STUMPS

    # single_cutoff = None
    # two_cutoffs = None
    # three_cutoffs = None
    # four_cutoffs = None
    # essential_cutoffs = None


    # =============================== SCENARIOS ====================================

    y = SAMPLE[[outcome]].to_numpy().astype('int')    
    y[y==0]= -1


    if scenario == 'single':
        

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
        

        print(f"Start training {year}{scenario}{features}")
        start = time.time()    
        model_info, mip_info, lcpa_info = slim.risk_slim_constrain(new_train_data, 
                                                                   max_coefficient=max_points, 
                                                                   max_L0_value=max_features,
                                                                   c0_value=c, 
                                                                   max_runtime=1000, 
                                                                   max_offset=20,
                                                                   class_weight=weight,
                                                                   single_cutoff=single_cutoff,
                                                                   two_cutoffs=two_cutoffs,
                                                                   three_cutoffs=three_cutoffs,
                                                                   essential_cutoffs=essential_cutoffs)
        print_model(model_info['solution'], new_train_data)
        print(f"{round(time.time() - start, 1)} seconds")

        # print("\n Constraints:")
        # for i in range(mip_info['risk_slim_mip'].linear_constraints.get_num()):
        #     row = mip_info['risk_slim_mip'].linear_constraints.get_rows(i)
        #     senses = mip_info['risk_slim_mip'].linear_constraints.get_senses(i)
        #     rhs = mip_info['risk_slim_mip'].linear_constraints.get_rhs(i)
        #     names = mip_info['risk_slim_mip'].linear_constraints.get_names(i)
        #     print(f"Constraint {names}: {row}, {senses}, {rhs}")

        ## Results
        outer_train_x = outer_train_x[:,1:]
        outer_train_y[outer_train_y == -1] = 0
        outer_train_prob = slim.riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        
        export_results_single(outer_train_y, outer_train_pred, outer_train_prob, 
        filename = f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}.csv')
        

        if roc == True: compute_roc(outer_train_prob, outer_train_y, y_pred, year, features, scenario, weight, name)
        if output_y == True: np.savetxt(f'{workdir}Result/riskSLIM_y.csv', outer_train_pred, delimiter=",")


    elif scenario == 'cv':

        print("Start training " + str(year) + scenario + features)
        start = time.time()

        c, risk_summary = slim.risk_cv_constrain(X=x, Y=y,
                                                y_label=outcome, 
                                                max_coef=max_points,
                                                max_coef_number=max_features,
                                                c=c,
                                                max_offset=20,
                                                class_weight=weight, 
                                                single_cutoff=single_cutoff,
                                                two_cutoffs=two_cutoffs,
                                                three_cutoffs=three_cutoffs,
                                                essential_cutoffs=essential_cutoffs,
                                                max_runtime=1000,
                                                seed=42)

        end = time.time()
        print(str(round(end - start,1)) + ' seconds')    
        export_results_cv(risk_summary, f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}.csv') 

        return c


    elif scenario == 'nested':
        
        print("Start training " + str(year) + scenario + features)
        start = time.time()
        risk_summary = slim.risk_nested_cv_constrain(X=x,
                                                     Y=y,
                                                     y_label=outcome, 
                                                     max_coef=max_points, 
                                                     max_coef_number=max_features,
                                                     c=c,
                                                     class_weight=weight,
                                                     single_cutoff=single_cutoff,
                                                     two_cutoffs=two_cutoffs,
                                                     three_cutoffs=three_cutoffs,
                                                     four_cutoffs=four_cutoffs,
                                                     seed=42)
        
        end = time.time()
        print(str(round(end - start,1)) + ' seconds')    
        export_results_cv(risk_summary, f'{workdir}Result/{str(year)}_{features}_{scenario}_{weight}{name}.csv') 


    else:
        raise Exception("Scenario undefined.")




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




def export_results_single(y, y_pred, y_prob, filename):

    train_results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                    "Recall": str(round(recall_score(y, y_pred), 4)),
                    "Precision": str(round(precision_score(y, y_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                    "PR AUC": str(round(average_precision_score(y, y_prob), 4))}

    train_results = pd.DataFrame.from_dict(train_results, orient='index', columns=['Train'])
    riskslim_results = train_results.T
    riskslim_results.to_csv(filename)


def export_results_cv(risk_summary, filename):

    results = {"Accuracy": str(round(np.mean(risk_summary['holdout_test_accuracy']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_accuracy']), 4)) + ")",
                "Recall": str(round(np.mean(risk_summary['holdout_test_recall']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_recall']), 4)) + ")",
                "Precision": str(round(np.mean(risk_summary['holdout_test_precision']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_precision']), 4)) + ")",
                "ROC AUC": str(round(np.mean(risk_summary['holdout_test_roc_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_roc_auc']), 4)) + ")",
                "PR AUC": str(round(np.mean(risk_summary['holdout_test_pr_auc']), 4)) + " (" + str(round(np.std(risk_summary['holdout_test_pr_auc']), 4)) + ")"}

    results = pd.DataFrame.from_dict(results, orient='index', columns=['riskSLIM'])
        
    riskslim_results = results.T
    riskslim_results.to_csv(filename)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023
Test scoring table

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from risk_train import risk_train
from risk_test import risk_test
from risk_stumps import create_stumps
from baseline_main import baseline_main
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import sys 

year = sys.argv[1] # 2018
mode = sys.argv[2] # train/test

if mode == 'train':
    scenario = sys.argv[3] # single
    max_points = int(sys.argv[4]) # 5 
    max_features = int(sys.argv[5]) # 10
    weight = sys.argv[6] # unbalanced
    c0 = float(sys.argv[7]) # 1e-15
    name = sys.argv[8] # LTOUR

    first = any(['first' in arg for arg in sys.argv])
    upto180 = any(['upto180' in arg for arg in sys.argv])

    feature = any(['feature' in arg for arg in sys.argv])
    if feature:
        feature_arg = [arg for arg in sys.argv if 'feature' in arg][0]
        feature_set = feature_arg.replace('feature', '')
    else: feature_set = None

    essential = any(['essential' in arg for arg in sys.argv])
    if essential:
        essential_arg = [arg for arg in sys.argv if 'essential' in arg][0]
        essential_num = float(essential_arg.replace('essential', ''))
    else: essential_num = None

    setting_tag = f'_{year}_{mode}_{scenario}_{max_points}p_{max_features}f_{weight}_{name}'
    setting_tag += f"_feature{feature_set}" if feature else ""
    setting_tag += f"_essential{essential_num}" if essential else ""
    setting_tag += f"_first" if first else ""
    setting_tag += f"_upto180" if upto180 else ""

    # setting_tag += "_second_round"
elif mode == 'test':
    suffix = sys.argv[3] # suffix
    first = any(['first' in arg for arg in sys.argv])
    upto180 = any(['upto180' in arg for arg in sys.argv])
    table = any(['table' in arg for arg in sys.argv])

    if table:
        table_arg = [arg for arg in sys.argv if 'table' in arg][0]
        table_case = table_arg.replace('table', '')
    else: table_case = None

    setting_tag = f'_{year}_{mode}_{suffix}'
    setting_tag += f"_first" if first else ""
    setting_tag += f"_upto180" if upto180 else ""

elif mode == 'base_train':
    model = sys.argv[3] # model
    first = any(['first' in arg for arg in sys.argv])
    upto180 = any(['upto180' in arg for arg in sys.argv])

    setting_tag = f'_{year}_{mode}_{model}'
    setting_tag += f"_first" if first else ""
    setting_tag += f"_upto180" if upto180 else ""


def main(year, mode, first, upto180, setting_tag, scenario=None, max_points=None, max_features=None, weight=None, c0=None, feature_set=None, essential_num=None, table_case=None, model=None):

    # =================================== Train LTOUR  ======================================
    if mode == 'train':

        if scenario == 'single':
            print(f'Start single training, file saved with setting tag {setting_tag}\n')
            risk_train(year=year, case=name, first=first, upto180=upto180, scenario=scenario, c=c0, feature_set=feature_set, essential_num=essential_num, max_points=max_points, max_features=max_features, weight=weight, setting_tag=setting_tag)
        
        else: 
            # print(f'Start CV training, points {max_points}, features {max_features}, file saved with name {name}\n')
            # c = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
            # best_c = risk_train(year=2018, features='LTOUR', scenario=scenario, c=c, max_points=max_points, max_features=max_features, weight=weight, name=name)
            # print(f'Single training with the best c = {best_c}\n')
            # risk_train(year=2018, features='LTOUR', scenario='single', c=best_c, max_points=max_points, max_features=max_features, weight=weight, name=name)
            return
    # ===================================  Test LTOUR  =======================================

    elif mode == 'test':
        
        print(f'Start testing, file saved with setting tag {setting_tag}\n')

        # CURES_table = {'intercept': 0,
        #                'conditions':['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers', 
        #                              'num_pharmacies', 'consecutive_days', 'concurrent_benzo'],
        #                              'cutoffs': [90, 40, 6, 6, 90, 1], # '4' refers to quartiles
        #                              'scores': [1, 1, 1, 1, 1, 1]}

        # LTOUR_table = {'intercept': -6,
        #                'conditions':['avgDays_past180', 'avgDays_past180', 'concurrent_MME', 'Medicaid', 'HMFO'],
        #                              'cutoffs': [14, 25, 30, 1, 1], # '4' refers to quartiles
        #                              'scores': [2, 2, 1, 1, 1]}

        all_table = {'intercept': -7, 
                     'conditions': ['age', 'num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
                                    'concurrent_MME', 'HMFO', 'pharmacy_yr_avg_days_quartile'], 
                                    'cutoffs': [30, 1, '4', 30, 1, '4'],
                                    'scores': [3, 2, 2, 1, 1, 1]}

        first_table = {'intercept': -6,
                       'conditions':['prescriber_yr_avg_days_quartile', 'age', 'concurrent_MME', 
                                     'HMFO', 'ever_switch_drug', 'pharmacy_yr_avg_days_quartile'],
                                     'cutoffs': ['4', 40, 40, 1, 1, '4'],
                                     'scores': [3, 2, 1, 1, 1, 1]}
        
        upto180_table = {'intercept': -7,
                         'conditions':['prescriber_yr_avg_days_quartile', 'age', 'concurrent_MME', 
                                       'HMFO', 'Medicare', 'pharmacy_yr_avg_days_quartile'],
                                       'cutoffs': ['4', 30, 30, 1, 1, '4'],
                                       'scores': [3, 2, 1, 1, 1, 1]}

        # conic_table_one = {'intercept': -5,
        #                'conditions':['concurrent_MME', 'num_prior_prescriptions', 'concurrent_benzo', 
        #                              'Oxycodone', 'Medicare', 'ever_switch_drug'],
        #                              'cutoffs': [0.1, 1, 1, 1, 1, 1],
        #                              'scores': [1, 3, 1, 1, 2, 3]}
        
        # conic_table_two = {'intercept': -5,
        #                'conditions':['concurrent_MME', 'num_prior_prescriptions', 'concurrent_benzo', 'Oxycodone', 'Medicare'],
        #                              'cutoffs': [0.05, 1, 1, 1, 1],
        #                              'scores': [3, 3, 1, 1, 2]}

        
        # or_table = {'intercept': -2,
        #          'conditions':[['num_prior_prescriptions', 'patient_zip_avg_days'], ['prescriber_yr_avg_days_quartile'],
        #                        ['concurrent_MME', 'HMFO', 'age', 'ever_switch_drug', 'pharmacy_yr_avg_days_quartile']],
        #                        'cutoffs': [[1, 14], ['4'], [75, 1, 40, 1, '4']],
        #                        'scores': [2, 3, 1]}
        
        
        if table_case == 'all': table = all_table
        if table_case == 'upto180': table = upto180_table
        if table_case == 'first': table = first_table

        # if table_case == 'conic_one': table = conic_table_one
        # if table_case == 'conic_two': table = conic_table_two

        print(table)

        df, calibration_table = risk_test(year=year, table=table, first=first, upto180=upto180, setting_tag=setting_tag)

        # ========================================================================================
        ## Calibration plot
        # color_list = ['red', 'blue', 'orange', 'green', 'brown']
        if False: 
            fig, ax = plt.subplots(figsize=(8, 8))   

            for i in range(len(calibration_table_list)):
                calibration_table = calibration_table_list[i]
                plt.plot(calibration_table['Prob'], calibration_table['Observed Risk'], label=f"Case {i+1}, Error = {df['Calibration error'].iloc[i]}", marker='^', markersize=8, linestyle='solid', color=f'tab:{color_list[i]}')

            min_val, max_val = 0, 0.4
            plt.plot([min_val, max_val], [min_val, max_val], label='y=x', linestyle='--', color='gray')

            plt.xlabel('Predicted Probability Risk', fontsize=20)
            plt.xticks(fontsize=18)
            plt.ylabel('Observed Risk', fontsize=20)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=18)

            plt.show()
            fig.savefig(f'/mnt/phd/jihu/opioid/Result/Plot_LTOUR_raw.pdf', dpi=300)

        # ========================================================================================
        ## ROC plot
        if False:        
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')

            for i in range(len(fpr_list)):
                plt.plot(fpr_list[i], tpr_list[i], label=f"Case {i + 1}, AUC = {df['ROC AUC'].iloc[i]}", linestyle='solid', color=f'tab:{color_list[i]}')
            plt.plot([0, 1], [0, 1], '--', label='Baseline (random classifier)', color='gray')
            
            # for i in range(1, len(thresholds3)):
            #     plt.annotate(f"{thresholds3[i]:.2f}", (fpr_list[2][i], tpr_list[2][i]), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=8, color='black')

            for i in range(1, len(thresholds)):
                plt.annotate(f"{thresholds[i]:.2f}", (fpr_list[0][i], tpr_list[0][i]), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=8, color='black')


            # specificity (true negative rate)
            plt.xlabel("1 - Specificity (false positive rate)", fontsize=20)
            plt.xticks(fontsize=18)
            plt.ylabel("Sensitivity (true positive rate)", fontsize=20)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=18)

            # set x-axis and y-axis limits
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            plt.show()
            fig.savefig(f'/mnt/phd/jihu/opioid/Result/ROC_LTOUR_raw.pdf', dpi=300)


    elif mode == 'base_train':
        baseline_main(year=2018, model=model, first=first, upto180=upto180, setting_tag=setting_tag)

    else:
        raise Exception("Case undefined")


    return



if __name__ == "__main__":

    if mode == 'train': main(year, mode, first, upto180, setting_tag, scenario, max_points, max_features, weight, c0, feature_set, essential_num)
    elif mode == 'test': main(year, mode, first, upto180, setting_tag, table_case=table_case)
    elif mode == 'base_train': main(year, mode, first, upto180, setting_tag, model=model)
    else: raise KeyError("Mode undefined.")
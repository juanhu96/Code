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
from risk_test import test_table, test_table_full
from risk_stumps import create_stumps
from baseline_main import baseline_main
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import sys 

case = sys.argv[1]
scenario = sys.argv[2]
max_points = int(sys.argv[3])
max_features = int(sys.argv[4])
weight = sys.argv[5]
c0 = float(sys.argv[6])
name = sys.argv[7]

def main(case, scenario, max_points, max_features, weight, c0, name):

    print(f'Start {case}ing with {scenario}, points {max_points}, features {max_features}, c = {c0}, file saved with name {name}\n')

    # =================================== Train LTOUR  ======================================
    if case == 'train':

        if scenario == 'single':
            c = c0
            risk_train(year=2018, features='LTOUR', scenario=scenario, c=c, max_points=max_points, max_features=max_features, weight=weight, name=name)
        else: 
            c = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
            best_c = risk_train(year=2018, features='LTOUR', scenario=scenario, c=c, max_points=max_points, max_features=max_features, weight=weight, name=name)
            print(f'Single training with the best c = {best_c}\n')
            risk_train(year=2018, features='LTOUR', scenario='single', c=best_c, max_points=max_points, max_features=max_features, weight=weight, name=name)

    # ===================================  Test LTOUR  =======================================

    elif case == 'test':

        test_result = []
        calibration_table_list = []
        tpr_list = []
        fpr_list = []

        SAMPLE = import_data()

        intercept = -6
        conditions = ['avgDays', 'avgDays', 'concurrent_MME', 'HMFO', 'Medicaid']
        cutoffs = [14, 25, 30, 1, 1]
        scores = [2, 2, 1, 1, 1]

        results, calibration_table, tpr, fpr, thresholds = test_table(SAMPLE, year=2019, table='LTOUR', intercept=intercept, conditions=conditions, cutoffs=cutoffs, scores=scores,
                                                                    calibration=True, roc=True, output_table=True, filename=f'{name}')
        test_result.append(results)
        calibration_table_list.append(calibration_table)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        df = pd.DataFrame(test_result)
        print(df)
        print(calibration_table)


        # =============


        intercept = 0
        conditions = ['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers', 'num_pharmacies', 'consecutive_days', 'concurrent_benzo']
        cutoffs = [90, 40, 6, 6, 90, 1]
        scores = [1, 1, 1, 1, 1, 1]

        results, calibration_table, tpr, fpr, thresholds = test_table(SAMPLE, year=2019, table='CURES', intercept=intercept, conditions=conditions, cutoffs=cutoffs, scores=scores,
                                                                    calibration=True, roc=True, output_table=True, filename=f'{name}')
        test_result.append(results)
        calibration_table_list.append(calibration_table)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        df = pd.DataFrame(test_result)
        print(df)
        print(calibration_table)

        return 

        # df.insert(0, 'c', c_list)
        # df['c'] = df['c'].astype(str)
        # print(df[['c', 'Accuracy', 'ROC AUC', 'Calibration error']])

        color_list = ['red', 'blue', 'orange', 'green', 'brown']

        # ========================================================================================
        ## Calibration plot
        if True: 
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
        if True:        
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


    elif case == 'base_train':
        baseline_main(year=2018, Model_list=['SVM'])

    else:
        raise Exception("Case undefined")


    return



def import_data(year=2019, datadir='/mnt/phd/jihu/opioid/Data'):
    
    SAMPLE = pd.read_csv(f'{datadir}/FULL_{year}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
    dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float, 
    'num_prescribers': int, 'num_pharmacies': int,
    'concurrent_benzo': int, 'consecutive_days': int,
    'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})

    SAMPLE = SAMPLE.fillna(0)
    SAMPLE.rename(columns={'num_presc': 'num_prior_presc'}, inplace=True)

    return SAMPLE




if __name__ == "__main__":
    main(case, scenario, max_points, max_features, weight, c0, name)
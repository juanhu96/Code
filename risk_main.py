#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2023
Test scoring table

@author: Jingyuan Hu
"""

import sys 
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


scenario = sys.argv[1] # single
max_points = int(sys.argv[2]) # 5 
max_features = int(sys.argv[3]) # 10
c0 = float(sys.argv[4]) # 1e-15

balanced = any(['balanced' in arg for arg in sys.argv])

first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])
median =  any(['median' in arg for arg in sys.argv])

feature = any(['feature_' in arg for arg in sys.argv])
if feature:
    feature_arg = [arg for arg in sys.argv if 'feature_' in arg][0]
    feature_set = feature_arg.replace('feature_', '')
else: feature_set = None

essential = any(['essential' in arg for arg in sys.argv])
if essential:
    essential_arg = [arg for arg in sys.argv if 'essential' in arg][0]
    essential_num = float(essential_arg.replace('essential', ''))
else: essential_num = None

nodrug =  any(['nodrug' in arg for arg in sys.argv])

setting_tag = f'_{scenario}_p{max_points}_f{max_features}'

setting_tag += f"_balanced" if balanced else ""

setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_median" if median else ""
setting_tag += f"_nodrug" if nodrug else ""

setting_tag += f"_feature{feature_set}" if feature else ""
setting_tag += f"_essential{essential_num}" if essential else ""


def main(scenario, max_points, max_features, c0, balanced, first, upto180, median, feature_set, essential_num, nodrug, setting_tag):

    # =================================== Train ======================================

    if scenario == 'single':
        print(f'Start single training, file saved with setting tag {setting_tag}\n')
        weight = 'balanced' if balanced else 'original'
        table = risk_train(scenario, 2018, max_points, max_features, c0, weight, first, upto180, median, feature_set, essential_num, nodrug, setting_tag)
    
    else: 
        print(f'Start CV training, points {max_points}, features {max_features}, file saved with name {name}\n')
        raise Exception("CV training not implemented.")
        # c = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
        # best_c = risk_train(year=2018, features='LTOUR', scenario=scenario, c=c, max_points=max_points, max_features=max_features, weight=weight, name=name)
        # print(f'Single training with the best c = {best_c}\n')
        # risk_train(year=2018, features='LTOUR', scenario='single', c=best_c, max_points=max_points, max_features=max_features, weight=weight, name=name)
        # return

    # =================================== Test ========================================

    CURES_table = {'intercept': -5,
                   'conditions':['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers_past180', 
                                 'num_pharmacies_past180', 'consecutive_days', 'concurrent_benzo'],
                                 'cutoffs': [90, 40, 6, 6, 90, 1], # '4' refers to quartiles
                                 'scores': [5, 5, 5, 5, 5, 5]}

    LTOUR_table = {'intercept': -7, 
    'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
    'concurrent_MME', 'age', 'Medicaid', 'pharmacy_yr_avg_days_quartile'], 
    'cutoffs': [1, '1', 40, 30, 1, '1'], 
    'scores': [2, 2, 1, 1, 1, 1]}

    or_table = {'intercept': -2,
             'conditions':[['num_prior_prescriptions', 'patient_zip_avg_days'], ['prescriber_yr_avg_days_quartile'],
                           ['concurrent_MME', 'HMFO', 'age', 'ever_switch_drug', 'pharmacy_yr_avg_days_quartile']],
                           'cutoffs': [[1, 14], ['4'], [75, 1, 40, 1, '4']],
                           'scores': [2, 3, 1]}

    Medicare_Medicaid_table = {'intercept': -7, 
    'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
    'concurrent_MME', 'age', 'Medicare_Medicaid', 'pharmacy_yr_avg_days_quartile'], 
    'cutoffs': [1, '1', 40, 30, 1, '1'], 
    'scores': [2, 2, 1, 1, 1, 1]}

    Long_acting_table = {'intercept': -7, 
    'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
    'concurrent_MME', 'age', 'long_acting', 'pharmacy_yr_avg_days_quartile'],
    'cutoffs': [1, '1', 40, 30, 1, '1'],
    'scores': [2, 2, 1, 1, 1, 1]}

    # table = Long_acting_table
    
    # print("Start testing with table:")
    # print(f"Intercept: {table['intercept']}\n")
    # for condition, cutoff, score in zip(table['conditions'], table['cutoffs'], table['scores']):
    #     print(f" - Condition: {condition}, Cutoff: {cutoff}, Score: {score}")
    
    # risk_test(2019, table, first, upto180, median, setting_tag)

    return



if __name__ == "__main__":
    main(scenario, max_points, max_features, c0, balanced, first, upto180, median, feature_set, essential_num, nodrug, setting_tag)
    
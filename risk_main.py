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

stage = sys.argv[1] 

# =================================== Train ====================================

maxpoint = any(['maxpoint' in arg for arg in sys.argv])
if maxpoint:
    maxpoint_arg = [arg for arg in sys.argv if 'maxpoint' in arg][0]
    max_points = int(maxpoint_arg.replace('maxpoint', ''))
else: max_points = 5

maxfeatures = any(['maxfeatures' in arg for arg in sys.argv])
if maxfeatures:
    maxfeatures_arg = [arg for arg in sys.argv if 'maxfeatures' in arg][0]
    max_features = int(maxfeatures_arg.replace('maxfeatures', ''))
else: max_features = 10

c0 = any(['c0' in arg for arg in sys.argv])
if c0:
    c0_arg = [arg for arg in sys.argv if 'c0' in arg][0]
    c0 = float(c0_arg.replace('c0', ''))
else: c0 = None

single = any(['single' in arg for arg in sys.argv])
scenario = 'single' if single else 'CV'

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

county_arg = next((arg for arg in sys.argv if arg.startswith('county')), None)
if county_arg:
    county_name = county_arg.replace('county', '').strip()
else:
    county_name = None

# =================================== Train ====================================

setting_tag = f"{stage}"
setting_tag += f"_{scenario}" if scenario else ""
setting_tag += f"_p{max_points}" if maxpoint else ""
setting_tag += f"_f{max_features}" if maxfeatures else ""
setting_tag += f"_c0{c0}" if c0 else ""
setting_tag += f"_single" if single else ""
setting_tag += f"_balanced" if balanced else ""
setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_median" if median else ""
setting_tag += f"_feature{feature_set}" if feature else ""
setting_tag += f"_essential{essential_num}" if essential else ""
setting_tag += f"_nodrug" if nodrug else ""
setting_tag += f"_county{county_name}" if county_name is not None else ""
print(f"Setting tag: {setting_tag}")


def main(stage, scenario, max_points, max_features, c0, balanced, first, upto180, median, feature_set, essential_num, nodrug, county_name, setting_tag):

    # =================================== Train ======================================
    if stage == 'train':
        
        if scenario == 'single':
            print(f'Start single training, file saved with setting tag {setting_tag}\n')
            raise SystemExit("..")
            weight = 'balanced' if balanced else 'original'
            table = risk_train(scenario, 2019, max_points, max_features, c0, weight, first, upto180, median, feature_set, essential_num, nodrug, county_name, setting_tag)
        
        else: 
            print(f'Start CV training, points {max_points}, features {max_features}, file saved with name {name}\n')
            raise Exception("CV training not implemented.")
            c = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
            best_c = risk_train(year=2018, features='LTOUR', scenario=scenario, c=c, max_points=max_points, max_features=max_features, weight=weight, name=name)
            print(f'Single training with the best c = {best_c}\n')
            risk_train(year=2018, features='LTOUR', scenario='single', c=best_c, max_points=max_points, max_features=max_features, weight=weight, name=name)
            return

    # =================================== Test ========================================
    elif stage == 'test':
        
        print(f'Start testing...')

        CURES_table = {'intercept': -7, 
        'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
        'concurrent_MME', 'age', 'long_acting', 'pharmacy_yr_avg_days_quartile'],
        'cutoffs': [1, '1', 40, 30, 1, '1'],
        'scores': [2, 2, 1, 1, 1, 1]}

        LTOUR_table = {'intercept': -7, 
        'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
        'concurrent_MME', 'age', 'Medicaid', 'pharmacy_yr_avg_days_quartile'], 
        'cutoffs': [1, '1', 40, 30, 1, '1'], 
        'scores': [2, 2, 1, 1, 1, 1]}

        table = CURES_table
        
        print("Start testing with table:")
        print(f"Intercept: {table['intercept']}\n")
        for condition, cutoff, score in zip(table['conditions'], table['cutoffs'], table['scores']):
            print(f" - Condition: {condition}, Cutoff: {cutoff}, Score: {score}")
        
        risk_test(2019, table, first, upto180, median, setting_tag)

    return



if __name__ == "__main__":
    main(stage, scenario, max_points, max_features, c0, balanced, first, upto180, median, feature_set, essential_num, nodrug, county_name, setting_tag)
    
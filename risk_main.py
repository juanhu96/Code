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

interceptub = any(['interceptub' in arg for arg in sys.argv])
if interceptub:
    interceptub_arg = [arg for arg in sys.argv if 'interceptub' in arg][0]
    interceptub = int(interceptub_arg.replace('interceptub', ''))
else: interceptub = None

interceptlb = any(['interceptlb' in arg for arg in sys.argv])
if interceptlb:
    interceptlb_arg = [arg for arg in sys.argv if 'interceptlb' in arg][0]
    interceptlb = int(interceptlb_arg.replace('interceptlb', ''))
else: interceptlb = None

single = any(['single' in arg for arg in sys.argv])
scenario = 'single' if single else 'CV'

balanced = any(['balanced' in arg for arg in sys.argv])

first = any(['first' in arg for arg in sys.argv])
upto180 = any(['upto180' in arg for arg in sys.argv])

feature = any(['feature_' in arg for arg in sys.argv])
if feature:
    feature_arg = [arg for arg in sys.argv if 'feature_' in arg][0]
    feature_set = feature_arg.replace('feature_', '')
else: feature_set = None

cutoff = any(['cutoff_' in arg for arg in sys.argv])
if cutoff:
    cutoff_arg = [arg for arg in sys.argv if 'cutoff_' in arg][0]
    cutoff_set = cutoff_arg.replace('cutoff_', '')
else: cutoff_set = None

essential = any(['essential' in arg for arg in sys.argv])
if essential:
    essential_arg = [arg for arg in sys.argv if 'essential' in arg][0]
    essential_num = float(essential_arg.replace('essential', ''))
else: essential_num = None

nodrug =  any(['nodrug' in arg for arg in sys.argv])
noinsurance = any(['noinsurance' in arg for arg in sys.argv])

gender_arg = next((arg for arg in sys.argv if arg.startswith('gender')), None)
if gender_arg == 'gender_male':
    gender = 'male'
elif gender_arg == 'gender_female':
    gender = 'female'
else:
    gender = None

county_arg = next((arg for arg in sys.argv if arg.startswith('county')), None)
if county_arg:
    county_name = county_arg.replace('county', '').strip()
else:
    county_name = None

table_name = next((arg for arg in sys.argv if arg.startswith('table')), None)
if table_name:
    table_name = table_name.replace('table', '').strip()
else:
    table_name = None

stretch = any(['stretch' in arg for arg in sys.argv])
exact = any(['exact' in arg for arg in sys.argv])

# =================================== Train ====================================

setting_tag = f"_{stage}"
setting_tag += f"_{scenario}" if scenario else ""
setting_tag += f"_p{max_points}" if maxpoint else ""
setting_tag += f"_f{max_features}" if maxfeatures else ""
setting_tag += f"_c0{c0}" if c0 else ""
setting_tag += f"_interceptub{interceptub}" if interceptub else ""
setting_tag += f"_interceptlb{interceptlb}" if interceptlb else ""
setting_tag += f"_single" if single else ""
setting_tag += f"_balanced" if balanced else ""
setting_tag += f"_first" if first else ""
setting_tag += f"_upto180" if upto180 else ""
setting_tag += f"_feature{feature_set}" if feature else ""
setting_tag += f"_cutoff{cutoff_set}" if cutoff else ""
setting_tag += f"_essential{essential_num}" if essential else ""
setting_tag += f"_nodrug" if nodrug else ""
setting_tag += f"_noinsurance" if noinsurance else ""
setting_tag += f"_gender{gender}" if gender is not None else ""
setting_tag += f"_county{county_name}" if county_name is not None else ""
setting_tag += f"_stretch" if stretch else ""
setting_tag += f"_exact" if exact else ""
print(f"Setting tag: {setting_tag}")


def main(stage, scenario, max_points, max_features, c0, interceptub, interceptlb, balanced, first, upto180, feature_set, cutoff_set, essential_num, nodrug, noinsurance, county_name, table_name, stretch, exact, setting_tag):

    print(f"Stage: {stage}, Scenario: {scenario}, Max Points: {max_points}, Max Features: {max_features}, C0: {c0},\
        Intercept Upper Bound: {interceptub}, Intercept Lower Bound: {interceptlb}, Balanced: {balanced},\
            First: {first}, Upto180: {upto180}, Feature Set: {feature_set}, Cutoff Set: {cutoff_set}, Essential Num: {essential_num}, No Drug: {nodrug},\
                No Insurance: {noinsurance}, Gender: {gender}, County Name: {county_name}, Setting Tag: {setting_tag}")

    # =================================== Train ======================================
    if stage == 'train':
        print(f'Start single training, file saved with setting tag {setting_tag}\n')
        weight = 'balanced' if balanced else 'original'
        table = risk_train(scenario, 2018, max_points, max_features, c0, interceptub, interceptlb, weight, first, upto180, feature_set, cutoff_set, essential_num, nodrug, noinsurance, gender, county_name, stretch, exact, setting_tag)

    # =================================== Test ========================================
    elif stage == 'test':
        
        print(f'Start testing...')

        CURES = {'intercept': -7, 
        'conditions': ['num_prior_prescriptions', 'prescriber_yr_avg_days_quartile', 
        'concurrent_MME', 'age', 'long_acting', 'pharmacy_yr_avg_days_quartile'],
        'cutoffs': [1, '1', 40, 30, 1, '1'],
        'scores': [2, 2, 1, 1, 1, 1]}

        LTOUR_naive = {'intercept': -7, 
        'conditions': ['days_supply', 'daily_dose', 'HMFO', 'long_acting', 'Medicaid', 
        'prescriber_yr_avg_days_quartile', 'pharmacy_yr_avg_days_quartile'], 
        'cutoffs': [10, 30, 1, 1, 1, '1', '1'], 
        'scores': [3, 1, 1, 1, 1, 1, 1]}

        LTOUR_6 = {'intercept': -5, 
        'conditions': ['days_supply', 'num_prior_prescriptions', 'daily_dose', 'HMFO', 'long_acting', 'prescriber_yr_avg_days_above75'], 
        'cutoffs': [10, 1, 90, 1, 1, 1], 
        'scores': [2, 2, 1, 1, 1, 1]}

        LTOUR_naive_6 = {'intercept': -7,
        'conditions': ['days_supply', 'prescriber_yr_avg_days_above75', 'concurrent_benzo', 'daily_dose', 'HMFO', 'long_acting'],
        'cutoffs': [10, 1, 1, 25, 1, 1],
        'scores': [3, 2, 1, 1, 1, 1]}

        # iterate through tables
        tables = {
            'CURES': CURES,
            'LTOUR_naive': LTOUR_naive,
            'LTOUR_6': LTOUR_6,
            'LTOUR_naive_6': LTOUR_naive_6,
        }
                
        table = tables[table_name]
        setting_tag = f"_{table_name}"
        if county_name is not None: setting_tag += f"_county{county_name}"
        if first: setting_tag += f"_first"
        print(f"Start testing with {table_name}:")
        print(f"Intercept: {table['intercept']}\n")

        for condition, cutoff, score in zip(table['conditions'], table['cutoffs'], table['scores']):
            print(f" - Condition: {condition}, Cutoff: {cutoff}, Score: {score}")
        
        print(f"Setting tag: {setting_tag}\n")
        
        risk_test(2018, table, first, upto180, county_name, f'{setting_tag}_2018')
        risk_test(2019, table, first, upto180, county_name, f'{setting_tag}_2019')

    return



if __name__ == "__main__":
    main(stage, scenario, max_points, max_features, c0, interceptub, interceptlb, balanced, first, upto180, feature_set, cutoff_set, essential_num, nodrug, noinsurance, county_name, table_name, stretch, exact, setting_tag)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 2023

Complete version of predicting long-term 180

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from risk_utils import risk_train, risk_train_two_stage, risk_train_last_stage, risk_train_patient
from iterative_table import iterative_table, test_table, test_table_full, test_table_extra
from stumps_new import create_stumps

def main():
    
    ###########################################################################
    # create_stumps_patient(year = 2018, scenario = 'THIRD')     
    # risk_train_patient(year = 2018, features = 'flexible', scenario = 'nested', case = 'THIRD')
    
    # create_stumps_patient(year = 2018, scenario = 'SECOND') 
    # risk_train_patient(year = 2018, features = 'flexible', scenario = 'nested', case = 'SECOND')
    
    # create_stumps_patient(year = 2018, scenario = 'FIRST') 
    # risk_train_patient(year = 2018, features = 'flexible', scenario = 'nested', case = 'FIRST')
    
    # create_stumps(year = 2018) 
    # risk_train(year = 2018, features = 'base', scenario = 'single', c = 1e-4)
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4)
    
    
    # 'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers'
    # 'num_pharmacies', 'concurrent_benzo', 'consecutive_days'
                
    # iterative_table(year=2018, 
    #                 current_cutoffs = [0, 60, 10, 3, 0, 2, 20],
    #                 scores = [-2, 1, 3, 1, 0, 1, 3],
    #                 feature = 'num_prescribers', 
    #                 new_cutoff = [1,2,4,5,6])
      
    ###########################################################################
    ### Train single table under base, flexible, full for 2018
    # risk_train(year = 2018, features = 'base', scenario = 'single', c = 1e-4)
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4)
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4)
    
    ### Test the table under base, flexible, full for 2019 multiple times   
    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 0, 0, 5], case = 'base') # base
    # test_table(year=2019, cutoffs=[0, 40, 80, 3, 0, 2, 20], scores=[-2, 1, 1, 1, 0, 1, 3], case = 'flexible') # flexible
    # test_table_full(year=2019) # full

    
    ###########################################################################
    ### Iterate over num pharmacies
    # iterative_table(year = 2018, 
    #                 current_cutoffs = [0, 40, 80, 3, 0, 2, 20],
    #                 scores = [-2, 1, 1, 1, 0, 1, 3],
    #                 feature = 'num_prescribers', 
    #                 new_cutoff = [1,2,4,5,6])

    # iterative_table(year = 2018, 
    #                 current_cutoffs = [0, 40, 80, 3, 0, 2, 20],
    #                 scores = [-2, 1, 1, 1, 0, 1, 3],
    #                 feature = 'consecutive_days', 
    #                 new_cutoff = [10, 30, 40, 50, 60, 70, 80, 90])
    
    # iterative_table(year = 2018, 
    #                 current_cutoffs = [0, 40, 80, 3, 0, 2, 20],
    #                 scores = [-2, 1, 1, 1, 0, 1, 3],
    #                 feature = 'concurrent_MME', 
    #                 new_cutoff = [10, 20, 30, 50, 60, 70, 80, 90])
    
    # iterative_table(year = 2018, 
    #                 current_cutoffs = [0, 40, 80, 3, 0, 2, 20],
    #                 scores = [-2, 1, 1, 1, 0, 1, 3],
    #                 feature = 'concurrent_benzo', 
    #                 new_cutoff = [1, 3, 4, 5])
    
    ###########################################################################
    ### Different weights, nested CV
    # weight_list = ['original', 'balanced', 'positive', 'positive_2', 'positive_4']
    # for weight in weight_list:
    #     print('Start on ' + weight)
    #     risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4], weight = weight) # for nested, c has to be a list
    
    ###########################################################################
    ### Second stage training
    # candidate_features = ['age', 'concurrent_MME',  'consecutive_days', 
    #                       'concurrent_benzo_same', 'num_presc', 'dose_diff',
    #                       'switch_drug', 'switch_payment', 'Medicare']
    # risk_train_two_stage(year = 2018, candidate_features = candidate_features, c = [1e-4])
    
    ###########################################################################
    ### Third stage training
    # candidate_features = ['age', 'concurrent_MME',  'consecutive_days', 
    #                       'concurrent_benzo_same', 'num_presc',
    #                       'switch_drug', 'switch_payment']
    # risk_train_two_stage(year = 2018, candidate_features = candidate_features, c = [1e-4])
    
    ###########################################################################
    ### Fourth stage training
    # candidate_features = ['age', 'concurrent_MME',  'consecutive_days', 
    #                       'concurrent_benzo_same', 'num_presc',
    #                       'switch_drug', 'switch_payment']
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '1')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '2')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '3')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '4')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '5')
    
    ###########################################################################
    ### Extra round, train 
    # risk_train(year = 2018, features = 'selected', selected_feautres = ['age30', 'consecutive_days20', 'num_presc4'], scenario = 'nested', c = [1e-4])
    # test_table_extra(year=2019)
    
    ###########################################################################
    ### Out of sample test on 2019
    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 0, 0, 6], case = 'base') # base
      
    # concurrent_MME, concurrent_methadone_MME, num_prescribers, num_pharmacies, concurrent_benzo, consecutive_days
    # test_table(year=2019, cutoffs=[0, 40, 10, 4, 0, 2, 20], scores=[-2, 1, 1, 1, 0, 1, 3], case = 'flexible1') # flexible
    # test_table(year=2019, cutoffs=[0, 60, 10, 2, 0, 2, 20], scores=[-2, 1, 1, 1, 0, 1, 3], case = 'flexible2') # flexible
    # test_table(year=2019, cutoffs=[0, 60, 0, 2, 5, 0, 20], scores=[-2, 1, 0, 1, 5, 0, 3], case = 'flexible3') # flexible
    # test_table(year=2019, cutoffs=[0, 100, 80, 3, 0, 1, 20], scores=[-2, 1, 5, 1, 0, 1, 4], case = 'flexible4') # flexible
    # test_table(year=2019, cutoffs=[0, 30, 0, 2, 2, 1, 20], scores=[-3, 1, 0, 1, 1, 1, 4], case = 'flexible5') # flexible
    
    # Full
    # test_table_full(year=2019) # full

    ###########################################################################
    ### Exponential weights
    # print('Alpha = 0.1, beta = 10')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '01', beta = '10', name = '1')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '01', beta = '10', name = '2')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '01', beta = '10', name = '3')
    
    # print('Alpha = 1, beta = 10')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '1', beta = '10', name = '1')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '1', beta = '10', name = '2')
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', c = 1e-4, alpha = '1', beta = '10', name = '3')

    ###########################################################################
    ### Alternative metric: how early can the model detect a long-term user
    # test_table(year=2019, cutoffs=[0, 40, 80, 3, 0, 2, 20], scores=[-2, 1, 1, 1, 0, 1, 3], case = 'flexible', output_table = True) # flexible
    
    ###########################################################################
    ### To compare with baseline, do single + flexible + no constraint
    # risk_train(year = 2018, features = 'flexible', scenario = 'single', constraint = False, c = 1e-4, output_y = True)
    
    ###########################################################################
    ### Redo analysis with age < 18 dropped
    # create_stumps(year = 2018)
    # risk_train(year = 2018, features = 'base', scenario = 'nested', c = [1e-4])
    # risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4], interaction_effects=False)
    
    ### Out of sample test with age < 18 dropped
    # test_table_full(year=2019) # full
    
    # create_intervals(year=2018, scenario='flexible')
    
    ###########################################################################
    ### Redo flexible with intervals (instead of stumps)
    
    # with constraints
    # risk_train_intervals(year = 2018, features = 'flexible', scenario = 'nested', c = [1e-4])
    # without constraints for baseline comparison
    # risk_train_intervals(year = 2018, features = 'flexible', scenario = 'nested', constraint = False, c = [1e-4], name='noconstraint')

    ## allow more conditions to be selected
    # risk_train_intervals(year = 2018, features = 'flexible', scenario = 'nested', c = [1e-4], L0=8)
    # risk_train_intervals(year = 2018, features = 'flexible', scenario = 'nested', constraint = False, c = [1e-4], L0=8, name='noconstraint')    
    
    ###########################################################################
    ########################### NO 18 NO CHRONIC ##############################
    ###########################################################################
    
    # Quick trial (temp)
    # create_stumps(year = 2018)
    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = True, name = 'interc_1')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = True, name = 'interc_2')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = True, name = 'interc_3')
    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = False, name = '1')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = False, name = '2')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects = False, name = '3')
    
    ###########################################################################
    
    ### Test the all table under base, full for 2019 multiple times   
    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 2, 0, 0, 5], case = 'base') # base, no variance
    # test_table_full(year=2019) # full
    
    ###########################################################################
    ### Train 2018 single table multiple times
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, name = 'one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, name = 'two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, name = 'three')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, name = 'four')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, name = 'five')
    
    ###########################################################################
    ### How early can the model detects long-term user?
    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 2, 0, 0, 5], case = 'base', output_table = True) # base
    # test_table_full(year=2019, output_table = True)
    
    # test_table(year=2018, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 2, 0, 0, 5], case = 'base', output_table = True) # base
    # test_table_full(year=2018, output_table = True)
    
    ## Unconstrained
    # risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4], interaction_effects=False, constraint=False, name='noconstr')  
    
    ###########################################################################
    ### For UPTOFIRST prescriptions
    # create_stumps_uptofirst(year = 2018)
    # risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4], interaction_effects=False, name='uptofirst')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='uptofirst_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='uptofirst_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='uptofirst_three')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='uptofirst_four')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='uptofirst_five')
    
    ###########################################################################
    
    # create_stumps_uptofirst(year = 2018)
    
    ### Operational constraint
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='three')
    
    
    ### Operational constraint, 10 maximum points
    # print('================================================================\n')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, interaction_effects=False, name='10one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, interaction_effects=False, name='10two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, interaction_effects=False, name='10three')
    
    
    ### Operational constraint, 3 maximum points
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, interaction_effects=False, name='2one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, interaction_effects=False, name='2two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, interaction_effects=False, name='2three')
    
    
    ### Two stumps for selected continuous variables (requires modifying lattice_cpa)
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='multiple_constr_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='multiple_constr_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='multiple_constr_three')
      
    ### Extra round with candidate features
    # candidate_features = ['age', 'consecutive_days', 'ever_switch_drug', 'ever_switch_payment']
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '1')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '2')
    # risk_train_last_stage(year = 2018, candidate_features = candidate_features, c = 1e-4, name = '3')
    
    ###########################################################################
    ### Allow ten features
    # create_stumps_uptofirst(year = 2018)
    
    # print('2_6\n')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, max_features=6, interaction_effects=False, name='2_6_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, max_features=6, interaction_effects=False, name='2_6_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=2, max_features=6, interaction_effects=False, name='2_6_three')
    # print('3_6\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=6, interaction_effects=False, name='3_6_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=6, interaction_effects=False, name='3_6_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=6, interaction_effects=False, name='3_6_three')
    # print('5_6\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=6, interaction_effects=False, name='5_6_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=6, interaction_effects=False, name='5_6_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=6, interaction_effects=False, name='5_6_three')
    # print('10_6\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, max_features=6, interaction_effects=False, name='10_6_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, max_features=6, interaction_effects=False, name='10_6_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=10, max_features=6, interaction_effects=False, name='10_6_three')
    # print('2_10\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-6, max_points=2, max_features=10, interaction_effects=False, name='2_10_three')
    # print('3_10\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=10, interaction_effects=False, name='3_10_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=10, interaction_effects=False, name='3_10_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=3, max_features=10, interaction_effects=False, name='3_10_three')
    # print('5_10\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=10, interaction_effects=False, name='5_10_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=10, interaction_effects=False, name='5_10_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, max_points=5, max_features=10, interaction_effects=False, name='5_10_three')
    # print('10_10\n')    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=10, max_features=10, interaction_effects=False, name='10_10_one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=10, max_features=10, interaction_effects=False, name='10_10_two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-6, max_points=10, max_features=10, interaction_effects=False, name='10_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_one_nodays')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_two_nodays')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_three_nodays')
    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_one_onedays')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_two_onedays')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_three_onedays')
    
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='test')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_two_onedays')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_three_onedays')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=1, max_features=10, interaction_effects=False, name='1_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_three')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_four')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_five')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=5, max_features=10, interaction_effects=False, name='5_10_three')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_one')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_two')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_three')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_four')
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-8, max_points=2, max_features=10, interaction_effects=False, name='2_10_five')
    
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_iterone')
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_itertwo')
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_iterthree')
    
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_iterone')
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_itertwo')
    # risk_train(year = 2018, features = 'selected', scenario = 'single', c = 1e-8, max_points=3, max_features=10, interaction_effects=False, name='3_10_iterthree')
    
    # risk_train(year = 2018, features = 'base', scenario = 'nested', c = [1e-4])
    
    # risk_train(year = 2018, features = 'speical', scenario = 'nested', c = [1e-4], max_points=3, max_features=10, interaction_effects=False, name='')
    
    test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'base', output_table = True) # base
    # test_table_full(year=2019, output_table = True) # full

    pass
    
    
    
if __name__ == "__main__":
    main()

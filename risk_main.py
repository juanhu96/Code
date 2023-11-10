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
from risk_test import test_table, test_table_full, test_table_temp
from risk_stumps import create_stumps


import sys 

case = sys.argv[1]
scenario = sys.argv[2]
max_points = int(sys.argv[3])
max_features = int(sys.argv[4])
weight = sys.argv[5]
name = sys.argv[6]

def main(case, scenario, max_points, max_features, weight, name):

    # =================================== Train LTOUR (nested) ===================================
    # risk_train(year = 2018, features = 'LTOUR', scenario = 'single', c = 1e-4)
    # risk_train(year = 2018, features = 'LTOUR', scenario = 'single', c = 1e-4, max_points=3, max_features=10)
    # risk_train(year = 2018, features = 'LTOUR', scenario = 'nested', c = [1e-4], max_points=3, max_features=10)


    # =================================== Test CURES ============================================
    # test_table(year = 2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'CURES', calibration = True, roc = True) 
    

    # =================================== Test LTOUR ===================================
    # test_table_full(year = 2019, calibration = True)
    # test_table_temp(year = 2019, calibration = True)

    ### OUTSAMPLE TEST BASE & FULL ROC
    # test_table_full_final(year = 2019, case = 'LTOUR', outcome='long_term_180', roc=True)


    # =================================== Train LTOUR (test) ===================================
    if case == 'train':
        
        if scenario == 'single': c = 1e-10
        if scenario == 'nested': c = [1e-10]

        risk_train(year=2018, features='LTOUR', scenario=scenario, c=1e-8, max_points=max_points, max_features=max_features, weight=weight, name=name)

    elif case == 'test':
        
        if max_features == 6:
            intercept = -5
            # conditions = ['avgDays', 'avgDays', 'quantity', 'avgDays', 'num_presc', 'Codeine']
            # cutoffs = [60, 14, 50, 21, 2, 1]
            # scores = [4, 2, 1, 1, -1, -1]
            conditions = ['avgDays', 'avgDays', 'avgDays', 'ever_switch_drug', 'num_presc', 'Codeine']
            cutoffs = [90, 10, 21, 1, 2, 1]
            scores = [4, 2, 2, 1, -1, -1]
            '''
            NEW CONSTRAINT

                SIX-FEATURE TABLE
            +----------------------------------------------+-------------------+-----------+
            | Pr(Y = +1) = 1.0/(1.0 + exp(-(-5 + score))   |                   |           |
            | ============================================ | ================= | ========= |
            | avgDays90                                    |          4 points |   + ..... |
            | avgDays10                                    |          2 points |   + ..... |
            | avgDays21                                    |          2 points |   + ..... |
            | ever_switch_drug                             |          1 points |   + ..... |
            | num_presc2                                   |         -1 points |   + ..... |
            | Codeine                                      |         -1 points |   + ..... |
            | ============================================ | ================= | ========= |
            | ADD POINTS FROM ROWS 1 to 6                  |             SCORE |   = ..... |
            +----------------------------------------------+-------------------+-----------+
            '''

        elif max_features == 10:
            # intercept = -4
            # conditions = ['avgDays', 'concurrent_MME', 'quantity', 'avgDays', 
            # 'avgDays', 'avgDays', 'avgDays', 'quantity', 'num_presc', 'num_presc']
            # cutoffs = [14, 30, 50, 3, 7, 25, 30, 40, 2, 1]
            # scores = [2, 1, 1, 1, 1, 1, 1, -1, -1, -3]
            
            intercept = -5
            conditions = ['avgDays', 'avgDays', 'avgDays', 'concurrent_MME', 'quantity', 'quantity', 
            'num_presc', 'num_presc', 'Codeine']
            cutoffs = [90, 7, 21, 10, 40, 20, 1, 2, 1]
            scores = [5, 2, 2, 1, 1, -1, -1, -1, -1]

            '''
            NEW CONSTRAINT
            
                TEN-FEATURE TABLE
            +----------------------------------------------+-------------------+-----------+
            | Pr(Y = +1) = 1.0/(1.0 + exp(-(-5 + score))   |                   |           |
            | ============================================ | ================= | ========= |
            | avgDays90                                    |          5 points |   + ..... |
            | avgDays7                                     |          2 points |   + ..... |
            | avgDays21                                    |          2 points |   + ..... |
            | concurrent_MME10                             |          1 points |   + ..... |
            | quantity40                                   |          1 points |   + ..... |
            | quantity20                                   |         -1 points |   + ..... |
            | num_presc1                                   |         -1 points |   + ..... |
            | num_presc2                                   |         -1 points |   + ..... |
            | Codeine                                      |         -1 points |   + ..... |
            | ============================================ | ================= | ========= |
            | ADD POINTS FROM ROWS 1 to 9                  |             SCORE |   = ..... |
            +----------------------------------------------+-------------------+-----------+
            '''
        
        elif max_features == 20:
            intercept = -5
            # conditions = ['avgDays', 'avgDays', 'concurrent_MME', 'avgDays', 'concurrent_MME',
            # 'quantity', 'quantity', 'Medicare', 'concurrent_MME', 'quantity', 'num_presc',
            # 'num_prescribers', 'quantity', 'num_presc']
            # cutoffs = [10, 60, 20, 25, 50, 10, 15, 1, 25, 150, 2, 2, 20, 1]
            # scores = [4, 3, 2, 2, 1, 1, 1, 1, -1, -1, -1, -2, -2, -3]

            conditions = ['avgDays', 'avgDays', 'num_presc']
            cutoffs = [10, 14, 1]
            scores = [4, 2, -3]

            '''
            +----------------------------------------------+-------------------+-----------+
            | Pr(Y = +1) = 1.0/(1.0 + exp(-(-5 + score))   |                   |           |
            | ============================================ | ================= | ========= |
            | avgDays10                                    |          4 points |   + ..... |
            | avgDays14                                    |          2 points |   + ..... |
            | num_presc1                                   |         -3 points |   + ..... |
            | ============================================ | ================= | ========= |
            | ADD POINTS FROM ROWS 1 to 3                  |             SCORE |   = ..... |
            +----------------------------------------------+-------------------+-----------+
            '''
        
        else:
            raise Exception("Max features undefined \n")
        
        test_table(year=2019, intercept=intercept, conditions=conditions, cutoffs=cutoffs, scores=scores, calibration=True, filename=f'_{max_points}_{max_features}_{weight}')
    
    else:
        raise Exception("Case undefined")


    return



if __name__ == "__main__":
    main(case, scenario, max_points, max_features, weight, name)
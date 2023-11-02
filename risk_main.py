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
intercept = sys.argv[1]


def main(intercept):

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
    print(intercept)
    # risk_train(year = 2018, features = 'LTOUR', scenario = 'nested', c = [1e-6], max_points=5, max_features=10, intercept_case=intercept, name=intercept)
    # risk_train(year=2018, features='LTOUR', scenario='single', c=1e-8, max_points=5, max_features=10, weight='original', intercept_case=intercept, name=intercept)
    test_table_temp(year=2019, intercept=intercept, calibration=True)
    # risk_train(year=2018, features='LTOUR', scenario='single', c=1e-8, max_points=5, max_features=20, weight='original', intercept_case=intercept, name=intercept)


    pass





if __name__ == "__main__":
    main(intercept)
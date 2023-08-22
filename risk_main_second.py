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

from risk_utils import risk_train, risk_train_two_stage, risk_train_last_stage, \
    risk_train_patient, create_stumps, create_stumps_patient, create_intervals, risk_train_intervals
from iterative_table import iterative_table, test_table, test_table_full, test_table_extra


def main():
    
    ###########################################################################
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='one')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='two')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='three')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='four')
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, interaction_effects=False, name='five')
    
    # risk_train(year = 2018, features = 'speical', scenario = 'nested', c = [1e-8], max_points=3, max_features=10, interaction_effects=False, constraint=False, name='noconstr')
    
    
    # Test base & full
    test_table(year=2018, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'base', output_table = True) # base
    test_table_full(year=2018, output_table = True)
    
    test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'base', output_table = True) # base
    test_table_full(year=2019, output_table = True)
    
    pass
    
    
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 2023

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from risk_utils import risk_train, risk_train_patient, create_stumps, create_stumps_patient
from iterative_table import iterative_table, test_table, test_table_full


def main():
    
    ###########################################################################
    ### Out of sample test on 2019
    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 0, 0, 6], case = 'base') # base
      
    # concurrent_MME, concurrent_methadone_MME, num_prescribers, num_pharmacies, concurrent_benzo, consecutive_days
    # test_table(year=2019, cutoffs=[0, 40, 10, 4, 0, 2, 20], scores=[0, 1, 1, 1, 0, 1, 3], case = 'flexible1') # flexible
    # test_table(year=2019, cutoffs=[0, 60, 10, 2, 0, 2, 20], scores=[0, 1, 1, 1, 0, 1, 3], case = 'flexible2') # flexible
    # test_table(year=2019, cutoffs=[0, 60, 0, 2, 5, 0, 20], scores=[0, 1, 0, 1, 5, 0, 3], case = 'flexible3') # flexible
    # test_table(year=2019, cutoffs=[0, 100, 80, 3, 0, 1, 20], scores=[0, 1, 5, 1, 0, 1, 4], case = 'flexible4') # flexible
    # test_table(year=2019, cutoffs=[0, 30, 0, 2, 2, 1, 20], scores=[0, 1, 0, 1, 1, 1, 4], case = 'flexible5') # flexible
    
    ###########################################################################
    ### Alternative metric: how early can the model detect a long-term user

    # test_table(year=2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 0, 0, 6], case = 'base', output_table = True) # base    
    # test_table_full(year=2019, output_table = True) # full
    # test_table(year=2019, cutoffs=[0, 40, 80, 3, 0, 2, 20], scores=[-2, 1, 1, 1, 0, 1, 3], case = 'flexible', output_table = True) # flexible
      
    ###########################################################################
    # CURES vs. Full (nested)
    # risk_train(year = 2018, features = 'base', scenario = 'nested', c = [1e-4])
    # risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4])

    ###########################################################################

    
    
    
    pass
    
    
    
if __name__ == "__main__":
    main()

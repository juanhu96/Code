#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 2023

@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd

from risk_utils import risk_train, risk_train_patient, create_stumps, create_stumps_patient
from iterative_table import iterative_table, test_table, test_table_full, test_table_full_final


def main():
    
    ###########################################################################
    # CURES vs. Full (single, for ROC-AUC curve)
    # risk_train(year = 2018, features = 'base', scenario = 'single', c = 1e-4, roc=True)
    # risk_train(year = 2018, features = 'full', scenario = 'single', c = 1e-4, roc=True)

    # CURES doesn't require train, test directly
    # obtain roc for roc-curve
    # obtain output table for how early to detect long-term
    # test_table(year = 2018, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'base', outcome='long_term_180', output_table=True, roc=True)
    
    # LTOUR
    # risk_train(year = 2018, features = 'speical', scenario = 'single', c = 1e-4, max_points=3, max_features=10, interaction_effects=False, roc=True, name='roc')

    # LTOUR how early to detect long-term requires output table from the line above
    

    ###########################################################################
    ### OUTSAMPLE TEST BASE & FULL ROC
    # test_table(year = 2019, cutoffs=[0, 90, 40, 6, 6, 1, 90], scores=[0, 1, 1, 1, 1, 1, 1], case = 'CURES', outcome='long_term_180', roc=True)
    test_table_full_final(year = 2019, case = 'LTOUR', outcome='long_term_180', roc=True)

    pass
    
    
    
if __name__ == "__main__":
    main()

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
    risk_train_patient, create_stumps, create_stumps_patient
from iterative_table import iterative_table, test_table, test_table_full, test_table_extra


def main():
    
    ### Different weights, nested CV
    weight_list = ['original', 'balanced', 'positive', 'positive_2', 'positive_4']
    risk_train(year = 2018, features = 'full', scenario = 'nested', c = [1e-4], weight = 'positive_2')
    
    pass
    
    
    
if __name__ == "__main__":
    main()

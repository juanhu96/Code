#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

resultdir = '/mnt/phd/jihu/opioid/Result/'
year = 2019 # out-sample


CURES_calibration = pd.read_csv(f'{resultdir}calibration_CURES.csv', delimiter = ",")
LTOUR_calibration = pd.read_csv(f'{resultdir}calibration_LTOUR_two.csv', delimiter = ",")

# ====================================================================================

if False:
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.plot(CURES_calibration['Prob'], CURES_calibration['Observed Risk'], label='CURES', marker='o', markersize=8, linestyle='solid', color='tab:red')
    plt.plot(LTOUR_calibration['Prob'], LTOUR_calibration['Observed Risk'], label='LTOUR', marker='^', markersize=8, linestyle='solid', color='tab:blue')

    plt.xlabel('Predicted Probability Risk', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel('Observed Risk', fontsize=20)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18)
    plt.show()
    fig.savefig(f'{resultdir}Risk_CURES_LTOUR.pdf', dpi=300)

# ====================================================================================

if False:
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.plot(CURES_calibration['Prob'], CURES_calibration['Accuracy'], label='CURES', marker='o', linestyle='solid', color='red')
    plt.plot(LTOUR_calibration['Prob'], LTOUR_calibration['Accuracy'], label='LTOUR', marker='x', linestyle='solid', color='blue')

    plt.xlabel('Predicted Probability Risk', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel('Accuracy', fontsize=20)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18)
    plt.show()
    fig.savefig(f'{resultdir}Accuracy_CURES_LTOUR.pdf', dpi=300)

# ========

if True:

    probs_CURES = list(np.around(np.array(CURES_calibration['Prob'].tolist()), 2))
    accuracy_CURES = list(np.around(np.array(CURES_calibration['Accuracy'].tolist()), 2))
    probs_LTOUR = list(np.around(np.array(LTOUR_calibration['Prob'].tolist()), 2))
    accuracy_LTOUR = list(np.around(np.array(LTOUR_calibration['Accuracy'].tolist()), 2))

    # make them same length
    probs_CURES = [probs_LTOUR[0], probs_LTOUR[1]] + probs_CURES
    accuracy_CURES = [0, 0] + accuracy_CURES

    x = np.arange(len(probs_CURES))    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.bar(x, accuracy_CURES, width=0.4, label='CURES', color='tab:red')
    plt.bar(x + 0.4, accuracy_LTOUR, width=0.4, label='LTOUR', color='tab:blue')

    
    plt.xlabel('Predicted Probability Risk', fontsize=20)
    plt.xticks(x + 0.4/2, probs_CURES, fontsize=18)
    plt.ylabel('Accuracy', fontsize=20)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18)
    plt.show()
    fig.savefig(f'{resultdir}Accuracy_CURES_LTOUR.pdf', dpi=300)
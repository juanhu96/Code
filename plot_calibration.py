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

fig, ax = plt.subplots(figsize=(8, 8))

plt.plot(CURES_calibration['Prob'], CURES_calibration['Observed Risk'], label='CURES', marker='o', linestyle='solid', color='red')
plt.plot(LTOUR_calibration['Prob'], LTOUR_calibration['Observed Risk'], label='LTOUR', marker='x', linestyle='solid', color='blue')

plt.xlabel('Predicted Probability Risk', fontsize=20)
plt.xticks(fontsize=18)
plt.ylabel('Observed Risk', fontsize=20)
plt.yticks(fontsize=18)

plt.legend(fontsize=18)
plt.show()
fig.savefig(f'{resultdir}Risk_CURES_LTOUR.pdf', dpi=300)

# ====================================================================================

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
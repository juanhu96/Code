#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

resultdir = '/mnt/phd/jihu/opioid/Result/'
year = 2019 # out-sample

## CURES
case = 'CURES'
CURES_fpr = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fpr.csv', delimiter = ",", dtype = float)
CURES_tpr = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tpr.csv', delimiter = ",", dtype = float)
CURES_threshold = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_thresholds.csv', delimiter = ",", dtype = float)

CURES_tp = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tp.csv', delimiter = ",", dtype = float)
CURES_tn = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tn.csv', delimiter = ",", dtype = float)
CURES_fp = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fp.csv', delimiter = ",", dtype = float)
CURES_fn = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fn.csv', delimiter = ",", dtype = float)

CURES_fpr = np.append(0, CURES_fpr)
CURES_fpr = np.append(CURES_fpr, 1)
CURES_tpr = np.append(0, CURES_tpr)
CURES_tpr = np.append(CURES_tpr, 1)


# LTOUR
case = 'LTOUR'
LTOUR_fpr = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fpr.csv', delimiter = ",", dtype = float)
LTOUR_tpr = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tpr.csv', delimiter = ",", dtype = float)
LTOUR_threshold = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_thresholds.csv', delimiter = ",", dtype = float)

LTOUR_tp = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tp.csv', delimiter = ",", dtype = float)
LTOUR_tn = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_tn.csv', delimiter = ",", dtype = float)
LTOUR_fp = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fp.csv', delimiter = ",", dtype = float)
LTOUR_fn = np.genfromtxt(f'{resultdir}result_{str(year)}_{case}_single_balanced_fn.csv', delimiter = ",", dtype = float)


################################################################################################
################################################################################################
################################################################################################


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

plt.plot(CURES_fpr, CURES_tpr, linestyle='solid', color='red', label="CURES (AUC = 0.533)" )
plt.plot(LTOUR_fpr, LTOUR_tpr, linestyle='solid', color='blue', label="LTOUR (AUC = 0.861)")

# specificity (true negative rate)
plt.xlabel("1 - Specificity (false positive rate)", fontsize=22)
plt.xticks(fontsize=20)
plt.ylabel("Sensitivity (true positive rate)", fontsize=22)
plt.yticks(fontsize=20)

# CURES
# plt.plot([CURES_fpr[5], CURES_fpr[8]],
#          [CURES_tpr[5], CURES_tpr[8]], 'o', markersize=8)
# plt.text(CURES_fpr[5] + 0.02, CURES_tpr[5] - 0.03, "A", fontsize=24)
# plt.text(CURES_fpr[8] + 0.02, CURES_tpr[8] + 0.03, "B", fontsize=24)
plt.plot([CURES_fpr[5]],
         [CURES_tpr[5]], 'o', color='red', markersize=8)
plt.text(CURES_fpr[5] + 0.02, CURES_tpr[5] - 0.02, "A", fontsize=24)

# LTOUR
# plt.plot([LTOUR_fpr[5], LTOUR_fpr[8]],
#          [LTOUR_tpr[5], LTOUR_tpr[8]], 'o', markersize=8)
# plt.text(LTOUR_fpr[5] + 0.02, LTOUR_tpr[5], "a", fontsize=24)
# plt.text(LTOUR_fpr[8] + 0.01, LTOUR_tpr[8], "b", fontsize=24)
plt.plot([LTOUR_fpr[5]],
         [LTOUR_tpr[5]], 'o', color='blue', markersize=8)
plt.text(LTOUR_fpr[5] + 0.02, LTOUR_tpr[5], "B", fontsize=24)

# plot a diagonal line from (0,0) to (1,1)
plt.plot([0, 1], [0, 1], '--', label='Baseline (random classifier)')

# set x-axis and y-axis limits
plt.xlim([0, 1])
plt.ylim([0, 1])

# add legend to the plot
plt.legend(loc='lower right', fontsize=18)

# create a table
# data = [['', 'Threshold', 'TP', 'TN', 'FP', 'FN'],
#         ['A', round(CURES_threshold[5],1), round(CURES_tp[5]/1000000,2), round(CURES_tn[5]/1000000,2),
#          round(CURES_fp[5]/1000000,2), round(CURES_fn[5]/1000000,2)],
#         ['B', CURES_threshold[8], round(CURES_tp[8]/1000000,2), round(CURES_tn[8]/1000000,2),
#          round(CURES_fp[8]/1000000,2), round(CURES_fn[8]/1000000,2)],
#         ['a', round(LTOUR_threshold[5],1), round(LTOUR_tp[5]/1000000,2), round(LTOUR_tn[5]/1000000,2),
#          round(LTOUR_fp[5]/1000000,2), round(LTOUR_fn[5]/1000000,2)],
#         ['b', LTOUR_threshold[8], round(LTOUR_tp[8]/1000000,2), round(LTOUR_tn[8]/1000000,2),
#          round(LTOUR_fp[8]/1000000,2), round(LTOUR_fn[8]/1000000,2)]]

# data = [['', 'Threshold', 'TP', 'TN', 'FP', 'FN'],
#         ['A', round(CURES_threshold[5],1), round(CURES_tp[5]/1000000,2), round(CURES_tn[5]/1000000,2),
#          round(CURES_fp[5]/1000000,2), round(CURES_fn[5]/1000000,2)],
#         ['B', round(LTOUR_threshold[5],1), round(LTOUR_tp[5]/1000000,2), round(LTOUR_tn[5]/1000000,2),
#          round(LTOUR_fp[5]/1000000,2), round(LTOUR_fn[5]/1000000,2)]]
# table = plt.table(cellText=data, loc='center right', colWidths=[0.1,0.4,0.2,0.2,0.2,0.2],
#                   rowLabels=None, cellLoc='center', bbox=[1.05, 0, 1, 1])
# table.auto_set_font_size(False)
# table.set_fontsize(22)
# table.auto_set_column_width(range(len(data[0])))

plt.show()
fig.savefig(f'{resultdir}ROC_CURES_LTOUR.pdf', dpi=300)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pickle
import glob
import os
import sys 
import csv

exportdir = '/export/storage_cures/CURES/Results/'
median =  any(['median' in arg for arg in sys.argv])
full = any(['full' in arg for arg in sys.argv])

model_name_dict = {
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest", 
    "L1": "L1 Logistic", 
    "L2": "L2 Logistic", 
    "LinearSVM": "Linear SVM", 
    "XGB": "XGBoost", 
    "NN": "Neural Network",
    "riskSLIM": "LTOUR"
}

feature_dict = {"DecisionTree": 5, "RandomForest": 25, "L1": 53, "L2": 75, "LinearSVM": 75, "XGB": 57, "NN": 75, "riskSLIM": 6}

color_map = {
    'DecisionTree': 'blue',
    'RandomForest': 'gray',
    'L1': 'green',
    'L2': 'orange',
    'LinearSVM': 'purple',
    'XGB': 'brown',
    'NN': 'pink',
    'riskSLIM': 'red',
}

model_order = ['riskSLIM', 'L1', 'DecisionTree', 'XGB', 'L2', 'LinearSVM', 'RandomForest', 'NN']

bold_font = FontProperties(weight='bold')

# =============================================================================
# =============================================================================
# =============================================================================

pattern = '_roc_test_info_median.pkl' if median else '_roc_test_info.pkl'
roc_files = glob.glob(f'output/baseline/*{pattern}')
roc_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
if not full: roc_files = [f for f in roc_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
print(roc_files)

export_data = []
for roc_file in roc_files:
    model_name = os.path.basename(roc_file).replace(pattern, '')
    with open(roc_file, 'rb') as f:
        roc_info = pickle.load(f)
    
    # Append the FPR, TPR, AUC, and #Features to export_data
    for i in range(len(roc_info['fpr'])):
        export_data.append({
            'model': model_name_dict[model_name],
            'fpr': roc_info['fpr'][i],
            'tpr': roc_info['tpr'][i],
            'auc': roc_info['auc'],
            'features': feature_dict[model_name]  # Add the number of features
        })

# Export data to a CSV file
export_csv = f'{exportdir}roc_data{"_median" if median else ""}{"_full" if full else ""}.csv'
with open(export_csv, 'w', newline='') as csvfile:
    fieldnames = ['model', 'fpr', 'tpr', 'auc', 'features']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(export_data)

print(f"ROC data exported to {export_csv}")

# =============================================================================
# =============================================================================
# =============================================================================

pattern = '_calibration_test_info_median.pkl' if median else '_calibration_test_info.pkl'
calibration_files = glob.glob(f'output/baseline/*{pattern}')
calibration_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
if not full: calibration_files = [f for f in calibration_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
print(calibration_files)

export_data = []
for calib_file in calibration_files:
    model_name = os.path.basename(calib_file).replace(pattern, '')
    with open(calib_file, 'rb') as f:
        calib_info = pickle.load(f)

    for i in range(len(calib_info['prob_pred'])):
        export_data.append({
            'model': model_name_dict[model_name],
            'prob_pred': calib_info['prob_pred'][i],
            'prob_true': calib_info['prob_true'][i],
            'ece': calib_info['ece']  # ECE is constant for a model
        })

# Export data to a CSV file
export_csv = f'{exportdir}calibration_data{"_median" if median else ""}{"_full" if full else ""}.csv'
with open(export_csv, 'w', newline='') as csvfile:
    fieldnames = ['model', 'prob_pred', 'prob_true', 'ece']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(export_data)

print(f"Calibration data exported to {export_csv}")

# =============================================================================
# =============================================================================
# =============================================================================

pattern = '_proportions_test_info_median.pkl' if median else '_proportions_test_info.pkl'
proportions_files = glob.glob(f'output/baseline/*{pattern}')
proportions_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
if not full: proportions_files = [f for f in proportions_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
print(proportions_files)

export_data = []
months = [1, 2, 3]
num_months = len(months)

# Loop through each proportions file and extract data
for idx, prop_file in enumerate(proportions_files):
    model_name = os.path.basename(prop_file).replace(pattern, '')

    with open(prop_file, 'rb') as f:
        prop_info = pickle.load(f)

    # Append month and proportion data for the model
    for month, proportion in zip(months, prop_info['proportion']):
        export_data.append({
            'model': model_name_dict[model_name],
            'month': month,
            'proportion': proportion
        })

# Export data to a CSV file
export_csv = f'{exportdir}proportions_data{"_median" if median else ""}{"_full" if full else ""}.csv'
with open(export_csv, 'w', newline='') as csvfile:
    fieldnames = ['model', 'month', 'proportion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(export_data)

print(f"Proportions data exported to {export_csv}")
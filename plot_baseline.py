import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pickle
import glob
import os
import sys 

exportdir = '/export/storage_cures/CURES/Plots/'
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

feature_dict = {"DecisionTree": 5, "RandomForest": 27, "L1": 51, "L2": 75, "LinearSVM": 75, "XGB": 57, "NN": 75, "riskSLIM": 6}

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
if False:
    pattern = '_roc_test_info_median.pkl' if median else '_roc_test_info.pkl'
    roc_files = glob.glob(f'../output/baseline/*{pattern}')
    roc_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
    if not full: roc_files = [f for f in roc_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
    print(roc_files)

    # Create a figure for the ROC plot
    plt.figure()

    # Loop through each file, load the data, and plot the ROC curve
    for roc_file in roc_files:
        # Extract the model name from the file path
        model_name = os.path.basename(roc_file).replace(pattern, '')

        # Load the ROC data
        with open(roc_file, 'rb') as f:
            roc_info = pickle.load(f)
        
        # Plot the ROC curve
        plt.plot(roc_info['fpr'], roc_info['tpr'], label=f'{model_name_dict[model_name]} (AUC = {roc_info["auc"]:.3f}, #Features = {feature_dict[model_name]})', markersize=5, alpha=0.7)

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    if full: plt.legend(loc="lower right", fontsize=9)
    else: plt.legend(loc="lower right", fontsize=10, prop=bold_font)

    # Save the figure as a PDF
    output_pdf = f'{exportdir}baseline_roc_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
    plt.savefig(output_pdf, format='pdf', dpi=300)
    print(f"ROC curves saved to {output_pdf}")


# =============================================================================
# =============================================================================
# =============================================================================

if False:
    pattern = '_calibration_test_info_median.pkl' if median else '_calibration_test_info.pkl'
    calibration_files = glob.glob(f'../output/baseline/*{pattern}')
    calibration_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
    if not full: calibration_files = [f for f in calibration_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
    print(calibration_files)

    plt.figure()
    for calib_file in calibration_files:
        model_name = os.path.basename(calib_file).replace(pattern, '')

        with open(calib_file, 'rb') as f:
            calib_info = pickle.load(f)
        
        if model_name == 'riskSLIM': print(calib_info)

        plt.plot(calib_info['prob_pred'], calib_info['prob_true'], marker='o', label=f'{model_name_dict[model_name]} (ECE = {calib_info["ece"]:.3f})', markersize=5, alpha=0.7)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)

    plt.xlabel('Mean Predicted Risk', fontsize=14)
    plt.ylabel('Observed Risk (Fraction of Positives)', fontsize=14)
    if full: plt.legend(loc="lower right", fontsize=9)
    else: plt.legend(loc="lower right", fontsize=10, prop=bold_font)

    output_pdf = f'{exportdir}baseline_calibration_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
    plt.savefig(output_pdf, format='pdf', dpi=300)
    print(f"Calibration curves saved to {output_pdf}")


# =============================================================================
# =============================================================================
# =============================================================================


# pattern = '_proportions_test_info_median.pkl' if median else '_proportions_test_info.pkl'
# proportions_files = glob.glob(f'../output/baseline/*{pattern}')
# proportions_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
# if not full: proportions_files = [f for f in proportions_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
# print(proportions_files)

# plt.figure()
# for prop_file in proportions_files:
#     model_name = os.path.basename(prop_file).replace(pattern, '')

#     with open(prop_file, 'rb') as f:
#         prop_info = pickle.load(f)
    
#     plt.plot(prop_info['month'], prop_info['proportion'], marker='o', label=f'{model_name_dict[model_name]}', markersize=5, alpha=0.7)

# plt.xticks([1, 2, 3])
# plt.xlabel('Month', fontsize=14)
# plt.ylabel('Proportion (%)', fontsize=14)
# plt.legend(loc="lower right", fontsize=9)

# plt.figure()

# months = [1, 2, 3]
# num_months = len(months)
# bar_width = 0.1  # Width of each bar

# for idx, prop_file in enumerate(proportions_files):
#     model_name = os.path.basename(prop_file).replace(pattern, '')

#     with open(prop_file, 'rb') as f:
#         prop_info = pickle.load(f)
    
#     positions = np.array(months) + idx * bar_width

#     plt.bar(positions, prop_info['proportion'], width=bar_width, label=f'{model_name_dict[model_name]}', alpha=0.7)

# plt.xticks(np.array(months) + bar_width * (len(proportions_files) - 1) / 2, months)
# plt.yticks(np.arange(0, 101, 10))
# plt.xlabel('Month', fontsize=14)
# plt.ylabel('Proportion (%)', fontsize=14)
# if full: plt.legend(loc="upper left", fontsize=9)
# else: plt.legend(loc="upper left", fontsize=10, prop=bold_font)
# plt.show()

# output_pdf = f'{exportdir}baseline_proportions_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
# plt.savefig(output_pdf, format='pdf', dpi=300)
# print(f"proportions curves saved to {output_pdf}")


# =============================================================================
# =============================================================================
# =============================================================================

'''
pattern = '_recallMME_test_info_median.pkl' if median else '_recallMME_test_info.pkl'
recallMME_files = glob.glob(f'output/baseline/*{pattern}')
recallMME_files.sort(key=lambda f: model_order.index(os.path.basename(f).replace(pattern, '')))
if not full: recallMME_files = [f for f in recallMME_files if 'DecisionTree' in f or 'riskSLIM' in f or 'XGB' in f or 'L1' in f]
print(recallMME_files)

plt.figure(figsize=(10, 6))

bar_width = 0.3  # Width of each bar
num_models = len(recallMME_files)
total_bar_width = bar_width * num_models

# for recall_file in recallMME_files:
for idx, recall_file in enumerate(recallMME_files):
    model_name = os.path.basename(recall_file).replace(pattern, '')

    with open(recall_file, 'rb') as f:
        recall_info = pickle.load(f)
    
    print(recall_info)
    pos_ratio_percent = np.array(recall_info['pos_ratio']) * 100
    # plt.bar(x_positions, recall_info['recall'], width=0.8, label=f'{model_name_dict[model_name]}', alpha=0.7)
    x_positions = np.arange(len(recall_info['MME'])) + (idx - (num_models - 1) / 2) * bar_width

    if model_name == 'DecisionTree':
        plt.bar(x_positions, pos_ratio_percent, width=bar_width, label=f'{model_name_dict[model_name]}', alpha=0.7, color='green')
    else:
        plt.bar(x_positions, pos_ratio_percent, width=bar_width, label=f'{model_name_dict[model_name]}', alpha=0.7)


    if model_name =='riskSLIM':
        true_pos_ratio_percent = np.array(recall_info['true_pos_ratio']) * 100
        plt.plot(np.arange(len(recall_info['MME'])), true_pos_ratio_percent, color='black', marker='o', linestyle='--', label='Total', alpha=0.7)

plt.xticks(np.arange(len(recall_info['MME'])), recall_info['MME'], fontsize=9)
plt.xlabel('MME', fontsize=14)
# plt.ylabel('Recall', fontsize=14)
plt.ylabel('% predict positive', fontsize=14)
if full: plt.legend(loc="upper left", fontsize=9)
else: plt.legend(loc="upper left", fontsize=12, prop=bold_font)

output_pdf = f'{exportdir}baseline_recallMME_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
plt.savefig(output_pdf, format='pdf', dpi=300)
print(f"RecallMME curves saved to {output_pdf}")
'''


# =============================================================================
# =============================================================================
# =============================================================================
year = '2019'
if False:

    pattern = 'riskSLIM_roc_test_info_median' if median else 'riskSLIM_roc_test_info'
    roc_files = glob.glob(f'../output/baseline/{pattern}*')
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    plt.figure(figsize=(10, 6))
    for i, roc_file in enumerate(roc_files):
        filename = os.path.basename(roc_file)
        # Only use files with model suffix after the pattern
        if filename.startswith(pattern) and filename != f"{pattern}.pkl":
            model_name = filename[len(pattern):].replace('.pkl', '').lstrip('_')
            print(model_name)  # For debugging
        else:
            continue

        with open(roc_file, 'rb') as f:
            roc_info = pickle.load(f)

        marker = marker_styles[i % len(marker_styles)]
        plt.plot(
            roc_info['fpr'],
            roc_info['tpr'],
            marker=marker,
            label=f'{model_name} (AUC = {roc_info["auc"]:.3f})',
            markersize=6,
            alpha=0.8,
            linewidth=2
        )

    # Plot the diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('riskSLIM ROC Curves', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    output_pdf = f'{exportdir}riskSLIM_roc_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
    plt.savefig(output_pdf, format='pdf', dpi=300)
    print(f"ROC curves saved to {output_pdf}")



    pattern = 'riskSLIM_calibration_test_info_median' if median else 'riskSLIM_calibration_test_info'
    calibration_files = glob.glob(f'../output/baseline/{pattern}*')
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    for i, calib_file in enumerate(calibration_files):
        filename = os.path.basename(calib_file)
        if filename.startswith(pattern) and filename != f"{pattern}.pkl":
            model_name = filename[len(pattern):].replace('.pkl', '').lstrip('_')
            print(model_name) 
        else: continue
        
        with open(calib_file, 'rb') as f:
            calib_info = pickle.load(f)

        marker = marker_styles[i % len(marker_styles)]
        plt.plot(calib_info['prob_pred'], calib_info['prob_true'], marker=marker, label=f'{model_name} (AUC = {calib_info["ece"]:.3f})', markersize=5, alpha=0.7)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)

    plt.xlabel('Mean Predicted Risk', fontsize=14)
    plt.ylabel('Observed Risk (Fraction of Positives)', fontsize=14)
    if full: plt.legend(loc="lower right", fontsize=9)
    else: plt.legend(loc="lower right", fontsize=10, prop=bold_font)

    output_pdf = f'{exportdir}riskSLIM_calibration_curves{"_median" if median else ""}{"_full" if full else ""}.pdf'
    plt.savefig(output_pdf, format='pdf', dpi=300)
    print(f"Calibration curves saved to {output_pdf}")



if True:
    
    county_list = ['Kern', 'San Francisco']

    for county in county_list:
        if county == 'Kern': table = 'TableKern'
        elif county == 'San Francisco': table = 'TableSF'
        else: raise ValueError("Invalid county name. Choose 'Kern' or 'San Francisco'.")
        filepath = '../output/baseline/'
        roc_files = [
            f'{filepath}riskSLIM_roc_test_info_median_LTOUR_county{county}_{year}.pkl',
            f'{filepath}riskSLIM_roc_test_info_median_{table}_county{county}_{year}.pkl'
        ]

        calibration_files = [
            f'{filepath}riskSLIM_calibration_test_info_median_LTOUR_county{county}_{year}.pkl',
            f'{filepath}riskSLIM_calibration_test_info_median_{table}_county{county}_{year}.pkl'
        ]

        marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        plt.figure(figsize=(10, 6))

        for i, roc_file in enumerate(roc_files):
            try:
                with open(roc_file, 'rb') as f:
                    roc_info = pickle.load(f)
            except FileNotFoundError:
                print(f"File not found: {roc_file}")
                continue

            marker = marker_styles[i % len(marker_styles)]
            model_name = 'LTOUR' if 'LTOUR' in roc_file else table

            plt.plot(roc_info['fpr'], roc_info['tpr'], marker=marker, label=f'{model_name} (AUC = {roc_info["auc"]:.3f})', markersize=6, alpha=0.8, linewidth=2)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('riskSLIM ROC Curves', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        output_pdf = f'{exportdir}riskSLIM_roc_curves_median_{county}_{year}.pdf'
        plt.savefig(output_pdf, format='pdf', dpi=300)
        print(f"ROC curves saved to {output_pdf}")

        # Calibration curves
        plt.figure(figsize=(10, 6))
        for i, calib_file in enumerate(calibration_files):
            try:
                with open(calib_file, 'rb') as f:
                    calib_info = pickle.load(f)
            except FileNotFoundError:
                print(f"Calibration file not found: {calib_file}")
                continue

            marker = marker_styles[i % len(marker_styles)]
            model_name = 'LTOUR' if 'LTOUR' in calib_file else table
            plt.plot(calib_info['prob_pred'], calib_info['prob_true'], marker=marker, label=f'{model_name} (AUC = {calib_info["ece"]:.3f})', markersize=5, alpha=0.7)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)

        plt.xlabel('Mean Predicted Risk', fontsize=14)
        plt.ylabel('Observed Risk (Fraction of Positives)', fontsize=14)
        if full: plt.legend(loc="lower right", fontsize=9)
        else: plt.legend(loc="lower right", fontsize=10, prop=bold_font)

        output_pdf = f'{exportdir}riskSLIM_calibration_curves_median_{county}_{year}.pdf'
        plt.savefig(output_pdf, format='pdf', dpi=300)
        print(f"Calibration curves saved to {output_pdf}")
    
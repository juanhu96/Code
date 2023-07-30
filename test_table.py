import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score

workdir = "/mnt/phd/jihu/opioid_conic/"


# [0, MME, methadone_MME, prescribers, pharmacies, benzo, consecutive_days]


# Basic
cutoffs = {10000: {'original': [0, 9.8, 41, 0, 0, 0, 14], 'quartiles': [0, 45, 0, 0, 0, 1, 16], 'polyapprox': [0, 38, 41, 0, 0, 1, 14], 'IGApolyapprox': [0, 0.375, 40.9, 0, 0, 2, 15], 'IGA': [0, 19.09, 40.9, 0, 0, 0, 14], 'original1e2': [0, 9.78, 40.9, 0, 0, 0, 14]}, 
20000: {'original': [0, 20, 0, 0, 0, 1, 14], 'quartiles': [0, 0, 0, 0, 5, 1, 30], 'polyapprox': [0, 58, 8, 0, 0, 1, 14], 'IGApolyapprox': [0, 0.6, 8, 0, 0, 2, 14], 'IGA': [0, 995, 8, 0, 0, 3, 14], 'original1e2': [0, 29.6, 0, 0, 0, 1, 14]},
50000: {'original': [0, 38, 5, 0, 0, 1, 15], 'quartiles': [0, 0, 0, 0, 0, 1, 30], 'polyapprox': [0, 0, 13, 0, 0, 1, 15], 'IGApolyapprox': [0, 0, 5.3, 0, 0, 2, 15], 'IGA': [0, 0.166, 0, 0, 0, 6, 14], 'original1e2': [0, 38.4, 5.3, 0, 0, 1, 15]}}

scores = {10000: {'original': [-2, 1, 2, 0, 0, 0, 3], 'quartiles': [-2, 1, 0, 0, 0, 1, 3], 'polyapprox': [-3, 1, 5, 0, 0, 4, 5], 'IGApolyapprox': [-7, 5, 5, 0, 0, 5, 5], 'IGA': [-3, 1, 1, 0, 0, 0, 5], 'original1e2': [-2, 1, 2, 0, 0, 0, 3]},
20000: {'original': [-2, 1, 0, 0, 0, 1, 3], 'quartiles': [-1, 0, 1, 0, 5, 1, 3], 'polyapprox': [-3, 1, 5, 0, 0, 4, 5], 'IGApolyapprox': [-6, 4, 5, 0, 0, 5, 5], 'IGA': [-2, 4, 1, 0, 0, 2, 4], 'original1e2': [-2, 1, 0, 0, 0, 1, 3]},
50000: {'original': [-2, 1, 1, 0, 0, 1, 3], 'quartiles': [-1, 0, 1, 0, 0, 1, 3], 'polyapprox': [-2, 0, 5, 0, 0, 5, 5], 'IGApolyapprox': [-3, 0, 5, 0, 0, 5, 5], 'IGA': [-6, 4, 0, 0, 0, 4, 4], 'original1e2': [-2, 1, 1, 0, 0, 1, 3]}}


# Full




def main():
    
    '''
    year = 2018
    N_list = [10000, 20000, 50000]
    model_list = ['original', 'quartiles', 'polyapprox', 'IGA', 'IGApolyapprox', 'original1e2']

    results = []
    for N in N_list:

        SAMPLE = pd.read_csv(f'{workdir}Data/SAMPLE_{str(year)}_LONGTERM_stratified_{str(N)}.csv', delimiter = ",", 
                            dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                    'num_prescribers': int, 'num_pharmacies': int,
                                    'concurrent_benzo': int, 'consecutive_days': int})
        
        SAMPLE = SAMPLE.fillna(0)
        x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',\
                    'num_pharmacies', 'concurrent_benzo', 'consecutive_days']] 
        y = SAMPLE[['long_term_180']].to_numpy().astype('int')

        for model in model_list:
            results.append(test_table(x=x, y=y, model=model, N=N, cutoffs=cutoffs[N][model], scores=scores[N][model]))


    results = pd.DataFrame(results)
    results.to_csv(f'{workdir}Results/test_{str(year)}.csv')
    '''


    year = 2018
    N_list = [10000]
    model_list = ['original', 'IGA']
    results = []
    for N in N_list:

        SAMPLE = pd.read_csv(f'{workdir}Data/SAMPLE_{str(year)}_LONGTERM_stratified_{str(N)}.csv', delimiter = ",", 
                            dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                    'num_prescribers': int, 'num_pharmacies': int,
                                    'concurrent_benzo': int, 'consecutive_days': int})
        
        SAMPLE = SAMPLE.fillna(0)
        x = SAMPLE
        y = SAMPLE[['long_term_180']].to_numpy().astype('int')

        for model in model_list:
            results.append(test_table_full(x=x, y=y, model=model, N=N))
        

    results = pd.DataFrame(results)
    results.to_csv(f'{workdir}Results/test_full_{str(year)}.csv')

    print("Finished.")
    
    

def test_table(x, y, model, N, cutoffs, scores):

    x['Prob'] = x.apply(compute_score, axis=1, args=(cutoffs, scores,))
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    # results = {"N": N, "Model": model,
    # "Accuracy": str(round(accuracy_score(y, y_pred), 4)),
    # "Recall": str(round(recall_score(y, y_pred), 4)),
    # "Precision": str(round(precision_score(y, y_pred), 4)),
    # "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
    # "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    
    results = {"N": N, "Model": model,
    "Accuracy": str(round(accuracy_score(y, y_pred), 4)),
    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4))}

    return results



def compute_score(row, cutoff, scores):
       
    score = 0
    intercept = scores[0]
    
    if row['concurrent_MME'] >= cutoff[1]:
        score += scores[1]
    if row['concurrent_methadone_MME'] >= cutoff[2]:
        score += scores[2]
    if row['num_prescribers'] >= cutoff[3]:
        score += scores[3]
    if row['num_pharmacies'] >= cutoff[4]:
        score += scores[4]
    if row['concurrent_benzo'] >= cutoff[5]:
        score += scores[5]
    if row['consecutive_days'] >= cutoff[6]:
        score += scores[6]
    
    return 1 / (1+np.exp(-(score + intercept)))





def test_table_full(x, y, model, N):

    if model == 'original': x['Prob'] = x.apply(compute_full_original, axis=1)
    if model == 'IGA': x['Prob'] = x.apply(compute_full_IGA, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    # results = {"N": N, "Model": model,
    # "Accuracy": str(round(accuracy_score(y, y_pred), 4)),
    # "Recall": str(round(recall_score(y, y_pred), 4)),
    # "Precision": str(round(precision_score(y, y_pred), 4)),
    # "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
    # "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    
    results = {"N": N, "Model": model,
    "Accuracy": str(round(accuracy_score(y, y_pred), 4)),
    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4))}

    return results



def compute_full_original(row):

    score = 0
    intercept = -5

    if row['concurrent_methadone_MME'] >= 40.9:
        score += 3
    if row['consecutive_days'] >= 15:
        score += 3
    if row['concurrent_benzo'] >= 2:
        score += 1
    if row['age'] >= 25:
        score += 3
    if row['MME_diff'] >= -0.21:
        score += 1

    return 1 / (1+np.exp(-(score + intercept)))


def compute_full_IGA(row):

    score = 0
    intercept = -6

    if row['consecutive_days'] >= 14:
        score += 5
    if row['age'] >= 24:
        score += 3
    if row['MME_diff'] >= -0.21:
        score += 1

    return 1 / (1+np.exp(-(score + intercept)))




if __name__ == "__main__":
    main()
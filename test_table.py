import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score


def main():
    
    # N = 1000
    # test_table(year=2018, cutoffs=[0, 1.65, 10, 5, 0, 2, 14], scores=[-1, 1, 1, 2, 0, 1, 3], filename='IGA')
    # test_table(year=2018, cutoffs=[0, 1.65, 75, 0, 0, 2, 14], scores=[-1, 1, 5, 0, 0, 5, 3], filename='optimal')

    # N = 10000
    test_table(year=2018, cutoffs=[0, 9.78, 30, 0, 0, 0, 14], scores=[-2, 1, 2, 0, 0, 0, 3], filename='IGA')
    test_table(year=2018, cutoffs=[0, 9.78, 40.9, 0, 0, 0, 14], scores=[-2, 1, 2, 0, 0, 0, 3], filename='optimal')
    
    pass
    
    

def test_table(year, cutoffs, scores, filename):
    
    os.chdir('/mnt/phd/jihu/opioid')
    SAMPLE = pd.read_csv('Data/SAMPLE_2018_LONGTERM_stratified_1000.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int})
    
    SAMPLE = SAMPLE.fillna(0)
    x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',\
                'num_pharmacies', 'concurrent_benzo', 'consecutive_days']] 
    y = SAMPLE[['long_term_180']].to_numpy().astype('int')



    x['Prob'] = x.apply(compute_score, axis=1, args=(cutoffs, scores,))
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
               "Recall": str(round(recall_score(y, y_pred), 4)),
               "Precision": str(round(precision_score(y, y_pred), 4)),
               "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
               "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results = pd.DataFrame.from_dict(results, orient='index', columns=['Test'])
    results = results.T
    results.to_csv('Result/results_test_' + str(year) + '_' + filename + '.csv')




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



if __name__ == "__main__":
    main()
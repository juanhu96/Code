import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def test_table(problem, resultdir='/mnt/phd/jihu/opioid_conic/Results_gurobi'):

    dataset, model, N, gamma = problem['data'], problem['model'], problem['N'], problem['gamma']
    x, y = import_dataset(dataset)

    # ============================================================================================
    
    filename = f"{problem['data']}_{problem['model']}"
    if problem['N'] is not None: filename += f"N{problem['N']}"
    scoring_table = pd.read_csv(f'{resultdir}/table_{filename}.csv', delimiter = ",")

    print("*"*20 + 'SCORING TABLE' + "*"*20)
    print(scoring_table)

    # ============================================================================================
    logistic_models = ['soc', 'bestscale']

    if model in logistic_models: x['Prob'] = x.apply(compute_score_logistic, axis=1, args=(scoring_table,))
    else: x['Prob'] = x.apply(compute_score_linear, axis=1, args=(scoring_table, gamma)) 

    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()

    # calibration error:
    calibration_score = compute_calibration(y, y_prob, y_pred)

    results = {"Model": model, "Accuracy": str(round(accuracy_score(y, y_pred), 3)), "ROC AUC": str(round(roc_auc_score(y, y_prob), 3)), "Calibration Error": str(round(calibration_score, 3))}
    results = pd.DataFrame(results, index=[0]) # NOTE: if we have multiple scenarios, get rid of index
    
    # ============================================================================================

    print("*"*20 + 'RESULTS' + "*"*20)
    print(results)

    results.to_csv(f'{resultdir}/Test_{filename}.csv', index=False)

    return 


def import_dataset(dataset, datadir='/mnt/phd/jihu/opioid_conic/Data'):

    if dataset == 'Framingham': 
        SAMPLE = pd.read_csv(f'{datadir}/framingham.csv', delimiter = ",")
        SAMPLE = SAMPLE.fillna(0)
        x, y = SAMPLE, SAMPLE[['TenYearCHD']].to_numpy().astype('int')

    elif dataset == 'Iris':
        iris = fetch_ucirepo(id=53) 
        X, y = iris.data.features, iris.data.targets  
        y = LabelEncoder().fit_transform(y)
        x, y = X[y != 2], y[y != 2]

    elif dataset == 'Breast':
        breast_cancer = fetch_ucirepo(id=14) 
        categorical_features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'breast-quad']
        X, y = breast_cancer.data.features, breast_cancer.data.targets 
        x = X.drop(categorical_features, axis=1)
        binary_features = ["node-caps", "breast", "irradiat"]
        for feature in binary_features:
            x[feature] = LabelEncoder().fit_transform(x[feature])
        y = LabelEncoder().fit_transform(y)
    
    elif dataset == 'Opioid': 
        # SAMPLE = pd.read_csv(f"{datadir}/SAMPLE_2018_LONGTERM_stratified_{N}.csv")
        # SAMPLE = pd.read_csv(f"/mnt/phd/jihu/opioid/Data/FULL_2019_LONGTERM_UPTOFIRST.csv") # NOTE: test
        SAMPLE = SAMPLE.fillna(0)
        x, y = SAMPLE, SAMPLE[['long_term_180']].to_numpy().astype('int')

    else: 
        raise Exception("Dataset cannot be found!")

    return x, y



def compute_score_logistic(row, scoring_table):

    score = 0
    intercept = scoring_table.iloc[0]['point']

    for index, table_row in scoring_table.iterrows():
        
        if index == scoring_table.index[0]: continue

        selected_feature = table_row['selected_feature']
        cutoff = table_row['cutoff']
        point = table_row['point']

        if row[selected_feature] >= cutoff: score += point
    
    return 1 / (1+np.exp(-(score + intercept)))



def compute_score_linear(row, scoring_table, gamma):

    score = 0
    intercept = scoring_table.iloc[0]['point']

    for index, table_row in scoring_table.iterrows():
        
        if index == scoring_table.index[0]: continue

        selected_feature = table_row['selected_feature']
        cutoff = table_row['cutoff']
        point = table_row['point']

        if row[selected_feature] >= cutoff: score += point

    return (score + intercept) / gamma



def compute_calibration(y, y_prob, y_pred):
    
    num_total_presc = len(y)
    calibration_error = 0
    
    for prob in np.unique(y_prob):
        
        y_temp = y[y_prob == prob]
        y_pred_temp = y_pred[y_prob == prob]
        
        # prescription-level results 
        TN, FP, FN, TP = confusion_matrix(y_temp, y_pred_temp, labels=[0,1]).ravel() 
        observed_risk = np.count_nonzero(y_temp == 1) / len(y_temp)
        num_presc = TN + FP + FN + TP

        calibration_error += abs(prob - observed_risk) * num_presc/num_total_presc

    return calibration_error
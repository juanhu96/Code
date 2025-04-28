#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2022
@Author: Jingyuan Hu
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, confusion_matrix, recall_score, precision_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
import matplotlib.pyplot as plt
import statsmodels.api as sm

def cross_validate(model, X, Y, estimator, c_grid, seed, cv=5, partial=False, exportdir='/export/storage_cures/CURES/Results/'):
    
    # Use half of it for hyperparmeter selection
    cv = StratifiedKFold(n_splits=cv, random_state=seed, shuffle=True) 
    if partial:
        X_sub, _, Y_sub, _ = train_test_split(X, Y, test_size=0.5, random_state=seed)
    else:
        X_sub, Y_sub = X, Y

    start = time.time()

    clf = GridSearchCV(estimator=estimator, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cv, 
                       return_train_score=True).fit(X_sub, Y_sub) 
    
    print(f'GridSearchCV for {model}: {str(round(time.time() - start,1))}s\n')

    # best_params = clf.best_params_
    # final_model = estimator.set_params(**best_params)
    
    start = time.time()
    if model == 'LinearSVM':
            # used with classifiers like SVM (LinearSVC) which do not have a predict_proba method by default
            best_model = clf.best_estimator_

            # clf = CalibratedClassifierCV(clf, cv=cv).fit(X, Y)
            # prob = clf.predict_proba(X)[:, 1]
            # pred = clf.predict(X)   

            feature_names = X.columns.tolist()
            selected_features = [feature_names[i] for i in np.where(best_model.coef_.ravel() != 0)[0]]   
            print(f'{len(selected_features)} features selected: {selected_features}') 

            best_model = CalibratedClassifierCV(best_model, cv=cv)
            best_model.fit(X, Y)
            print(f'\nFitting for {model}: {str(round(time.time() - start,1))}s\n')

            prob = best_model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5)

    elif model == 'NN':
        best_model = clf.best_estimator_
        best_model.fit(X, Y)
        print(f'\nFitting for {model}: {str(round(time.time() - start,1))}s\n')

        num_features = X.shape[1]
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5)
        print(f'All {num_features} features are selected as input for neural network\n')

    else:
        best_model = clf.best_estimator_
        best_model.fit(X, Y)
        print(f'\nFitting for {model}: {str(round(time.time() - start,1))}s\n')

        prob = best_model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5)

        feature_names = X.columns.tolist()

        if model == 'DT' or model == 'RF' or model == 'XGB':
            feature_importances = best_model.feature_importances_
            selected_features = [feature_names[i] for i in np.where(feature_importances > 0)[0]]

            if model == 'DT':
                plt.figure(figsize=(25, 25))
                tree.plot_tree(best_model, feature_names=feature_names, filled=True, fontsize=8)
                plt.savefig(f'{exportdir}DecisionTree_CV.png', dpi=150, bbox_inches='tight')
            
        else: # L1, L2
            coefficients = best_model.coef_[0]
            selected_features = [feature_names[i] for i in np.where(coefficients != 0)[0]]

            if model == 'Lasso':
                best_C = clf.best_params_['C']
                logit_model = sm.Logit(Y, X).fit_regularized(method='l1', alpha=best_C)

                coefficients = logit_model.params
                conf = logit_model.conf_int()
                odds_ratios = np.exp(coefficients)
                conf['OR lower'] = conf[0]
                conf['OR upper'] = conf[1]

                results = pd.DataFrame({
                    'Feature': X_sub.columns,  # Use existing column names of X_sub
                    'Coefficient': coefficients,
                    'Odds Ratio': odds_ratios,
                    'CI Lower': conf['OR lower'],
                    'CI Upper': conf['OR upper']
                })
                print(results)
                results.to_csv(f'{exportdir}LogisticRegression_L1.csv', index=False)

        print(f'{len(selected_features)} features selected: {selected_features}')

    results = {'training_accuracy': round(accuracy_score(Y, pred), 3),
               'training_roc_auc': round(roc_auc_score(Y, prob), 3),
               "training_calibration_error": round(compute_calibration(Y, prob, pred), 3)}
    print(results)
    
    return results, best_model, prob, pred



def compute_calibration(y, y_prob, y_pred, n_bins=50, output_table=False):
    
    # Bin the predicted probabilities into `n_bins`
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=True)
    
    # Initialize variables
    num_total_presc = len(y)
    calibration_error = 0
    table = []
    
    # Group by bin index
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        y_temp = y[mask]
        y_pred_temp = y_pred[mask]
        prob = np.mean(y_prob[mask])

        if len(y_temp) == 0:  # Skip empty bins
            continue

        # Confusion matrix components
        TN, FP, FN, TP = confusion_matrix(y_temp, y_pred_temp, labels=[0,1]).ravel() 
        observed_risk = np.mean(y_temp == 1)
        num_presc = TN + FP + FN + TP
        calibration_error += abs(prob - observed_risk) * num_presc / num_total_presc

        table.append({'Prob': prob, 'Num_presc': num_presc,
                      'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                      'Observed Risk': observed_risk})

    if output_table:
        print(pd.DataFrame(table))

    return calibration_error



def compute_patient(FULL, setting_tag, exportdir='/export/storage_cures/CURES/Results/'):
    
    FULL['Pred'] = FULL['Pred'].astype(int)

    # TP prescriptions
    FULL_TP = FULL[(FULL['long_term_180'] == 1) & (FULL['Pred'] == 1)] # presc from TP patient 

    # prescriptions from true positive patients
    TP_PATIENT_ID = FULL_TP['patient_id'].unique()
    FULL = FULL[FULL.patient_id.isin(TP_PATIENT_ID)]
    print("-"*100)
    print(f"Total prescriptions from true positive patients: \n {FULL.shape}")
    print("-"*100)

    if FULL_TP.shape[0] == 0:
        print("No true positive patient found!")
        return

    PATIENT_TP = FULL.groupby('patient_id').apply(lambda x: pd.Series({
        'first_presc_date': x['date_filled'].iloc[0],
        'first_pred_date': x.loc[x['Pred'] == 1, 'date_filled'].iloc[0],
        'first_pred_presc': x.index[x['Pred'] == 1][0] - x.index.min(),
        'first_long_term_180_date': x.loc[x['long_term_180'] == 1, 'date_filled'].iloc[0]
        # 'first_long_term_date': x.loc[x['long_term'] == 1, 'date_filled'].iloc[0] # don't exist if upto first long_term_180
    })).reset_index()
    
    PATIENT_TP = PATIENT_TP.groupby('patient_id').agg(
        first_presc_date=('first_presc_date', 'first'),
        first_pred_date=('first_pred_date', 'first'),
        first_pred_presc=('first_pred_presc', 'first'),
        first_long_term_180_date=('first_long_term_180_date', 'first')
    ).reset_index()    

    # NOTE: we don't have first_long_term_date as we focus up to first presciption
    # PATIENT_TP['day_to_long_term'] = (pd.to_datetime(PATIENT_TP['first_long_term_date'], errors='coerce')
    #                                    - pd.to_datetime(PATIENT_TP['first_pred_date'], errors='coerce')).dt.days

    # PATIENT_TP['day_to_long_term_180'] = (pd.to_datetime(PATIENT_TP['first_long_term_180_date'], errors='coerce')
    #                                         - pd.to_datetime(PATIENT_TP['first_pred_date'], errors='coerce')).dt.days
    
    # main metric how long it takes the model to predict long term
    PATIENT_TP['firstpred_from_firstpresc'] = (pd.to_datetime(PATIENT_TP['first_pred_date'], errors='coerce')
                                                - pd.to_datetime(PATIENT_TP['first_presc_date'], errors='coerce')).dt.days
    
    proportions = {}
    for months in [1, 2, 3]:
        within_month = (PATIENT_TP['firstpred_from_firstpresc'] <= months * 30)
        proportions[months] = round(within_month.mean() * 100, 1)
        
    print(f"Proportion of LT users detected within a month: {proportions[1]}; two months: {proportions[2]}, three months: {proportions[3]}")  
    
    return proportions



def compute_fairness(x, y, y_prob, y_pred, optimal_threshold, setting_tag, plot=False, exportdir='/export/storage_cures/CURES/Results/'):
    
    genders = x['patient_gender'].unique()
    roc_auc_by_gender, accuracy_by_gender, calibration_by_gender = {}, {}, {}
    fig, ax = plt.subplots()

    for gender in genders: # Male: 0, Female: 1
        gender_mask = x['patient_gender'] == gender
        X_gender = x[gender_mask]
        y_true_gender = y[gender_mask]
        y_prob_gender = y_prob[gender_mask]

        # print(y_true_gender)
        # print("-"*100)
        # print(y_prob_gender)
            
        # roc
        fpr, tpr, _ = roc_curve(y_true_gender, y_prob_gender)
        roc_auc = auc(fpr, tpr)
        roc_auc_by_gender[gender] = roc_auc
            
        # accuracy
        y_pred_gender = (y_prob_gender >= optimal_threshold).astype(int)
        assert np.array_equal(y_pred_gender, y_pred[gender_mask])
        accuracy = accuracy_score(y_true_gender, y_pred_gender)
        accuracy_by_gender[gender] = accuracy
        
        # calibration
        calibration_error = compute_calibration(y_true_gender, y_prob_gender, y_pred_gender)
        # _, calibration_error = compute_calibration(X_gender, y_true_gender, y_prob_gender, y_pred_gender, f'{setting_tag}_gender')
        calibration_by_gender[gender] = calibration_error

        ax.plot(fpr, tpr, label=f'{gender} (AUC = {roc_auc:.2f})')

    if plot:
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve by Gender')
        ax.legend(loc='best')
        fig.savefig(f'{exportdir}/Figures/ROC{setting_tag}_gender.pdf', dpi=300)
        print(f"ROC curve saved as ROC{setting_tag}_gender.pdf\n")

    for gender in genders:
        accuracy = accuracy_by_gender.get(gender, 'N/A')
        roc_auc = roc_auc_by_gender.get(gender, 'N/A')
        calibration_error = calibration_by_gender.get(gender, 'N/A')
        print(f'{gender}: Accuracy = {accuracy:.4f}, ROC AUC = {roc_auc:.4f}, Calibration = {calibration_error:.4f}')

    return


def compute_nth_presc(FULL, exportdir='/export/storage_cures/CURES/Results/'):

    FULL['num_prescriptions'] = FULL['num_prior_prescriptions'] + 1
    test_results_by_prescriptions = FULL[FULL['num_prescriptions'] <= 3].groupby('num_prescriptions').apply(lambda x: {'test_accuracy': round(accuracy_score(x['long_term_180'], x['Pred']), 3),
                                                                                                                        'test_recall': round(recall_score(x['long_term_180'], x['Pred']), 3),
                                                                                                                        'test_precision': round(precision_score(x['long_term_180'], x['Pred']), 3),
                                                                                                                        'test_roc_auc': round(roc_auc_score(x['long_term_180'], x['Prob']), 3),
                                                                                                                        'test_pr_auc': round(average_precision_score(x['long_term_180'], x['Prob']), 3),
                                                                                                                        'test_calibration_error': round(compute_calibration(x['long_term_180'], x['Prob'], x['Pred']), 3)}).to_dict()
    print_results(test_results_by_prescriptions)

    return



def compute_MME_presc(FULL, exportdir='/export/storage_cures/CURES/Results/'):

    cutoffs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    bin_labels = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', 
    '[60, 70)', '[70, 80)', '[80, 90)', '[90, 100)', 'above 100']

    MME_bins = pd.cut(FULL['concurrent_MME'], bins=cutoffs, labels=bin_labels, right=False)
    FULL['MME_bins'] = MME_bins
    test_results_by_MME = FULL.groupby('MME_bins').apply(lambda x: {'test_accuracy': accuracy_score(x['long_term_180'], x['Pred']),
                                                                    'test_recall': recall_score(x['long_term_180'], x['Pred']),           
                                                                    'test_roc_auc': roc_auc_score(x['long_term_180'], x['Prob']),
                                                                    'test_pr_auc': average_precision_score(x['long_term_180'], x['Prob']),
                                                                    'test_calibration_error': compute_calibration(x['long_term_180'], x['Prob'], x['Pred']),
                                                                    'correctly_predicted_positives_ratio': ((x['Pred'] == 1) & (x['long_term_180'] == 1)).sum() / len(x)}).to_dict()

    print_results(test_results_by_MME)

    return test_results_by_MME



def print_results(results):
    # print results of dict in seperate rows
    for key, value in results.items():
        print(key)
        print(value)   
    print('\n')
    return
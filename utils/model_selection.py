#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2022
@Author: Jingyuan Hu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
    average_precision_score, brier_score_loss, fbeta_score, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
import matplotlib.pyplot as plt



def nested_cross_validate(X, Y, estimator, c_grid, seed, model, n_splits=5, index=None, plot_DT=True, resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    best_params = []
    train_auc = []
    validation_auc = []
    auc_diffs = []
    
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_accuracy = []
    holdout_recall = []
    holdout_precision = []
    holdout_roc_auc = []
    holdout_pr_auc = []
    holdout_f1 = []
    holdout_f2 = []
    holdout_brier = []
    holdout_calibration = []
    
    ## inner cv
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
      
        '''
        ### Jingyuan: to specify grid on estimator, need to add 'estimator__' in c_grid in baseline_functions.py
        pipeline = Pipeline(steps=[('sampler', SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
                                    ('estimator', estimator)])

        clf = GridSearchCV(estimator=pipeline, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv, 
                           return_train_score=True).fit(train_x, train_y) 
        '''
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, 
                            param_grid=c_grid, 
                            scoring='roc_auc',
                            cv=inner_cv, 
                            return_train_score=True).fit(train_x, train_y) 
    
        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        train_auc.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]])
        validation_auc.append(clf.best_score_)
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
            holdout_acc = best_model.score(test_x, test_y)            
        else:
            prob = clf.predict_proba(test_x)[:, 1]
            holdout_pred = clf.predict(test_x)
            holdout_acc = clf.score(test_x, test_y)
        
        ## store results
        best_params.append(best_param)
        holdout_probability.append(prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_accuracy.append(accuracy_score(test_y, holdout_pred))
        holdout_recall.append(recall_score(test_y, holdout_pred))
        holdout_precision.append(precision_score(test_y, holdout_pred))
        holdout_roc_auc.append(roc_auc_score(test_y, prob))
        holdout_pr_auc.append(average_precision_score(test_y, prob))
        holdout_brier.append(brier_score_loss(test_y, prob))
        holdout_f1.append(fbeta_score(test_y, holdout_pred, beta = 1))
        holdout_f2.append(fbeta_score(test_y, holdout_pred, beta = 2))
    
        # NEW: calibration error
        holdout_calibration.append(compute_calibration(test_y, prob, holdout_pred))


        best_estimator = clf.best_estimator_
        if model == 'XGB':
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            print(f'XGB iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'RF':
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            print(f'RF iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'LinearSVM':
            # create a SelectFromModel object based on the trained model
            sfm = SelectFromModel(best_estimator)
            # fit the SelectFromModel object to the training data
            sfm.fit(train_x, train_y)
            # get the number of selected features
            num_selected = np.sum(sfm.get_support())
            # print('LinearSVM iteration ' + str(i) + '...' + str(num_selected) + '\n')
            print(f'LinearSVM iteration {str(i)}\n Indices: {num_selected}\n Number of features: {num_selected}\n')

        elif model == 'Lasso':
            coefficients = best_estimator.coef_[0]
            nonzero_indices = np.nonzero(coefficients)[0]
            # print('Lasso iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            # print(coefficients)
            print(f'Lasso (L1) iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Coefficients: {coefficients}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'Logistic':
            coefficients = best_estimator.coef_[0]
            nonzero_indices = np.nonzero(coefficients)[0]
            # print('Logistic iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            # print(coefficients)
            print(f'Logistic (L2) iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Coefficients: {coefficients}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'DT':
            
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            # print('DT iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            print(f'DT iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')
            print(best_estimator.tree_)

            # plot
            if plot_DT:
                plt.figure(figsize=(30, 30))
                tree.plot_tree(best_estimator)
                plt.savefig(f'{resultdir}DecisionTree_CV.png', dpi=300)


    return {'best_param': best_params,
            'train_auc': train_auc,
            'validation_auc': validation_auc,
            'auc_diffs': auc_diffs,
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2,
            "holdout_calibration_error": holdout_calibration}



def cross_validate(model, X, Y, estimator, c_grid, seed, cv=3, exportdir='/export/storage_cures/CURES/Results/'):
    
    # temp change to 3 for computational efficiency
    cv = StratifiedKFold(n_splits=cv, random_state=seed, shuffle=True) 
    clf = GridSearchCV(estimator=estimator, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cv, 
                       return_train_score=True).fit(X, Y) 
    
    if model == 'LinearSVM':
            # used with classifiers like SVM (LinearSVC) which do not have a predict_proba method by default
            best_model = clf.best_estimator_
            clf = CalibratedClassifierCV(clf, cv=cv).fit(X, Y)
            prob = clf.predict_proba(X)[:, 1]
            pred = clf.predict(X)     

            feature_names = X.columns.tolist()
            selected_features = [feature_names[i] for i in np.where(best_model.coef_.ravel() != 0)[0]]   
            print(f'{len(selected_features)} features selected: {selected_features}') 
    
    elif model == 'NN':
        best_model = clf.best_estimator_
        num_features = X.shape[1]
        pred = clf.predict(X)
        prob = clf.predict_proba(X)[:, 1]
        print(f'All {num_features} features are selected as input for neural network\n')

    else:
        pred = clf.predict(X)
        prob = clf.predict_proba(X)[:, 1]
        
        best_model = clf.best_estimator_
        feature_names = X.columns.tolist()

        if model == 'DT' or model == 'RF' or model == 'XGB':
            feature_importances = best_model.feature_importances_
            selected_features = [feature_names[i] for i in np.where(feature_importances > 0)[0]]

            if model == 'DT':
                plt.figure(figsize=(30, 30))
                tree.plot_tree(best_model)
                plt.savefig(f'{exportdir}DecisionTree_CV.png', dpi=300)
            
        else: # L1, L2
            coefficients = best_model.coef_[0]
            selected_features = [feature_names[i] for i in np.where(coefficients != 0)[0]]

        print(f'{len(selected_features)} features selected: {selected_features}')

    results = {'training_accuracy': accuracy_score(Y, pred),
               'training_recall': recall_score(Y, pred),
               "training_precision": precision_score(Y, pred),
               'training_roc_auc': roc_auc_score(Y, prob),
               'training_pr_auc': average_precision_score(Y, prob),
               "training_calibration_error": compute_calibration(Y, prob, pred)}
    
    print(results)
    
    return results, best_model
    
    

def compute_calibration(y, y_prob, y_pred, output_table=False):

    table = []
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

        table.append({'Prob': prob, 'Num_presc': num_presc,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'Observed Risk': observed_risk})

    if output_table: print(table)

    return calibration_error
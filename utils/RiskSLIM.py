#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14, 2022
@Author: Jingyuan Hu

Got rid of unconstrained RiskSLIM, regular CV, and unfound functions
"""

import os
import numpy as np
import pandas as pd
# os.chdir('/Users/jingyuanhu/Desktop/Research/Interpretable Opioid/Code')
os.chdir('/mnt/phd/jihu/opioid/Code')

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
    average_precision_score, brier_score_loss, fbeta_score, accuracy_score
from sklearn.utils import shuffle

from pprint import pprint
from riskslim import load_data_from_csv, print_model
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa
from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness,\
    fairness_in_auc, balance_positive_negative


def riskslim_prediction(X, feature_name, model_info):
    
    """
    @parameters
    
    X: test input features (np.array)
    feature_name: feature names
    model_info: output from RiskSLIM model
    
    """
    
    ## initialize parameters
    dictionary = {}
    prob = np.zeros(len(X))
    scores = np.zeros(len(X))
    
    ## prepare statistics
    subtraction_score = model_info['solution'][0]
    coefs = model_info['solution'][1:]
    index = np.where(coefs != 0)[0]
    
    nonzero_coefs = coefs[index]
    features = feature_name[index]
    X_sub = X[:,index]
    
    ## build dictionaries
    for i in range(len(features)):
        single_feature = features[i]
        coef = nonzero_coefs[i]
        dictionary.update({single_feature: coef})
        
    ## calculate probability
    for i in range(len(X_sub)):
        summation = 0
        for j in range(len(features)):
            a = X_sub[i,j]
            summation += dictionary[features[j]] * a
        scores[i] = summation
    
    prob = 1/(1+np.exp(-(scores + subtraction_score)))
    
    return prob



def riskslim_accuracy(X, Y, feature_name, model_info, threshold=0.5):
    
    prob = riskslim_prediction(X, feature_name, model_info)
    pred = np.mean((prob > threshold) == Y)
    
    return pred


###############################################################################################################################
###########################################     constrained RiskSLIM     ######################################################
###############################################################################################################################

    
def risk_slim_constrain(data, max_coefficient, max_L0_value, c0_value, max_offset, 
                        class_weight=None, new_constraints=None, new_constraints_multiple=None, essential_constraints=None,
                        max_runtime = 120, w_pos = 1):
    
    """
    @parameters:
    
    max_coefficient:  value of largest/smallest coefficient
    max_L0_value:     maximum model size (set as float(inf))
    max_offset:       maximum value of offset parameter (optional)
    c0_value:         L0-penalty parameter such that c0_value > 0; larger values -> 
                      sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    max_runtime:      max algorithm running time
    w_pos:            relative weight on examples with y = +1; w_neg = 1.00 (optional)
    
    class_weight: weight of positive outcomes
    new_constraints: new operatoinal constraints, 2d-list
    
    """
    
    # create coefficient set and set the value of the offset parameter
    coef_set = CoefficientSet(variable_names = data['variable_names'], lb = 0, ub = max_coefficient, sign = 0)
    # JH: get_conservative_offset is not defined
    # conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)   
    # max_offset = min(max_offset, conservative_offset)

    '''
    coef_set['(Intercept)'].ub = max_offset
    coef_set['(Intercept)'].lb = -max_offset
    '''
    coef_set['(Intercept)'].lb = -max_offset-0.5 # JH: if we have ub first this would lead to assertion error
    coef_set['(Intercept)'].ub = -max_offset+0.5

    # coef_set.update_intercept_bounds(X = data['X'], y = data['Y'], max_offset = max_offset)
    
    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # Jingyuan: scale positive weight if balanced training
    if class_weight == 'balanced':
        w_pos = sum(data['Y']==-1)[0] / sum(data['Y']==1)[0]
    elif class_weight == 'positive':
        w_pos = 2 * sum(data['Y']==-1)[0] / sum(data['Y']==1)[0]
    elif class_weight == 'positive_2':
        w_pos = 4 * sum(data['Y']==-1)[0] / sum(data['Y']==1)[0]
    elif class_weight == 'positive_4':
        w_pos = 8 * sum(data['Y']==-1)[0] / sum(data['Y']==1)[0]
        
    print('The weight is for positive' + str(w_pos))
    
    # Set parameters
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        # 'c1_value': c1_value,                               # Jingyuan: term for additive stumps
        'w_pos': w_pos,

        # LCPA Settings
        'max_runtime': max_runtime,                         # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': False,                    # print CPLEX progress on screen
        'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
        
        # LCPA Improvements
        'round_flag': False,                                # round continuous solutions with SeqRd
        'polish_flag': False,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': False,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        
        # Initialization
        'initialization_flag': True,                        # use initialization procedure
        'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,

        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings, 
                                                      new_constraints=new_constraints,
                                                      new_constraints_multiple=new_constraints_multiple,
                                                      essential_constraints=essential_constraints)
        
    return model_info, mip_info, lcpa_info


def risk_cv_constrain(X, 
                      Y,
                      y_label, 
                      max_coef,
                      max_coef_number,
                      c,
                      seed,
                      max_runtime=1000,
                      max_offset=100,
                      class_weight = None,
                      new_constraints = None):

    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    
    holdout_with_attr_test = []
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
    
    best_score = 0
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        print('Number of positive in training set: ' + str(len(outer_train_y[outer_train_y==1])))
        print('Number of positive in testing set: ' + str(len(outer_test_y[outer_test_y==1])))
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)            
        cols = outer_train_x.columns.tolist()        
        
        ## outer loop
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              c0_value=c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset=max_offset,
                                                              class_weight=class_weight,
                                                              new_constraints=new_constraints)
        print_model(model_info['solution'], new_train_data)  

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)
        
        print('Number of predicted positive: ' + str(len(outer_test_pred[outer_test_pred==1])))
        
        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob)) 
        
        ########################
        ## store results
        # holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        holdout_accuracy.append(accuracy_score(outer_test_y, outer_test_pred))
        holdout_recall.append(recall_score(outer_test_y, outer_test_pred))
        holdout_precision.append(precision_score(outer_test_y, outer_test_pred))
        holdout_roc_auc.append(roc_auc_score(outer_test_y, outer_test_prob))
        holdout_pr_auc.append(average_precision_score(outer_test_y, outer_test_prob))
        holdout_brier.append(brier_score_loss(outer_test_y, outer_test_prob))
        holdout_f1.append(fbeta_score(outer_test_y, outer_test_pred, beta = 1))
        holdout_f2.append(fbeta_score(outer_test_y, outer_test_pred, beta = 2))
        
        i += 1
        
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'holdout_test_auc': test_auc, 
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2}

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

def risk_nested_cv_constrain(X, 
                             Y,
                             y_label, 
                             max_coef,
                             max_coef_number,
                             c,
                             seed,
                             max_runtime=1000,
                             max_offset=5,
                             score = 'roc_auc',
                             class_weight = None,
                             new_constraints = None,
                             new_constraints_multiple = None,
                             fairness_indicator = 0,
                             y_alert_1=None, 
                             y_alert_2=None,
                             y_alert_3=None,
                             y_alert_4=None,
                             y_alert_5=None,
                             y_alert_6=None):

    """
    Implemented the nested cross-validation step for hyperparameter tuning
    For each inner_cv, pick one combination
    Find the best set of hyperparameter, fit it for outer_cv
    
    This leads to #combinations * 5 inner * 5 outer
    """
    
    
    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    
    holdout_with_attr_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_train_accuracy = []
    holdout_accuracy = []
    holdout_recall = []
    holdout_precision = []
    holdout_roc_auc = []
    holdout_pr_auc = []
    holdout_f1 = []
    holdout_f2 = []
    holdout_brier = []
    
    '''
    ## per alert type
    holdout1_accuracy, holdout2_accuracy, holdout3_accuracy, holdout4_accuracy, holdout5_accuracy, holdout6_accuracy = [], [], [], [], [], []
    holdout1_recall, holdout2_recall, holdout3_recall, holdout4_recall, holdout5_recall, holdout6_recall = [], [], [], [], [], []
    holdout1_precision, holdout2_precision, holdout3_precision, holdout4_precision, holdout5_precision, holdout6_precision = [], [], [], [], [], []
    holdout1_roc_auc, holdout2_roc_auc, holdout3_roc_auc, holdout4_roc_auc, holdout5_roc_auc, holdout6_roc_auc = [], [], [], [], [], []
    holdout1_pr_auc, holdout2_pr_auc, holdout3_pr_auc, holdout4_pr_auc, holdout5_pr_auc, holdout6_pr_auc = [], [], [], [], [], []
    '''
    
    # Fairness
    confusion_matrix_rets = []
    calibrations = []
    gender_auc = []
    condition_pn = []
    no_condition_pn = []
    
    best_score = 0
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        '''
        # results per alert type
        outer_test_y_alert_1 = y_alert_1[outer_test]
        outer_test_y_alert_2 = y_alert_2[outer_test]
        outer_test_y_alert_3 = y_alert_3[outer_test]
        outer_test_y_alert_4 = y_alert_4[outer_test]
        outer_test_y_alert_5 = y_alert_5[outer_test]
        outer_test_y_alert_6 = y_alert_6[outer_test]
        '''
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)
        '''
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'patient_gender1': 'patient_gender'})

        ## remove unused feature in modeling
        if fairness_indicator == 1:
            outer_train_x = outer_train_x.drop(['patient_gender1'], axis=1)
            outer_test_x = outer_test_x.drop(['patient_gender1'], axis=1)
        else:
            outer_train_x = outer_train_x.drop(['patient_gender1'], axis=1)
            outer_test_x = outer_test_x.drop(['patient_gender1'], axis=1)
        ''' 
        
        cols = outer_train_x.columns.tolist()        
        
        ################################################
        
        # for k in range(len(max_coef_number)):
        for k in range(len(c)):
            ## for each possible hyperparameter value, do inner cross validation
            performance_metric = []
            
            # print('Start CV with hyperparameter ' + str(max_coef_number[k]))
            print('Start CV with hyperparameter ' + str(c[k]))
            
            for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
                
                ## subset train data & store test data
                inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
                validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
                inner_train_sample_weight = outer_train_sample_weight[inner_train]
                validation_sample_weight = outer_train_sample_weight[validation]
                inner_train_y = inner_train_y.reshape(-1,1)
           
                ## create new data dictionary
                new_train_data = {
                    'X': inner_train_x,
                    'Y': inner_train_y,
                    'variable_names': cols,
                    'outcome_name': y_label,
                    'sample_weights': inner_train_sample_weight
                }
                
                
                ## fit the model: CV for max_coef_number
                # model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                #                                                       max_coefficient=max_coef, 
                #                                                       max_L0_value=max_coef_number[k], 
                #                                                       c0_value=c, 
                #                                                       max_runtime=max_runtime, 
                #                                                       max_offset=max_offset,
                #                                                       class_weight=class_weight,
                #                                                       new_constraints=new_constraints)
                
                model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                                      max_coefficient=max_coef, 
                                                                      max_L0_value=max_coef_number, 
                                                                      c0_value=c[k], 
                                                                      max_runtime=max_runtime, 
                                                                      max_offset=max_offset,
                                                                      class_weight=class_weight,
                                                                      new_constraints=new_constraints,
                                                                      new_constraints_multiple=new_constraints_multiple)
                
                ## check validation auc
                validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
                validation_y[validation_y == -1] = 0 ## change -1 to 0
                validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
                validation_pred = (validation_prob > 0.5)
                validation_auc.append(roc_auc_score(validation_y, validation_prob))
                
                ## Jingyuan: store the model_info with best performance (e.g. AUC)
                if score == 'accuracy':
                    performance_metric.append(accuracy_score(validation_y, validation_pred))
                elif score == 'recall':
                    performance_metric.append(recall_score(validation_y, validation_pred))
                elif score == 'precision':
                    performance_metric.append(precision_score(validation_y, validation_pred))
                elif score == 'roc_auc':
                    performance_metric.append(roc_auc_score(validation_y, validation_prob))
                elif score == 'pr_auc':
                    performance_metric.append(average_precision_score(validation_y, validation_prob))
                else:
                    print("Score undefined")
            
            ## average score of a hyperparameter value over inner cv
            current_score = sum(performance_metric) / len(performance_metric)
            
            ## find the best hyperparameter
            if current_score >= best_score:
                # best_max_coef_number = max_coef_number[k]
                best_regularized_c = c[k]
                best_score = current_score
            
        ################################################
        ## outer loop: use best model_info
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        ## fit the model
        # model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
        #                                                       max_coefficient=max_coef, 
        #                                                       max_L0_value=best_max_coef_number, 
        #                                                       c0_value=c, 
        #                                                       max_runtime=max_runtime, 
        #                                                       max_offset=max_offset,
        #                                                       class_weight=class_weight,
        #                                                       new_constraints=new_constraints)
        
        
        print('The best c selected through inner CV is ' + str(best_regularized_c))
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              c0_value=best_regularized_c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset=max_offset,
                                                              class_weight=class_weight,
                                                              new_constraints=new_constraints,
                                                              new_constraints_multiple=new_constraints_multiple)
        
        print_model(model_info['solution'], new_train_data)  

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        '''
        # per type
        outer_test_y_alert_1[outer_test_y_alert_1 == -1] = 0
        outer_test_y_alert_2[outer_test_y_alert_2 == -1] = 0
        outer_test_y_alert_3[outer_test_y_alert_3 == -1] = 0
        outer_test_y_alert_4[outer_test_y_alert_4 == -1] = 0
        outer_test_y_alert_5[outer_test_y_alert_5 == -1] = 0
        outer_test_y_alert_6[outer_test_y_alert_6 == -1] = 0
        '''
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        # outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info)
        outer_train_pred = (outer_train_prob > 0.5)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)

        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob)) 
        
        ########################
        ## store results
        # holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        holdout_train_accuracy.append(accuracy_score(outer_train_y, outer_train_pred))
        holdout_accuracy.append(accuracy_score(outer_test_y, outer_test_pred))
        holdout_recall.append(recall_score(outer_test_y, outer_test_pred))
        holdout_precision.append(precision_score(outer_test_y, outer_test_pred))
        holdout_roc_auc.append(roc_auc_score(outer_test_y, outer_test_prob))
        holdout_pr_auc.append(average_precision_score(outer_test_y, outer_test_prob))
        holdout_brier.append(brier_score_loss(outer_test_y, outer_test_prob))
        holdout_f1.append(fbeta_score(outer_test_y, outer_test_pred, beta = 1))
        holdout_f2.append(fbeta_score(outer_test_y, outer_test_pred, beta = 2))
        
        '''
        ## store results per alert type
        holdout1_accuracy.append(accuracy_score(outer_test_y_alert_1, outer_test_pred))
        holdout1_recall.append(recall_score(outer_test_y_alert_1, outer_test_pred))
        holdout1_precision.append(precision_score(outer_test_y_alert_1, outer_test_pred))
        holdout1_roc_auc.append(roc_auc_score(outer_test_y_alert_1, outer_test_prob))
        holdout1_pr_auc.append(average_precision_score(outer_test_y_alert_1, outer_test_prob))
        
        holdout2_accuracy.append(accuracy_score(outer_test_y_alert_2, outer_test_pred))
        holdout2_recall.append(recall_score(outer_test_y_alert_2, outer_test_pred))
        holdout2_precision.append(precision_score(outer_test_y_alert_2, outer_test_pred))
        holdout2_roc_auc.append(roc_auc_score(outer_test_y_alert_2, outer_test_prob))
        holdout2_pr_auc.append(average_precision_score(outer_test_y_alert_2, outer_test_prob))
        
        holdout3_accuracy.append(accuracy_score(outer_test_y_alert_3, outer_test_pred))
        holdout3_recall.append(recall_score(outer_test_y_alert_3, outer_test_pred))
        holdout3_precision.append(precision_score(outer_test_y_alert_3, outer_test_pred))
        holdout3_roc_auc.append(roc_auc_score(outer_test_y_alert_3, outer_test_prob))
        holdout3_pr_auc.append(average_precision_score(outer_test_y_alert_3, outer_test_prob))
        
        holdout4_accuracy.append(accuracy_score(outer_test_y_alert_4, outer_test_pred))
        holdout4_recall.append(recall_score(outer_test_y_alert_4, outer_test_pred))
        holdout4_precision.append(precision_score(outer_test_y_alert_4, outer_test_pred))
        holdout4_roc_auc.append(roc_auc_score(outer_test_y_alert_4, outer_test_prob))
        holdout4_pr_auc.append(average_precision_score(outer_test_y_alert_4, outer_test_prob))
        
        holdout5_accuracy.append(accuracy_score(outer_test_y_alert_5, outer_test_pred))
        holdout5_recall.append(recall_score(outer_test_y_alert_5, outer_test_pred))
        holdout5_precision.append(precision_score(outer_test_y_alert_5, outer_test_pred))
        holdout5_roc_auc.append(roc_auc_score(outer_test_y_alert_5, outer_test_prob))
        holdout5_pr_auc.append(average_precision_score(outer_test_y_alert_5, outer_test_prob))
        
        holdout6_accuracy.append(accuracy_score(outer_test_y_alert_6, outer_test_pred))
        holdout6_recall.append(recall_score(outer_test_y_alert_6, outer_test_pred))
        holdout6_precision.append(precision_score(outer_test_y_alert_6, outer_test_pred))
        holdout6_roc_auc.append(roc_auc_score(outer_test_y_alert_6, outer_test_prob))
        holdout6_pr_auc.append(average_precision_score(outer_test_y_alert_6, outer_test_prob))
        '''
        
        ########################
        '''
        ## Fairness results
        # confusion matrix
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds=outer_test_pred,
                                                                   labels=outer_test_y, 
                                                                   protected_variables=["patient_gender"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        # calibration matrix
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=outer_test_prob, 
                                                   labels=outer_test_y, 
                                                   protected_variables=["patient_gender"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)
        
        # gender auc
        try:
            gender_auc_matrix = fairness_in_auc(df = holdout_with_attrs,
                                                probs = outer_test_prob,
                                                labels = outer_test_y)
            gender_auc_matrix_final = gender_auc_matrix.assign(fold_num = [i]*gender_auc_matrix['Attribute'].count())
            gender_auc.append(gender_auc_matrix_final)
        except:
            pass
        
        # ebm_pn
        no_condition_pn_matrix = balance_positive_negative(df = holdout_with_attrs,
                                                           probs = outer_test_prob, 
                                                           labels = outer_test_y)
        no_condition_pn_matrix_final = no_condition_pn_matrix.assign(fold_num = [i]*no_condition_pn_matrix['Attribute'].count())
        no_condition_pn.append(no_condition_pn_matrix_final)
        
        # # ebm_condition_pn
        # condition_pn_matrix = conditional_balance_positive_negative(df = holdout_with_attrs,
        #                                                                      probs = outer_test_prob, 
        #                                                                      labels = outer_test_y)
        # condition_pn_matrix_final = condition_pn_matrix.assign(fold_num = [i]*condition_pn_matrix['Attribute'].count())
        # condition_pn.append(condition_pn_matrix_final)   
        '''
        ########################
        
        i += 1
    
    
    ######################### Outer iteration done ############################
    '''
    ## confusion matrix
    confusion_df = pd.concat(confusion_matrix_rets, ignore_index=True)
    confusion_df.sort_values(["Attribute", "Attribute Value"], inplace=True)
    confusion_df = confusion_df.reset_index(drop=True)
    
    ## calibration matrix
    calibration_df = pd.concat(calibrations, ignore_index=True)
    calibration_df.sort_values(["Attribute", "Lower Limit Score", "Upper Limit Score"], inplace=True)
    calibration_df = calibration_df.reset_index(drop=True)
    
    ## gender
    gender_auc_df = []
    try:
        gender_auc_df = pd.concat(gender_auc, ignore_index=True)
        gender_auc_df.sort_values(["fold_num", "Attribute"], inplace=True)
        gender_auc_df = gender_auc_df.reset_index(drop=True)
    except:
        pass
    
    ## no_condition_pn
    no_condition_pn_df = pd.concat(no_condition_pn, ignore_index=True)
    no_condition_pn_df.sort_values(["fold_num", "Attribute"], inplace=True)
    no_condition_pn_df = no_condition_pn_df.reset_index(drop=True)
    
    ## condition_pn
    # condition_pn_df = pd.concat(condition_pn, ignore_index=True)
    # condition_pn_df.sort_values(["fold_num", "Attribute"], inplace=True)
    # condition_pn_df = condition_pn_df.reset_index(drop=True)
    '''
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'holdout_test_auc': test_auc, 
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2}

    # return {'train_auc': train_auc,
    #         'validation_auc': validation_auc,
    #         'holdout_test_auc': test_auc, 
    #         'holdout_train_accuracy': holdout_train_accuracy,
    #         'holdout_test_accuracy': holdout_accuracy,
    #         'holdout_test_recall': holdout_recall,
    #         'holdout_test_precision': holdout_precision,
    #         'holdout_test_roc_auc': holdout_roc_auc,
    #         'holdout_test_pr_auc': holdout_pr_auc,
    #         'holdout_test_brier': holdout_brier,
    #         'holdout_test_f1': holdout_f1,
    #         'holdout_test_f2': holdout_f2,
    #         'holdout_test_accuracy_alert1': holdout1_accuracy,
    #         'holdout_test_recall_alert1': holdout1_recall,
    #         'holdout_test_precision_alert1': holdout1_precision,
    #         'holdout_test_roc_auc_alert1': holdout1_roc_auc,
    #         'holdout_test_pr_auc_alert1': holdout1_pr_auc,
    #         'holdout_test_accuracy_alert2': holdout2_accuracy,
    #         'holdout_test_recall_alert2': holdout2_recall,
    #         'holdout_test_precision_alert2': holdout2_precision,
    #         'holdout_test_roc_auc_alert2': holdout2_roc_auc,
    #         'holdout_test_pr_auc_alert2': holdout2_pr_auc,
    #         'holdout_test_accuracy_alert3': holdout3_accuracy,
    #         'holdout_test_recall_alert3': holdout3_recall,
    #         'holdout_test_precision_alert3': holdout3_precision,
    #         'holdout_test_roc_auc_alert3': holdout3_roc_auc,
    #         'holdout_test_pr_auc_alert3': holdout3_pr_auc,
    #         'holdout_test_accuracy_alert4': holdout4_accuracy,
    #         'holdout_test_recall_alert4': holdout4_recall,
    #         'holdout_test_precision_alert4': holdout4_precision,
    #         'holdout_test_roc_auc_alert4': holdout4_roc_auc,
    #         'holdout_test_pr_auc_alert4': holdout4_pr_auc,
    #         'holdout_test_accuracy_alert5': holdout5_accuracy,
    #         'holdout_test_recall_alert5': holdout5_recall,
    #         'holdout_test_precision_alert5': holdout5_precision,
    #         'holdout_test_roc_auc_alert5': holdout5_roc_auc,
    #         'holdout_test_pr_auc_alert5': holdout5_pr_auc,    
    #         'holdout_test_accuracy_alert6': holdout6_accuracy,
    #         'holdout_test_recall_alert6': holdout6_recall,
    #         'holdout_test_precision_alert6': holdout6_precision,
    #         'holdout_test_roc_auc_alert6': holdout6_roc_auc,
    #         'holdout_test_pr_auc_alert6': holdout6_pr_auc
    #         }


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
'''
riskSLIM with bagging

'''



def risk_nested_cv_constrain_bagging(X, 
                                     Y,
                                     y_label, 
                                     max_coef,
                                     max_coef_number,
                                     c,
                                     seed,
                                     max_runtime=1000,
                                     max_offset=100,
                                     score = 'roc_auc',
                                     class_weight = None,
                                     new_constraints = None):
    
    '''
    Nested cross validation leads to five submodels
    Bag the five submodels for prediction (requires a list that stores five models)
    '''
    
    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    
    holdout_with_attr_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_train_accuracy = []
    holdout_accuracy = []
    holdout_recall = []
    holdout_precision = []
    holdout_roc_auc = []
    holdout_pr_auc = []

    
    best_score = 0
    model_list = [] # store the submodels
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)
        
        cols = outer_train_x.columns.tolist()        
        
        ################################################
        
        for k in range(len(c)):
            ## for each possible hyperparameter value, do inner cross validation
            performance_metric = []
            
            # print('Start CV with hyperparameter ' + str(max_coef_number[k]))
            print('Start CV with hyperparameter ' + str(c[k]))
            
            for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
                
                ## subset train data & store test data
                inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
                validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
                inner_train_sample_weight = outer_train_sample_weight[inner_train]
                validation_sample_weight = outer_train_sample_weight[validation]
                inner_train_y = inner_train_y.reshape(-1,1)
           
                ## create new data dictionary
                new_train_data = {
                    'X': inner_train_x,
                    'Y': inner_train_y,
                    'variable_names': cols,
                    'outcome_name': y_label,
                    'sample_weights': inner_train_sample_weight
                }
                
                model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                                      max_coefficient=max_coef, 
                                                                      max_L0_value=max_coef_number, 
                                                                      c0_value=c[k], 
                                                                      max_runtime=max_runtime, 
                                                                      max_offset=max_offset,
                                                                      class_weight=class_weight,
                                                                      new_constraints=new_constraints)
                
                ## check validation auc
                validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
                validation_y[validation_y == -1] = 0 ## change -1 to 0
                validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
                validation_pred = (validation_prob > 0.5)
                validation_auc.append(roc_auc_score(validation_y, validation_prob))
                
                ## Jingyuan: store the model_info with best performance (e.g. AUC)
                if score == 'accuracy':
                    performance_metric.append(accuracy_score(validation_y, validation_pred))
                elif score == 'recall':
                    performance_metric.append(recall_score(validation_y, validation_pred))
                elif score == 'precision':
                    performance_metric.append(precision_score(validation_y, validation_pred))
                elif score == 'roc_auc':
                    performance_metric.append(roc_auc_score(validation_y, validation_prob))
                elif score == 'pr_auc':
                    performance_metric.append(average_precision_score(validation_y, validation_prob))
                else:
                    print("Score undefined")
            
            ## average score of a hyperparameter value over inner cv
            current_score = sum(performance_metric) / len(performance_metric)
            
            ## find the best hyperparameter
            if current_score >= best_score:
                best_regularized_c = c[k]
                best_score = current_score
            
        ################################################
        ## outer loop: use best model_info (best_regularized_c)
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
        
        
        print('The best c selected through inner CV is ' + str(best_regularized_c))
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              c0_value=best_regularized_c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset=max_offset,
                                                              class_weight=class_weight,
                                                              new_constraints=new_constraints)
        ## print and store the model
        print_model(model_info['solution'], new_train_data)  
        model_list.append(model_info)

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0

        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_train_pred = (outer_train_prob > 0.5)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)
        
        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob)) 
        
        ########################
        ## store results
        # holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        holdout_train_accuracy.append(accuracy_score(outer_train_y, outer_train_pred))
        holdout_accuracy.append(accuracy_score(outer_test_y, outer_test_pred))
        holdout_recall.append(recall_score(outer_test_y, outer_test_pred))
        holdout_precision.append(precision_score(outer_test_y, outer_test_pred))
        holdout_roc_auc.append(roc_auc_score(outer_test_y, outer_test_prob))
        holdout_pr_auc.append(average_precision_score(outer_test_y, outer_test_prob))

        i += 1
        
    ######################### Outer iteration done ############################
    ### Compute the bagging results using the entire X & Y
    
    cols = X.columns.tolist() 
    train_x = X.values
    train_y = Y.reshape(-1,1)
    
    train_data = {
        'X': train_x,
        'Y': train_y,
        'variable_names': cols,
        'outcome_name': y_label,
        'sample_weights': sample_weights
    }
    
    train_x = train_x[:,1:] # remove intercept
    train_y[train_y == -1] = 0
    
    test_prob_0 = riskslim_prediction(train_x, np.array(cols), model_list[0]).reshape(-1,1)
    test_prob_1 = riskslim_prediction(train_x, np.array(cols), model_list[1]).reshape(-1,1)
    test_prob_2 = riskslim_prediction(train_x, np.array(cols), model_list[2]).reshape(-1,1)
    # test_prob_3 = riskslim_prediction(train_x, np.array(cols), model_list[3]).reshape(-1,1)
    # test_prob_4 = riskslim_prediction(train_x, np.array(cols), model_list[4]).reshape(-1,1)
    
    test_pred_0 = (test_prob_0 > 0.5)
    test_pred_1 = (test_prob_1 > 0.5)
    test_pred_2 = (test_prob_2 > 0.5)
    # test_pred_3 = (test_prob_3 > 0.5)
    # test_pred_4 = (test_prob_4 > 0.5)
    
    test_prob_bag = np.array([np.median([test_prob_0[i], test_prob_1[i], test_prob_2[i]]) for i in range(len(test_prob_0))])
    # cannot add directly, because False + True = False
    test_pred_bag = (test_pred_0.astype(int) + test_pred_1.astype(int) + test_pred_2.astype(int) > 1) # if any model predicts 1
    # np.savetxt('../test_prob_bag.csv', test_prob_bag, delimiter=",")
    # np.savetxt('../test_pred_bag.csv', test_pred_bag, delimiter=",")
    
    # test_prob_bag = np.array([np.median([test_prob_0[i],test_prob_1[i],test_prob_2[i],test_prob_3[i],test_prob_4[i]]) for i in range(len(test_prob_0))])
    # test_pred_bag = (test_pred_0 + test_pred_1 + test_pred_2 + test_pred_3 + test_pred_4 > 2) # if more than two model predicts 1
     
    bag_accuracy = accuracy_score(train_y, test_pred_bag)
    bag_recall = recall_score(train_y, test_pred_bag)
    bag_precision = precision_score(train_y, test_pred_bag)
    bag_roc_auc = roc_auc_score(train_y, test_prob_bag)
    bag_pr_auc = average_precision_score(train_y, test_prob_bag)
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'holdout_test_auc': test_auc, 
            'holdout_train_accuracy': holdout_train_accuracy,
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            'holdout_test_precision': holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            'bag_accuracy': bag_accuracy,
            'bag_recall': bag_recall,
            'bag_precision': bag_precision,
            'bag_roc_auc': bag_roc_auc,
            'bag_pr_auc': bag_pr_auc
            }
    
    
    
    
        

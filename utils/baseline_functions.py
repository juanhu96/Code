#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2022
@Author: Jingyuanhu
"""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
    average_precision_score, brier_score_loss, fbeta_score, accuracy_score, confusion_matrix
# from interpret.glassbox import ExplainableBoostingClassifier
from utils.model_selection import nested_cross_validate, cross_validate


num_threads = 40

##################################### XGBoost ##########################################
def XGB(X,Y,
        learning_rate=None, 
        depth=None, 
        estimators=None, 
        gamma=None, 
        child_weight=None, 
        subsample=None,
        class_weight=None,
        seed=None):

    if class_weight == 'balanced':
        scale_pos_weight = np.bincount(Y)[0]/np.bincount(Y)[1]
    else:
        scale_pos_weight = None
         
    ### model & parameters
    xgboost = xgb.XGBClassifier(scale_pos_weight= scale_pos_weight, 
                                use_label_encoder=False, 
                                eval_metric='auc',
                                random_state=seed,
                                nthread=num_threads)
    c_grid = {"learning_rate": learning_rate,
              "max_depth": depth,
              "n_estimators": estimators,
              "gamma": gamma,
              "min_child_weight": child_weight,
              "subsample": subsample}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    # summary = nested_cross_validate(X=X,Y=Y,estimator=xgboost,c_grid=c_grid,seed=seed,model='XGB')
    _, best_model = cross_validate(X=X,Y=Y,estimator=xgboost,c_grid=c_grid,seed=seed,model='XGB')
    return best_model


################################# Random Forest ###########################################
def RF(X, Y,
       depth=None, 
       estimators=None, 
       impurity=None,
       class_weight=None,
       seed=None):

    ### model & parameters
    rf = RandomForestClassifier(class_weight=class_weight, 
                                bootstrap=True,
                                random_state=seed,
                                n_jobs=num_threads)
    c_grid = {"n_estimators": estimators,
              "max_depth": depth,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    # summary = nested_cross_validate(X=X,Y=Y,estimator=rf,c_grid=c_grid,seed=seed,model='RF')
    _, best_model = cross_validate(X=X,Y=Y,estimator=rf,c_grid=c_grid,seed=seed,model='RF')
    return best_model



##################################### LinearSVM #############################################
def LinearSVM(X, Y, C, class_weight=None, seed=None):
    
    ### model & parameters
    # Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm

    # svm = SVC(kernel='linear',
    #           probability=True,
    #           class_weight=class_weight,
    #           random_state=seed,
    #           max_iter=10000, # used to be 1e6
    #           tol=0.1) # # used to be 0.01

    svm = LinearSVC(class_weight=class_weight,
                    max_iter=1000,
                    random_state=seed,
                    tol=0.01)
    
    # c_grid = {"estimator__C": C}
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
    # summary = nested_cross_validate(X=X,Y=Y,estimator=svm,c_grid=c_grid,seed=seed,index=index,model='LinearSVM')
    _, best_model = cross_validate(X=X,Y=Y,estimator=svm,c_grid=c_grid,seed=seed,model='LinearSVM')
    return best_model



##################################### Lasso #############################################
def Lasso(X, Y, C, class_weight=None, seed=None):
    
    ### model & parameters
    # Lasso not for binary classification
    # lasso = Lasso(max_iter=10000, # used to be 1e6
    #               tol=0.1, # used to be 0.01
    #               random_state=seed,
    #               selection='random')

    lasso = LogisticRegression(class_weight=class_weight,
                               # solver = 'sag',
                               solver='liblinear',
                               max_iter=10000,
                               random_state=seed, 
                               penalty = 'l1')

    c_grid = {"C": C} # LogisticRegression
    # c_grid = {"alpha": C} # Lasso 
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
    # summary = nested_cross_validate(X=X,Y=Y,estimator=lasso,c_grid=c_grid,seed=seed,model='Lasso')
    _, best_model = cross_validate(X=X,Y=Y,estimator=lasso,c_grid=c_grid,seed=seed,model='Lasso')
    return best_model



##################################### Logistic ###########################################
def Logistic(X, Y, C, class_weight=None, seed=None):
    
    ### model & parameters
    lr = LogisticRegression(class_weight=class_weight,
                            # liblinear good for small datasets, sag are faster for large ones
                            # solver = 'sag',
                            solver='liblinear',
                            max_iter=10000,
                            random_state=seed, 
                            penalty = 'l2')
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    # summary = nested_cross_validate(X=X, Y=Y,estimator=lr,c_grid=c_grid,seed=seed,model='Logistic')
    _, best_model = cross_validate(X=X,Y=Y,estimator=lr,c_grid=c_grid,seed=seed,model='Logistic')
    return best_model



################################## Decision Tree ##################################
def DecisionTree(X, Y,
                 depth=None,
                 min_samples=None,
                 impurity=None,
                 class_weight=None,
                 seed=None):
    
    ### model & parameters
    dt = DecisionTreeClassifier(class_weight=class_weight,
                                random_state=seed)
    
    # c_grid = {"estimator__max_depth": depth,
    #           "estimator__min_samples_split": min_samples,
    #           "estimator__min_impurity_decrease": impurity}
    
    c_grid = {"max_depth": depth,
              "min_samples_split": min_samples,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    # summary = nested_cross_validate(X=X, Y=Y, estimator=dt, c_grid=c_grid,seed=seed,model='DT')
    _, best_model = cross_validate(X=X,Y=Y,estimator=dt,c_grid=c_grid,seed=seed,model='DT')
    return best_model



################################## Explainable Boosting Machine ##################################
# def EBM(X,Y,
#         learning_rate=None, 
#         validation_size=None, 
#         max_rounds=None,
#         min_samples=None,
#         max_leaves=None,
#         seed=None):
    
#     ### model & parameters
#     ebm = ExplainableBoostingClassifier(random_state=seed)
#     c_grid = {"learning_rate": learning_rate, 
#               "validation_size": validation_size, 
#               "max_rounds": max_rounds, 
#               "min_samples_leaf": min_samples, 
#               "max_leaves": max_leaves}
    
#     c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
#     summary = nested_cross_validate(X=X, Y=Y, estimator=ebm, c_grid=c_grid,seed=seed)
#     return summary


########################################### NN ###########################################

def NeuralNetwork(X, Y,
                  hidden_layers=(32,), # hidden_layers=(64, 32),
                  activation='relu',
                  solver='adam',
                  alpha=None,
                  batch_size=None,
                  learning_rate_init=None,
                  max_iter=200,
                  seed=None):
    
    ### Model & Parameters
    nn = MLPClassifier(hidden_layer_sizes=hidden_layers,
                       activation=activation,
                       solver=solver,
                       random_state=seed,
                       max_iter=max_iter)
   
    c_grid = {"alpha": alpha,
              "batch_size": batch_size,
              "learning_rate_init": learning_rate_init}
    
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    _, best_model = cross_validate(X=X, Y=Y, estimator=nn, c_grid=c_grid, seed=seed, model='NN')
    return best_model



################################## Deep Neural Network ##################################

'''
def DNN(X, Y):

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from keras_tuner import RandomSearch

    import sys
    import os
    import contextlib
    
    # Define a context manager to suppress output
    # @contextlib.contextmanager
    # def suppress_output():
    #     with open(os.devnull, 'w') as devnull:
    #         old_stdout = sys.stdout
    #         old_stderr = sys.stderr
    #         sys.stdout = devnull
    #         sys.stderr = devnull
    #         try:
    #             yield
    #         finally:
    #             sys.stdout = old_stdout
    #             sys.stderr = old_stderr

    # Split the dataset into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    X = scaler.fit_transform(X)
    
    # Build a function to create a model with hyperparameters  
    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(X.shape[1],)))
        model.add(Dense(units=hp.Choice('units', values=[32, 64]), activation='relu'))
        # model.add(Dense(units=hp.Choice('units', values=[32, 64, 128]), activation='relu', input_dim=X.shape[1]))
        for i in range(hp.Int('num_layers', 1, 2)):
            model.add(Dense(units=hp.Choice(f'units_{i}', values=[32, 64]), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.AUC(name='auc')])
        return model

    # Use Keras Tuner to perform hyperparameter tuning
    tuner = RandomSearch(
        build_model,
        objective='val_auc',
        max_trials=3, # Reduced number of trials
        executions_per_trial=1, # Executions per trial
        directory='my_dir',
        project_name='hyperparameter_tuning')
    
    # Suppress output
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    with HiddenPrints():
        tuner.search(X, Y, epochs=3, validation_split=0.2) # Reduced number of epochs for quicker search
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = build_model(best_hps)
        history = model.fit(X, Y, epochs=10, verbose=0, batch_size=16)

    # loss, accuracy = model.evaluate(X_test, Y_test)
        loss, accuracy = model.evaluate(X, Y)
        
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict probabilities on new data
    prob = model.predict(X)
    pred = (prob >= 0.5)

    def compute_calibration(y, y_prob, y_pred):

        table = []
        num_total_presc = len(y)
        calibration_error = 0
        
        for prob in np.unique(y_prob):
            
            print(prob)
            print("-"*100)
            print(y_prob)
            
            y_temp = y[y_prob == prob]
            y_pred_temp = y_pred[y_prob == prob]
            
            # prescription-level results 
            TN, FP, FN, TP = confusion_matrix(y_temp, y_pred_temp, labels=[0,1]).ravel() 
            observed_risk = np.count_nonzero(y_temp == 1) / len(y_temp)
            num_presc = TN + FP + FN + TP
            calibration_error += abs(prob - observed_risk) * num_presc/num_total_presc

            table.append({'Prob': prob, 'Num_presc': num_presc,
            'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'Observed Risk': observed_risk})

        print(table)

        return calibration_error

    results = {'test_accuracy': accuracy_score(Y, pred),
            'test_recall': recall_score(Y, pred),
            "test_precision": precision_score(Y, pred),
            'test_roc_auc': roc_auc_score(Y, prob),
            'test_pr_auc': average_precision_score(Y, prob),
            "calibration_error": compute_calibration(Y, prob, pred)}
    
    results['calibration_error'] = float(f"{results['calibration_error']:.6f}")

    input_shape = model.input_shape
    num_features = input_shape[1]
    print(f"Number of input features: {num_features}")
    print(results)
'''
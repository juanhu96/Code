#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:45:21 2023

@author: jingyuanhu
multiple attributes, start with m = 2
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score

os.chdir('/mnt/phd/jihu/opioid')

def main():
    ### User settings ###
    # os.chdir('/mnt/phd/jihu/opioid')
    K = 5 # maximum points allowed
    C_0 = 0.001 # regularization term
    Lc = 0
    Uc = 150
    eps = 0.0001
    feature = ['consecutive_days', 'concurrent_MME']
    
    ### Import ###
    # SAMPLE = pd.read_csv('Data/SAMPLE_2018_LONGTERM_stratified.csv', delimiter = ",")
    SAMPLE = pd.read_csv('Data/SAMPLE_2018_LONGTERM_stratified_temp.csv', delimiter = ",")
    SAMPLE = SAMPLE[['consecutive_days', 'concurrent_MME', 'long_term_180']]
    x = SAMPLE[feature].values
    x_min, x_max = SAMPLE[feature].min(), SAMPLE[feature].max()
    z = SAMPLE['long_term_180'].values
    num_obs, num_attr = SAMPLE.shape
    
    c1, c2, l0, l1, l2, new_results = riskSLIM_mip_mattr(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max)
    print('c1 = ' + str(c1) + 'c2 = ' + str(c1) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1) + ', l2 = ' + str(l2))
    print(new_results)
    # c1, c2, l0, l1, l2, old_results = riskSLIM_mip_mattr(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max, ind_constr='old')
    # print('c1 = ' + str(c1) + 'c2 = ' + str(c1) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1) + ', l2 = ' + str(l2))
    # all_results = pd.concat([new_results, old_results], axis=1)
    
    # all_results = all_results.T
    # all_results.to_csv("Result/mip_results_mattr.csv")
    

def riskSLIM_mip_mattr(x, z, K, C_0, Lc, Uc, eps, num_obs, num_attr, x_min, x_max, obj=0, norm=True, ind_constr='new', FuncPieces=0):

    ### Setup ###
    start = time.time()
    m = gp.Model("riskSLIM")
    m.Params.FuncPieces=FuncPieces
    
    
    ### Variables ###
    y1 = m.addVars(num_obs, vtype=GRB.BINARY, name = 'y1')
    y2 = m.addVars(num_obs, vtype=GRB.BINARY, name = 'y2')
    c1 = m.addVar(vtype=GRB.INTEGER, lb = x_min[0], ub = x_max[0], name = 'c1')
    c2 = m.addVar(vtype=GRB.INTEGER, lb = x_min[1], ub = x_max[1], name = 'c2')
    l = m.addVars(num_attr, vtype=GRB.INTEGER, lb = -K, ub = K, name = 'lambda')
    
    # Dummy
    a = m.addVars(num_obs, name = 'a')
    alpha = m.addVars(num_obs, lb = 0, name = 'alpha') # greater than 0
    b = m.addVars(num_obs, lb = 1, name = 'b') # greater than 1
    beta = m.addVars(num_obs, lb = 0, name = 'beta') # greater than 0
    gamma = m.addVar(lb = 0, name = 'gamma')
    
    
    ### Objective ###
    print('Constructing objective function')
    m.setObjective(1/num_obs * quicksum(-z[k] * (l[0] + l[1] * y1[k] + l[2] * y2[k]) + beta[k] for k in range(num_obs)) +  C_0 * gamma, gp.GRB.MINIMIZE)
    
    
    ### Constraints ###
    print('Constructing exp and log auxiliary constraints')
    for k in range(num_obs):
        m.addConstr(a[k] == l[0] + l[1] * y1[k] + l[2] * y2[k])
        m.addGenConstrExp(a[k], alpha[k]) # alpha = exp(a)
        
        m.addConstr(b[k] == 1 + alpha[k])
        # m.addGenConstrLog(b[k], beta[k]) # beta = log(b)
        m.addGenConstrExp(beta[k], b[k]) # b = exp(beta)
    
    print('Constructing l0-norm constraints')
    m.addGenConstrNorm(gamma, [l[1],l[2]], 0, "normconstr") # gamma = l0-norm of l[1], intercept not included
    
    print('Constructing indicator constraints')
    if ind_constr == 'old':
        for k in range(num_obs):
            # first attribute
            m.addConstr((x[:,0][k] - Lc) * y1[k] <= c1 - Lc)
            m.addConstr((-Uc + x[:,0][k] - eps) * y1[k] + c1 <= x[:,0][k] - eps)
            # second attribute
            m.addConstr((x[:,1][k] - Lc) * y2[k] <= c2 - Lc)
            m.addConstr((-Uc + x[:,1][k] - eps) * y2[k] + c2 <= x[:,1][k] - eps)
    elif ind_constr == 'new':
        for k in range(num_obs):   
            m.addConstr((x[:,0][k] - c1) * y1[k] <= 0)
            m.addConstr((1-y1[k]) * (c1 - x[:,0][k] + eps) <= y1[k])
            
            m.addConstr((x[:,1][k] - c2) * y2[k] <= 0)
            m.addConstr((1-y2[k]) * (c2 - x[:,1][k] + eps) <= y2[k])
    else:
        print('Indicator constraint not recognized')
    
    
    ### Solve ###
    m.update()
    m.optimize()
    
    print(str(round(time.time() - start,1)) + ' seconds')
    
    ### Export ###
    c1, c2, l0, l1, l2 = c1.X, c2.X, l[0].X, l[1].X, l[2].X
    y1_soln = np.zeros(num_obs)
    y2_soln = np.zeros(num_obs)
    for j in range(num_obs):
            y1_soln[j] = int(y1[j].X)
            y2_soln[j] = int(y2[j].X)
    
    scores_soln = l0 + l1 * y1_soln + l2*y2_soln
    prob = 1/(1+np.exp(-(scores_soln)))
    pred = (prob > 0.5)
    
    results = {"Accuracy": str(round(accuracy_score(z, pred),4)),
                "Recall": str(round(recall_score(z, pred),4)),
                "Precision": str(round(precision_score(z, pred),4)),
                "ROC AUC": str(round(roc_auc_score(z, prob),4))}
    results = pd.DataFrame.from_dict(results, orient='index', columns=[ind_constr])
    
    ## Clean up ###
    m.dispose()

    return c1, c2, l0, l1, l2, results


if __name__ == "__main__":
    main()


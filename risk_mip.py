#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:45:21 2023

@author: jingyuanhu
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


def main():
  
    ### User settings ###
    os.chdir('/mnt/phd/jihu/opioid')
    K = 5 # maximum points allowed
    C_0 = 0.001 # regularization term
    Lc = 0
    Uc = 150
    eps = 0.001
    feature = 'consecutive_days'
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    
    ### Import ###
    SAMPLE = pd.read_csv('Data/SAMPLE_2018_LONGTERM_stratified_temp.csv', delimiter = ",")
    SAMPLE = SAMPLE[['consecutive_days', 'concurrent_MME', 'long_term_180']]
    x = SAMPLE[feature].values
    x_min, x_max = SAMPLE[feature].min(), SAMPLE[feature].max()
    z = SAMPLE['long_term_180'].values
    num_obs, num_attr = SAMPLE.shape
    
    c, l0, l1, new_results = riskSLIM_mip(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max)
    print('C = ' + str(c) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1))
    print(new_results)
    # c, l0, l1, old_results = riskSLIM_mip(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max, ind_constr='old')
    # print('C = ' + str(c) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1))
    # all_results = pd.concat([new_results, old_results], axis=1)
    '''
    z[z==0]= -1
    c, l0, l1, new_results = riskSLIM_mip(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max, obj=-1)
    print('C = ' + str(c) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1))
    c, l0, l1, old_results = riskSLIM_mip(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max, obj=-1, ind_constr='old')
    print('C = ' + str(c) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1))
    all_results = pd.concat([new_results, old_results], axis=1)
    '''
    
    ###########################################################################
    ################################## 400 ####################################
    ########################################################################### 
    
    # SAMPLE = pd.read_csv('../Data/SAMPLE_2018_LONGTERM_stratified.csv', delimiter = ",")
    # SAMPLE = SAMPLE[['consecutive_days', 'concurrent_MME', 'long_term_180']]
    # x = SAMPLE[feature].values
    # x_min, x_max = SAMPLE[feature].min(), SAMPLE[feature].max()
    # z = SAMPLE['long_term_180'].values
    # num_obs, num_attr = SAMPLE.shape
    
    # c, l0, l1, new_results = riskSLIM_mip(x=x, z=z, K=K, C_0=C_0, Lc=Lc, Uc=Uc, eps=eps, num_obs=num_obs, num_attr=num_attr, x_min=x_min, x_max=x_max)
    # print('C = ' + str(c) + ', l0 = ' + str(l0) + ', l1 = ' + str(l1))
    # print(new_results)
    
    ###########################################################################

    # all_results = all_results.T
    # all_results.to_csv("mip_results_new.csv")


def riskSLIM_mip(x, z, K, C_0, Lc, Uc, eps, num_obs, num_attr, x_min, x_max, obj=0, norm=True, ind_constr='new', FuncPieces=0):

    ### Setup ###
    start = time.time()

    # Hide output    
    # env = gp.Env(empty=True)
    # env.setParam("OutputFlag",0)
    # env.start()
    
    m = gp.Model("riskSLIM")
    m.Params.FuncPieces=FuncPieces
    m.Params.NonConvex=2
    
    ### Variables ###
    # y = m.addVars(num_obs, vtype=GRB.BINARY, name = 'y')
    y = m.addVars(num_obs, lb=0, ub=1, name = 'y')
    l = m.addVars(num_attr, vtype=GRB.INTEGER, lb = -K, ub = K, name = 'lambda')
    c = m.addVar(vtype=GRB.INTEGER, lb = x_min, ub = x_max, name = 'c')
    
    # Dummy
    a = m.addVars(num_obs, name = 'a')
    alpha = m.addVars(num_obs, lb = 0, name = 'alpha') # greater than 0
    b = m.addVars(num_obs, lb = 1, name = 'b') # greater than 1
    beta = m.addVars(num_obs, lb = 0, name = 'beta') # greater than 0
    gamma = m.addVar(lb = 0, name = 'gamma')
    
    
    ### Objective ###
    print('Constructing objective function')
    if obj == 0 and norm == True:    
        m.setObjective(1/num_obs * quicksum(-z[k] * a[k] + beta[k] for k in range(num_obs)) +  C_0 * gamma, gp.GRB.MINIMIZE)
    elif obj == 0 and norm == False:
        m.setObjective(1/num_obs * quicksum(-z[k] * a[k] + beta[k] for k in range(num_obs)), gp.GRB.MINIMIZE)
    elif obj == -1 and norm == True:
        m.setObjective(1/num_obs * quicksum(beta[k] for k in range(num_obs)) +  C_0 * gamma, gp.GRB.MINIMIZE)
    elif obj == -1 and norm == False:
        m.setObjective(1/num_obs * quicksum(beta[k] for k in range(num_obs)), gp.GRB.MINIMIZE)        
    else:
        print('Objective not recognized')
    
    
    ### Constraints ###
    print('Constructing exp and log auxiliary constraints')
    if obj == 0:
        for k in range(num_obs):
            m.addConstr(a[k] == l[0] + l[1]*y[k])
            m.addGenConstrExp(a[k], alpha[k]) # alpha = exp(a)
            
            m.addConstr(b[k] == 1 + alpha[k])
            # m.addGenConstrLog(b[k], beta[k]) # beta = log(b)
            m.addGenConstrExp(beta[k], b[k]) # b = exp(beta)
    else:
        for k in range(num_obs):
            m.addConstr(a[k] == -(l[0] + l[1]*y[k]) * z[k])
            m.addGenConstrExp(a[k], alpha[k]) # alpha = exp(a)
            
            m.addConstr(b[k] == 1 + alpha[k])
            # m.addGenConstrLog(b[k], beta[k]) # beta = log(b)
            m.addGenConstrExp(beta[k], b[k]) # b = exp(beta)
    
    if norm == True:
        print('Constructing l0-norm constraints')
        m.addGenConstrNorm(gamma, [l[1]], 0, "normconstr") # gamma = l0-norm of l[1]. 
    
    print('Constructing indicator constraints')
    if ind_constr == 'old':
        for k in range(num_obs):
            m.addConstr((x[k] - Lc) * y[k] <= c - Lc)
            m.addConstr((-Uc + x[k] - eps) * y[k] + c <= x[k] - eps)
    elif ind_constr == 'new':
        for k in range(num_obs):
            ### Modified to greater than equal to
            m.addConstr((-x[k] + c) * y[k] <= 0)
            m.addConstr((1 - y[k]) * (-c + x[k] + eps) <= y[k])
    else:
        print('Indicator constraint not recognized')
    
    
    ### Solve ###
    m.update()
    m.optimize()
    
    print(str(round(time.time() - start,1)) + ' seconds')
    
    
    ### Export ###
    c, l0, l1 = c.X, l[0].X, l[1].X
    y_soln = np.zeros(num_obs)
    for j in range(num_obs):
            y_soln[j] = int(y[j].X)
    
    scores_soln = l0 + l1 * y_soln
    prob = 1/(1+np.exp(-(scores_soln)))
    pred = (prob > 0.5)
    
    # convert back
    if obj == -1: z[z==-1] = 0
    
    results = {"Accuracy": str(round(accuracy_score(z, pred),4)),
                "Recall": str(round(recall_score(z, pred),4)),
                "Precision": str(round(precision_score(z, pred),4)),
                "ROC AUC": str(round(roc_auc_score(z, prob),4))}
    results = pd.DataFrame.from_dict(results, orient='index', columns=[ind_constr])
    
    ## Clean up ###
    m.dispose()

    return c, l0, l1, results


if __name__ == "__main__":
    main()






'''
Alternative way of expressing objective function
###########################################################################
### Setup ###
start = time.time()
m = gp.Model("riskSLIM")
m.Params.FuncPieces=0


### Variables ###
y = m.addVars(num_obs, vtype=GRB.BINARY, name = 'y')
l = m.addVars(num_attr, vtype=GRB.INTEGER, lb = -K, ub = K, name = 'lambda')
c = m.addVar(vtype=GRB.INTEGER, lb = x_min, ub = x_max, name = 'c')

# Dummy
a = m.addVars(num_obs, name = 'a')
alpha = m.addVars(num_obs, lb = 0, name = 'alpha') # greater than 0
b = m.addVars(num_obs, lb = 1, name = 'b') # greater than 1
beta = m.addVars(num_obs, lb = 0, name = 'beta') # greater than 0
gamma = m.addVar(lb = 0, name = 'gamma')


### Objective ###
print('Constructing objective function')
# m.setObjective(1/num_obs * quicksum(beta[k] for k in range(num_obs)) +  C_0 * gamma, gp.GRB.MINIMIZE)
m.setObjective(1/num_obs * quicksum(beta[k] for k in range(num_obs)), gp.GRB.MINIMIZE)

### Constraints ###
print('Constructing exp and log auxiliary constraints')
for k in range(num_obs):
    m.addConstr(a[k] == -z[k] * (l[0] + l[1]*y[k]))
    m.addGenConstrExp(a[k], alpha[k]) # alpha = exp(a)
    
    m.addConstr(b[k] == 1 + alpha[k])
    m.addGenConstrLog(b[k], beta[k]) # beta = log(b)

print('Constructing l0-norm constraints')
# m.addGenConstrNorm(gamma, [l[1]], 0, "normconstr") # gamma = l0-norm of l[1]. 

print('Constructing indicator constraints')
for k in range(num_obs):
    m.addConstr((-x[k] - Lc) * y[k] <= -c - Lc)
    m.addConstr((-Uc - x[k] + eps) * y[k] + c <= x[k] - eps)


### Solve ###
m.update()
m.optimize()

print(str(round(time.time() - start,1)) + ' seconds')

### Export ###
print(c.X)
print(l[0].X)
print(l[1].X)
###########################################################################
'''

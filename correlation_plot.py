#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 6th, 2023

Correlation plot to compare prediction between different models

@author: Jingyuan Hu
"""

import os
import pandas as pd
import numpy as np 
from sklearn.metrics import recall_score,precision_score,confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")

base_font=0.7
os.chdir('/mnt/phd/jihu/opioid/Result')


def main():
    
    SAMPLE = pd.read_csv('../Data/FULL_2018_LONGTERM.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)
    
    ###########################################################################
    ########################## FEATURE CORRELATION ############################
    ###########################################################################
    
    # Filter out the discrete/categorical varaibles
    '''
    df = SAMPLE[['age', 'quantity', 'days_supply', 'quantity_per_day', 'daily_dose',
                 'total_dose', 'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                 'num_pharmacies', 'concurrent_benzo', 'consecutive_days', 'num_alert',
                 'num_presc', 'dose_diff', 'MME_diff', 'days_diff', 'opioid_days', 'long_term_180']]
    
    MC_heatmap_of_feature(df, font_scale=0.7, export_file=True, filename='MC_heatmap_of_feature.png')
    '''
    
    ###########################################################################
    ############################ MODEL CORRELATION ############################
    ###########################################################################
    
    '''
    y_true = SAMPLE[['long_term_180']].to_numpy().astype('int')  
    Y_list = []
    Y_list.append(('True', y_true))
    model_list = ['DT', 'L1', 'L2', 'SVM', 'RF', 'XGB', 'riskSLIM']
    for model in model_list:
        pred = np.loadtxt(model + '_y.csv', delimiter=",")
        Y_list.append((model, pred))
        
    MC_heatmap_of_prediction(Y_list, decimal=2, font_scale=0.7, figsize=(8,6), export_file=True, filename='MC_heatmap_of_prediction.png', dpi=300)
    '''

    ### Positive ones only
    y_true = SAMPLE[['long_term_180']].to_numpy().astype('int')  
    indices_of_ones = np.where(y_true == 1)[0]
    y_true = y_true[indices_of_ones]

    Y_list = []
    Y_list.append(('True', y_true))
    
    model_list = ['DT', 'L1', 'L2', 'SVM', 'RF', 'XGB', 'riskSLIM']
    for model in model_list:
        pred = np.loadtxt(model + '_y.csv', delimiter=",")
        pred = pred[indices_of_ones]
        Y_list.append((model, pred))
        
    MC_heatmap_of_prediction(Y_list, decimal=2, font_scale=0.7, figsize=(8,6), export_file=True, filename='MC_heatmap_of_prediction_true.png', dpi=300)



def MC_heatmap_of_feature(df, decimal=2, font_scale=None, figsize=(12,10), export_file=False, filename=None, dpi=300):
    """
    Parameters
    ----------
    decimal: decimal of numbers shown in the heatmap
    font_scale: scaling factor for font size
    figsize: figure size
    export_file: a Boolean parameter indicating whether to export the figure to a file or simply display to screen
    filename: output file name
    """
    
    matrix = df.corr().round(decimal)
    plt.figure(figsize=figsize)

    if font_scale is None:
        pass
    else:
        sns.set(font_scale=font_scale)
    
    sns.heatmap(matrix, annot=True, linewidths = .3)  
    
    if export_file is False:
        pass
    else:
        if filename is None:
            print('Please input filename for export.')
            pass
        else:
            plt.savefig(filename, dpi=dpi)
    sns.set(font_scale=base_font)


def MC_heatmap_of_prediction(Y_list, decimal=2, font_scale=None, figsize=(12,10), export_file=False, filename=None, dpi=300):
    """
    Pairwise comparison from a heatmap of model prediction agreement level.
    
    Parameters
    ----------
    Y_list: a list of tuples that consists of model names and model predictions. The first tuple contains the true labels, while the subsequent tuples contain model predictions.
    decimal: decimal of numbers shown in the heatmap
    font_scale: scaling factor for font size
    figsize: figure size
    export_file: a Boolean parameter indicating whether to export the figure to a file or simply display to screen
    filename: output file name
    """ 
    
    matrix=pd.DataFrame()
    col=[]
    
    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    
    n = matrix.shape[0]
    m = matrix.shape[1]
    res = np.zeros((m,m))
    for i1 in range(matrix.shape[1]):
        for i2 in range(matrix.shape[1]):
            acc = sum(matrix.iloc[:,i1]==matrix.iloc[:,i2])/n
            res[i1,i2]=acc

    res = pd.DataFrame(data=res, columns=matrix.columns.values, index=matrix.columns.values)
    res=res.round(decimal)

    plt.figure(figsize=figsize)
    
    if font_scale is None:
        pass
    else:
        sns.set(font_scale=font_scale)
        
    sns.heatmap(res, annot=True)
    
    if export_file is False:
        pass
    else:
        if filename is None:
            print('Please input filename for export.')
            pass
        else:
            plt.savefig(filename, dpi=dpi)
    sns.set(font_scale=base_font)
    
    
    
    
if __name__ == "__main__":
    main()  
    
    
    
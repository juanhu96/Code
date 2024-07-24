



# ========================================================================================


def create_intervals(year, scenario='flexible', feature_list=LTOUR_feature_list, datadir='/mnt/phd/jihu/opioid/Data/'):
    
    '''
    Create intervals stumps for the dataset
    For this we also need to edit stumps.create_stumps as well
    
    Parameters
    ----------
    year
    scenario: basic feature (flexible) / full
    '''

    FULL = pd.read_csv(f'{datadir}FULL_{str(year)}_LONGTERM_INPUT.csv', delimiter = ",", 
                        dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                              'num_prescribers': int, 'num_pharmacies': int,
                              'concurrent_benzo': int, 'consecutive_days': int}).fillna(0)
    FULL = FULL[FULL.columns.drop(list(FULL.filter(regex='alert')))]
    FULL = FULL.drop(columns = ['drug_payment'])
    
    
    if scenario == 'flexible':
        x_all = FULL[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                      'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
        
        cutoffs_i = []
        for column_name in ['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_i.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_i.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_i.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_i.append([n for n in range(0, 200) if n % 10 == 0])
        
        cutoffs_s = []
        for column_name in ['num_prescribers', 'num_pharmacies', 'concurrent_benzo']:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs_s.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo':
                cutoffs_s.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME':
                cutoffs_s.append([n for n in range(0, 90) if n % 10 == 0])
            else:
                cutoffs_s.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):
            FULL_fold = FULL_splited[i]
            x = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',
                          'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]
            
            x_i = FULL_fold[['concurrent_MME', 'concurrent_methadone_MME', 'consecutive_days']]
            x_s = FULL_fold[['num_prescribers', 'num_pharmacies', 'concurrent_benzo']]
            
            x_intervals = stumps.create_intervals(x_i.values, x_i.columns, cutoffs_i)
            x_stumps = stumps.create_stumps(x_s.values, x_s.columns, cutoffs_s)
            
            new_data = pd.concat([x_intervals.reset_index(drop=True), x_stumps.reset_index(drop=True)], axis = 1)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)  
    

    elif scenario == 'full':
        x_all = FULL[feature_list]
        
        cutoffs = []
        for column_name in x_all.columns:
            if column_name == 'num_prescribers' or column_name == 'num_pharmacies':
                cutoffs.append([n for n in range(0, 10)])
            elif column_name == 'concurrent_benzo' or column_name == 'concurrent_benzo_same' or \
                column_name == 'concurrent_benzo_diff' or column_name == 'num_presc':
                cutoffs.append([0, 1, 2, 3, 4, 5, 10])
            elif column_name == 'consecutive_days' or column_name == 'concurrent_methadone_MME' or \
                column_name == 'days_diff':
                cutoffs.append([n for n in range(0, 90) if n % 10 == 0])
            elif column_name == 'dose_diff' or column_name == 'concurrent_MME_diff':
                cutoffs.append([n for n in range(0, 100) if n % 10 == 0])
            elif column_name == 'age':
                cutoffs.append([n for n in range(20, 80) if n % 10 == 0])
            else:
                cutoffs.append([n for n in range(0, 200) if n % 10 == 0])
                
        ## Divide into 20 folds
        N = 20
        FULL_splited = np.array_split(FULL, N)
        for i in range(N):

            FULL_fold = FULL_splited[i]
            x = FULL_fold[feature_list]
            x_stumps = stumps.create_stumps(x.values, x.columns, cutoffs)
            x_rest = FULL_fold[FULL_fold.columns.drop(feature_list)]

            new_data = pd.concat([x_stumps.reset_index(drop=True), x_rest.reset_index(drop=True)], axis = 1)
            print(new_data.shape)
            new_data.to_csv('Data/FULL_' + str(year) + scenario + '_INTERVALS' + str(i) + '.csv', header=True, index=False)         
    
    else:
        print('Scenario cannot be identified')





# ========================================================================================



def test_table_full(year, output_table=False, roc=False, calibration=False, datadir='/mnt/phd/jihu/opioid/Data/', resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    '''
    Compute the performance metric given a scoring table for a given year
    '''
    
    ### Import 
    
    SAMPLE = pd.read_csv(f'{datadir}FULL_{str(year)}_LONGTERM_UPTOFIRST.csv', delimiter = ",", 
                         dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                'num_prescribers': int, 'num_pharmacies': int,
                                'concurrent_benzo': int, 'consecutive_days': int,
                                'alert1': int, 'alert2': int, 'alert3': int, 'alert4': int, 'alert5': int, 'alert6': int})
    SAMPLE = SAMPLE.fillna(0)

    x = SAMPLE
    y = SAMPLE[['long_term_180']].to_numpy().astype('int')
    

    ### Performance
    x['Prob'] = x.apply(compute_score_full_one, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_one = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                   "Recall": str(round(recall_score(y, y_pred), 4)),
                   "Precision": str(round(precision_score(y, y_pred), 4)),
                   "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                   "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_one = pd.DataFrame.from_dict(results_one, orient='index', columns=['1'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'one')

    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_one')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_two, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_two = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                   "Recall": str(round(recall_score(y, y_pred), 4)),
                   "Precision": str(round(precision_score(y, y_pred), 4)),
                   "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                   "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_two = pd.DataFrame.from_dict(results_two, orient='index', columns=['2'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'two')

    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_two')
        
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_three, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_three = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                     "Recall": str(round(recall_score(y, y_pred), 4)),
                     "Precision": str(round(precision_score(y, y_pred), 4)),
                     "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                     "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_three = pd.DataFrame.from_dict(results_three, orient='index', columns=['3'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'three')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_three')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_four, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_four = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                    "Recall": str(round(recall_score(y, y_pred), 4)),
                    "Precision": str(round(precision_score(y, y_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                    "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_four = pd.DataFrame.from_dict(results_four, orient='index', columns=['4'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'four')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_four')
    
    # ========================================================================================

    x['Prob'] = x.apply(compute_score_full_five, axis=1)
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()
    
    results_five = {"Accuracy": str(round(accuracy_score(y, y_pred), 4)),
                    "Recall": str(round(recall_score(y, y_pred), 4)),
                    "Precision": str(round(precision_score(y, y_pred), 4)),
                    "ROC AUC": str(round(roc_auc_score(y, y_prob), 4)),
                    "PR AUC": str(round(average_precision_score(y, y_prob), 4))}
    results_five = pd.DataFrame.from_dict(results_five, orient='index', columns=['5'])
    if output_table == True:
        store_predicted_table(year, table, SAMPLE, x, 'five')
    
    if calibration == True:
        compute_calibration(x, y, y_prob, y_pred, resultdir, 'LTOUR_five')
    
    # ========================================================================================

    results = pd.concat([results_one, results_two], axis=1)
    results = pd.concat([results, results_three], axis=1)
    results = pd.concat([results, results_four], axis=1)
    results = pd.concat([results, results_five], axis=1)
    results = results.T
    results.to_csv(f'{resultdir}results_test_{str(year)}_LTOUR.csv')
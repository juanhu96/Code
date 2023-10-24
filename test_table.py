import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
average_precision_score, accuracy_score


def main():
    
    # test_dataset(dataset='Opioid', model_list=['original', 'polyapprox', 'IGA'], feature = 'full', N_list = [10000, 20000, 50000])
    # test_dataset(dataset='Framingham', model_list=['IGA2e1', 'IGA1e1', 'IGA5e2'], maxiter_dict={'IGA2e1':9, 'IGA1e1':9, 'IGA5e2':9}, filename='')
    plot_results(['Accuracy', 'ROC AUC'])
    

def test_dataset(dataset, model_list, feature=None, N_list=None, year=2018, maxiter_dict=None, filename='', workdir='/mnt/phd/jihu/opioid_conic/'):
    
    '''
    feature, N only applicable to Opioid SAMPLE
    '''

    if dataset == 'Opioid':

        results = []

        for N in N_list:

            SAMPLE = pd.read_csv(f'{workdir}Data/SAMPLE_{str(year)}_LONGTERM_stratified_{str(N)}.csv', delimiter = ",", 
                                dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float,
                                        'num_prescribers': int, 'num_pharmacies': int,
                                        'concurrent_benzo': int, 'consecutive_days': int})
            
            SAMPLE = SAMPLE.fillna(0)
            
            if feature == 'core':
                x = SAMPLE[['concurrent_MME', 'consecutive_days']]

            elif feature == 'basic':
                x = SAMPLE[['concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers',\
                            'num_pharmacies', 'concurrent_benzo', 'consecutive_days']]

            elif feature == 'full':
                x = SAMPLE

            else: print('Warning: Undefined')

            y = SAMPLE[['long_term_180']].to_numpy().astype('int')

            if maxiter_dict is not None:
                for model in model_list:
                    maxiter = maxiter_dict[model]
                    for iters in range(1, maxiter+1):
                        results.append(test_table(dataset=dataset, x=x, y=y, model=model, feature=feature, N=N, iters=iters))
            else:
                for model in model_list:
                    results.append(test_table(dataset=dataset, x=x, y=y, model=model, feature=feature, N=N))

        results = pd.DataFrame(results)
        results.to_csv(f'{workdir}Results/TestResult_{feature}_{str(year)}_{filename}.csv', index=False)


    if dataset == 'Framingham':

        SAMPLE = pd.read_csv(f'{workdir}Data/framingham.csv', delimiter = ",")
            
        SAMPLE = SAMPLE.fillna(0)
        x = SAMPLE
        y = SAMPLE[['TenYearCHD']].to_numpy().astype('int')
        
        results = []
        if maxiter_dict is not None:
            for model in model_list:
                maxiter = maxiter_dict[model]
                for iters in range(1, maxiter+1):
                    results.append(test_table(dataset=dataset, x=x, y=y, model=model, iters=iters))
        else:
            for model in model_list:
                results.append(test_table(dataset=dataset, x=x, y=y, model=model))

        results = pd.DataFrame(results)
        results.to_csv(f'{workdir}Results/TestResult_Framingham_{filename}.csv', index=False)


    print("Finished.")



def test_table(dataset, x, y, model, feature=None, N=None, iters=None, workdir='/mnt/phd/jihu/opioid_conic/'):

    if dataset == 'Opioid':

        if iters is not None:
            scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/N{str(N)}_{feature}_{model}{iters}.csv', delimiter = ",")
        else:
            scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/N{str(N)}_{feature}_{model}.csv', delimiter = ",")

    if dataset == 'Framingham':
        N = 1000 # dummy

        if iters is not None:
            # scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/Framingham_{model}{iters}.csv', delimiter = ",")
            scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/N{str(N)}_Framingham_{model}{iters}.csv', delimiter = ",")
        else:
            # scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/Framingham_{model}.csv', delimiter = ",")
            scoring_table = pd.read_csv(f'{workdir}Results/{dataset}/N{str(N)}_Framingham_{model}.csv', delimiter = ",")

    x['Prob'] = x.apply(compute_score, axis=1, args=(scoring_table,))
    x['Pred'] = (x['Prob'] > 0.5)
    y_prob, y_pred = x['Prob'].to_numpy(), x['Pred'].to_numpy()

    if iters is not None:
        results = {"Model": model, 'Iteration': iters, "Accuracy": str(round(accuracy_score(y, y_pred), 3)), "ROC AUC": str(round(roc_auc_score(y, y_prob), 3))}
    else:
        results = {"Model": model, "Accuracy": str(round(accuracy_score(y, y_pred), 3)), "ROC AUC": str(round(roc_auc_score(y, y_prob), 3))}

    return results



def compute_score(row, scoring_table):

    score = 0
    intercept = scoring_table['intercept'][0]

    for index, table_row in scoring_table.iterrows():

        selected_feature = table_row['selected_feature']
        cutoff = table_row['cutoff']
        point = table_row['point']

        if row[selected_feature] >= cutoff: score += point

    
    return 1 / (1+np.exp(-(score + intercept)))
        


def plot_results(outcome_list, filename='', workdir='/mnt/phd/jihu/opioid_conic/'):
    
    df = pd.read_csv(f'{workdir}Results/TestResult_Framingham_{filename}.csv', delimiter = ",")
    
    for outcome in outcome_list:

        plt.figure(figsize=(10, 6))  # Set the figure size
        sns.scatterplot(data=df, x='Iteration', y=outcome, hue='Model', palette='Set1', legend=False)
        sns.lineplot(data=df, x='Iteration', y=outcome, hue='Model', palette='Set1', sort=False)

        plt.xlabel('Iteration')
        plt.ylabel(outcome)
        plt.title(f'{outcome} vs. Iteration by Model')
        plt.legend(title='Model')
        plt.savefig(f'{workdir}Results/Framingham_scatterplot_{outcome}.png')

    return


if __name__ == "__main__":
    main()
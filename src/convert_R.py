'''
Convert python results to R format for visualization
'''
import pickle
import numpy as np
import pandas as pd


def convert_benchmark(naive:bool = False,
                      metrics = ['roc', 'calibration', 'proportions'],
                      models = ["L1", "DecisionTree", "XGB", "L2", "LinearSVM", "RandomForest", "NN", "Logistic"], 
                    #   models = ["Logistic"], 
                      resultdir = '/mnt/phd/jihu/opioid/Code/output/baseline/files/',
                      exportdir = '/export/storage_cures/CURES/Results_R/'):
    
    for metric in metrics:
        for model in models:

            suffix = f"_naive" if naive else ""
            suffix = f"_bracket{suffix}" if model == 'Logistic' else suffix
            input_path = f'{resultdir}{model}_{metric}_test_info{suffix}.pkl'
            output_csv = f'{exportdir}{model}_{metric}{suffix}.csv'

            with open(input_path, 'rb') as f: data = pickle.load(f)

            arrays = {k: v for k, v in data.items() if isinstance(v, (np.ndarray, list))}
            scalars = {k: v for k, v in data.items() if isinstance(v, (float, np.floating, int))}

            if arrays:
                df = pd.DataFrame(arrays)
                for k, v in scalars.items(): df[k] = v  # scalar broadcast to all rows

                df['Model'] = model
                df['Presc'] = 'Naive' if naive else 'All'
                df.to_csv(output_csv, index=False)
                print(f"Exported to {output_csv}")



def convert_LTOUR(naive:bool = False,
                  metrics = ['roc', 'calibration', 'proportions'],
                  LTOUR_models = ['LTOUR_6'],
                  year = '2019',
                  resultdir = '/mnt/phd/jihu/opioid/Code/output/baseline/files/',
                  exportdir = '/export/storage_cures/CURES/Results_R/'):

    for metric in metrics:
        for model in LTOUR_models:

            input_path = f'{resultdir}/riskSLIM_{metric}_test_info_{model}_{f"first_{year}" if naive else year}.pkl'
            output_csv = f'{exportdir}LTOUR_{metric}{"_naive" if naive else ""}.csv'

            with open(input_path, 'rb') as f: data = pickle.load(f)

            arrays = {k: v for k, v in data.items() if isinstance(v, (np.ndarray, list))}
            scalars = {k: v for k, v in data.items() if isinstance(v, (float, np.floating, int))}

            if arrays:
                df = pd.DataFrame(arrays)
                for k, v in scalars.items(): df[k] = v  # scalar broadcast to all rows
                
                df['Model'] = 'LTOUR'
                df['Presc'] = 'Naive' if naive else 'All'
                df.to_csv(output_csv, index=False)
                print(f"Exported to {output_csv}")


def convert_LTOUR_county(naive:bool = False,
                         metrics = ['roc', 'calibration'],
                         county_list = ['Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras',
                                        'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno',
                                        'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern',
                                        'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera',
                                        'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Modoc',
                                        'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange',
                                        'Placer', 'Plumas', 'Riverside', 'Sacramento',
                                        'San Benito', 'San Bernardino', 'San Diego', 'San Francisco',
                                        'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara',
                                        'Santa Clara', 'Santa Cruz', 'Shasta', 'Sierra',
                                        'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus',
                                        'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne',
                                        'Ventura', 'Yolo', 'Yuba'],
                         year = '2019',
                         resultdir = '/mnt/phd/jihu/opioid/Code/output/baseline/files/',
                         exportdir = '/export/storage_cures/CURES/Results_R/'):

    for metric in metrics:
        combined_dfs = []
        for county in county_list:

            input_path = f'{resultdir}/riskSLIM_{metric}_test_info_LTOUR_6_county{county}_{f"first_{year}" if naive else year}.pkl'
            output_csv = f'{exportdir}LTOUR_{county}_{metric}{"_naive" if naive else ""}.csv'

            with open(input_path, 'rb') as f: data = pickle.load(f)

            arrays = {k: v for k, v in data.items() if isinstance(v, (np.ndarray, list))}
            scalars = {k: v for k, v in data.items() if isinstance(v, (float, np.floating, int))}

            if arrays:
                df = pd.DataFrame(arrays)
                for k, v in scalars.items(): df[k] = v  # scalar broadcast to all rows
                
                df['Model'] = 'LTOUR'
                df['Presc'] = 'Naive' if naive else 'All'
                df['County'] = county
                df.to_csv(output_csv, index=False)
                print(f"Exported to {output_csv}")
                combined_dfs.append(df.iloc[[0]])

        if combined_dfs:
            full_df = pd.concat(combined_dfs, ignore_index=True)
            combined_output_csv = f'{exportdir}LTOUR_AllCounties_{metric}{"_naive" if naive else ""}.csv'
            full_df.to_csv(combined_output_csv, index=False)
            print(f"Exported combined file to {combined_output_csv}")

    return

# convert_benchmark()
# convert_benchmark(naive=True)
# convert_LTOUR()
# convert_LTOUR(naive=True)
# convert_LTOUR_county()
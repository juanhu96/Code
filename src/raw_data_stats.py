# Raw data description

import pandas as pd
import numpy as np
basedir = "/export/storage_cures/CURES/"
datadir = '/export/storage_cures/CURES/Processed/'

year = 2018

if False:
    FULL_CURRENT = pd.read_csv(f"{basedir}RX_{year}.csv")
    print(FULL_CURRENT.columns.to_list())

if True:    
    df = pd.read_csv(f'{datadir}/Stumps/FULL_{year}_Explore_STUMPS_median_1.csv', delimiter=",")
    print(df.columns.to_list())
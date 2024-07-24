import sys
import numpy as np
import pandas as pd

from lpm_utils.initial import initial
from lpm_utils.optimize_lpm import optimize_lpm
from lpm_utils.export_table import export_table
from lpm_utils.test_table import test_table


year = sys.argv[1] # 2018
mode = sys.argv[2] # train/test
max_point = int(sys.argv[3]) # 5 
max_features = int(sys.argv[4]) # 10
weight = sys.argv[5] # unbalanced
c0 = float(sys.argv[6]) # 1e-15
suffix = sys.argv[7] # suffix

setting_tag = f'_{year}_{mode}_{max_point}p_{max_features}f_{weight}_{suffix}'


def main(year, mode, setting_tag, max_point, max_features, weight, c0):

    if mode == 'train': train_table(year, setting_tag, max_point, max_features, weight, c0)
    elif mode == 'test': test_table(year, setting_tag, max_point, max_features, weight, c0)
    else: raise KeyError("Mode undefined\n")

    return




def train_table(year, setting_tag, max_point, max_features, weight, c0):

    print(f"Start training table using {year} prescriptions under LPM;\n Setting tag {setting_tag}...\n")

    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(year)

    # ==========================================================================
    # OPTIMIZE

    intercept, theta, gamma = optimize_lpm(z, num_feature, num_obs, x_order, num_order, v_order, max_point, max_features, c0)

    # ==========================================================================
    # EXPORT
    export_table(setting_tag, intercept, theta, feature_list, max_point, num_feature, num_order, x_order)

    # ==========================================================================
    # TEST
    # test_table(problem) # uses gamma

    return




if __name__ == "__main__":
    main(year, mode, setting_tag, max_point, max_features, weight, c0)
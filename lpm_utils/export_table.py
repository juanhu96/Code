import pandas as pd
import numpy as np


def export_table(setting_tag, intercept, theta, feature_list, max_point, num_feature, num_order, x_order, export_file=True, expdirpath='/export/storage_cures/CURES/Results_LPM/'):

    cutoff = [0]
    point = [intercept]
    selected_feature = ['intercept']

    for j in range(num_feature):
        found = 0
        for p in range(1, max_point + 1):
            for t in range(num_order[j]):
                if theta[j, t, p] > 0.5 and found == 0:
                    cutoff.append(x_order[j][t])
                    point.append(p)
                    selected_feature.append(feature_list[j])
                    found = 1


    data = {
        'selected_feature': selected_feature,
        'cutoff': cutoff,
        'point': point
    }

    df = pd.DataFrame(data)

    if export_file:
        df.to_csv(f'{expdirpath}/table{setting_tag}.csv', index=False)


    return df

import pandas as pd
import numpy as np
import sys
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import seaborn as sns


datadir = "/export/storage_cures/CURES/Processed/"
exportdir = "/export/storage_cures/CURES/Plots/"
year = 2018

if False:
    
    FULL = pd.read_csv(f"{datadir}FULL_OPIOID_{year}_INPUT.csv")

    long_term = FULL[FULL['long_term_180'] == 1]
    non_long_term = FULL[FULL['long_term_180'] == 0]
    # features = ['concurrent_MME', 'daily_dose', 'days_supply']

    features = ['concurrent_MME', 'daily_dose']
    colors = {'concurrent_MME': '#1f77b4', 'daily_dose': '#ff7f0e'}


    # --------- Density Plot ---------
    def plot_density_overlay(group_df, group_label):
        plt.figure(figsize=(8, 5))

        legend_elements = []
        for feature in features:
            sns.kdeplot(group_df[feature].clip(upper=600), fill=True, alpha=0.4, label=feature, color=colors[feature])
            mean_val = group_df[feature].mean()
            plt.axvline(mean_val, color=colors[feature], linestyle='--', linewidth=2)
            
            legend_elements.append(
                Line2D([0], [0], color=colors[feature], lw=2, linestyle='--',
                    label=f"{feature} (Mean: {mean_val:.1f})")
            )

        plt.xlim(0, 600)
        plt.title(f"Feature Distribution - {group_label}")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
        plt.tight_layout()

        output_pdf = f'{exportdir}Density_{group_label.replace(" ", "_")}_{year}.pdf'
        plt.savefig(output_pdf, format='pdf', dpi=500)
        plt.close()
        print(f"Density plot saved to {output_pdf}")

    # Plot for each group
    plot_density_overlay(long_term, 'Long-Term')
    plot_density_overlay(non_long_term, 'Non-Long-Term')
    plot_density_overlay(FULL, 'All')


    # --------- Summary Stats ---------
    def summary_stats(df, feature):
        return {
            '10%': df[feature].quantile(0.1),
            '20%': df[feature].quantile(0.2),
            '30%': df[feature].quantile(0.3),
            '40%': df[feature].quantile(0.4),
            'Mean': df[feature].mean(),
            'Median (50%)': df[feature].median(),
            '60%': df[feature].quantile(0.6),
            '70%': df[feature].quantile(0.7),
            '80%': df[feature].quantile(0.8),
            '90%': df[feature].quantile(0.9)
        }

    summary_data = []
    for group_name, group_data in [('All', FULL), ('Long-Term', long_term), ('Non-Long-Term', non_long_term)]:
        for feature in features:
            stats = summary_stats(group_data, feature)
            summary_data.append({
                'Group': group_name,
                'Feature': feature,
                **stats
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = f"{exportdir}Summary_Stats_{year}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary stats saved to {summary_csv}")


if False:

    basedir = "/export/storage_cures/CURES/"
    export_dir = basedir + "Processed/"
    FULL_CURRENT = pd.read_csv(f"{basedir}RX_{year}.csv")
    unique_classes = FULL_CURRENT['class'].unique()
    class_counts = FULL_CURRENT['class'].value_counts()
    print(f"Unique classes in {year}: {unique_classes}")
    print("Class counts:")
    print(class_counts)


if False:

    # debug calibration issues: naive has more prescriptions full
    # import FULL_2019_INPUT and FIRST
    FULL_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")

    FULL_2019_FROMFULL = FULL_2019.groupby('patient_id', as_index=False).first()
    FULL_2019_FIRST = pd.read_csv(f"{datadir}FULL_OPIOID_2019_FIRST_INPUT.csv")

    diff_df = pd.concat([FULL_2019_FROMFULL, FULL_2019_FIRST]).drop_duplicates(keep=False)
    print(f"Number of differing rows between FULL_2019_FROMFULL and FULL_2019_FIRST: {diff_df.shape[0]}")

    print(FULL_2019.shape)

    def drop_na_rows(FULL):

        FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)

        feature_list = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo', 
                        'patient_gender', 'days_supply', 'daily_dose',
                        'num_prior_prescriptions', 'diff_MME', 'diff_days',
                        'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                        'patient_zip_yr_avg_days', 'patient_zip_yr_avg_MME']

        percentile_list = ['patient_zip_yr_num_prescriptions', 'patient_zip_yr_num_patients', 
                            'patient_zip_yr_num_pharmacies', 'patient_zip_yr_avg_MME', 
                            'patient_zip_yr_avg_days', 'patient_zip_yr_avg_quantity', 
                            'patient_zip_yr_num_prescriptions_per_pop', 'patient_zip_yr_num_patients_per_pop',
                            'prescriber_yr_num_prescriptions', 'prescriber_yr_num_patients', 
                            'prescriber_yr_num_pharmacies', 'prescriber_yr_avg_MME', 
                            'prescriber_yr_avg_days', 'prescriber_yr_avg_quantity',
                            'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 
                            'pharmacy_yr_num_prescribers', 'pharmacy_yr_avg_MME', 
                            'pharmacy_yr_avg_days', 'pharmacy_yr_avg_quantity',
                            'zip_pop_density', 'median_household_income', 
                            'family_poverty_pct', 'unemployment_pct']
        percentile_features = [col for col in FULL.columns if any(col.startswith(f"{prefix}_above") for prefix in percentile_list)]
        feature_list_extended = feature_list + percentile_features
        FULL = FULL.dropna(subset=feature_list_extended) # drop NA rows to match the stumps

        return FULL

    print("Dropping NA rows to match the stumps...")
    FULL_2019 = drop_na_rows(FULL_2019)
    FULL_2019_FROMFULL = FULL_2019.groupby('patient_id', as_index=False).first()

    FULL_2019_FIRST = drop_na_rows(FULL_2019_FIRST)

    # check if there's row in FULL_2019 but not in FULL_2019_FIRST
    diff_df = pd.concat([FULL_2019_FROMFULL, FULL_2019_FIRST]).drop_duplicates(keep=False)
    print(f"Number of differing rows between FULL_2019_FROMFULL and FULL_2019_FIRST: {diff_df.shape[0]}")


if False:

    # patient_ids = [22548077, 23368467, 38636296, 48650568, 50890526, 58122909, 72147379, 73656418]
    patient_ids = [22548077, 23368467, 38636296]

    FULL_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")

    FULL_2019_FROMFULL = FULL_2019.groupby('patient_id', as_index=False).first()
    FULL_2019_FIRST = pd.read_csv(f"{datadir}FULL_OPIOID_2019_FIRST_INPUT.csv")

    # get the rows for the patient_ids
    df_full = FULL_2019[FULL_2019['patient_id'].isin(patient_ids)]
    df_fromfull = FULL_2019_FROMFULL[FULL_2019_FROMFULL['patient_id'].isin(patient_ids)]
    df_first = FULL_2019_FIRST[FULL_2019_FIRST['patient_id'].isin(patient_ids)]
    print("Rows from FULL_2019:")
    print(df_full)
    print("Rows from FULL_2019_FROMFULL:")
    print(df_fromfull)
    print("Rows from FULL_2019_FIRST:")
    print(df_first)

if True:

    FULL_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
    print(FULL_2019.shape)

    def drop_na_rows(FULL):

        FULL.rename(columns={'quantity_diff': 'diff_quantity', 'dose_diff': 'diff_MME', 'days_diff': 'diff_days'}, inplace=True)

        feature_list = ['concurrent_MME', 'num_prescribers_past180', 'num_pharmacies_past180', 'concurrent_benzo', 
                        'patient_gender', 'days_supply', 'daily_dose',
                        'num_prior_prescriptions', 'diff_MME', 'diff_days',
                        'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment',
                        'patient_zip_yr_avg_days', 'patient_zip_yr_avg_MME']

        percentile_list = ['patient_zip_yr_num_prescriptions', 'patient_zip_yr_num_patients', 
                            'patient_zip_yr_num_pharmacies', 'patient_zip_yr_avg_MME', 
                            'patient_zip_yr_avg_days', 'patient_zip_yr_avg_quantity', 
                            'patient_zip_yr_num_prescriptions_per_pop', 'patient_zip_yr_num_patients_per_pop',
                            'prescriber_yr_num_prescriptions', 'prescriber_yr_num_patients', 
                            'prescriber_yr_num_pharmacies', 'prescriber_yr_avg_MME', 
                            'prescriber_yr_avg_days', 'prescriber_yr_avg_quantity',
                            'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 
                            'pharmacy_yr_num_prescribers', 'pharmacy_yr_avg_MME', 
                            'pharmacy_yr_avg_days', 'pharmacy_yr_avg_quantity',
                            'zip_pop_density', 'median_household_income', 
                            'family_poverty_pct', 'unemployment_pct']
        percentile_features = [col for col in FULL.columns if any(col.startswith(f"{prefix}_above") for prefix in percentile_list)]
        feature_list_extended = feature_list + percentile_features
        FULL = FULL.dropna(subset=feature_list_extended) # drop NA rows to match the stumps

        return FULL
    
    print("Dropping NA rows to match the stumps...")
    FULL_2019 = drop_na_rows(FULL_2019)
    print(FULL_2019.shape)
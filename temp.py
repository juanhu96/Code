import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


datadir = "/export/storage_cures/CURES/Processed/"
exportdir = "/export/storage_cures/CURES/Plots/"
year = 2018

if True:
    
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
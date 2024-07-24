import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

test = False
oversample = False


def initial(year):
    
    df, x, z, feature_list = import_dataset(year)
    num_feature = len(feature_list)
    num_obs, num_attr = df.shape

    x_min = x.min().tolist()
    x_max = x.max().tolist()

    x_order, num_order, v_order = [], [], []
    for feature in feature_list:
        x_order_item = sorted(x[feature].unique())
        v_order_item = x[feature].map({value: idx for idx, value in enumerate(x_order_item)}).tolist()
        x_order.append(x_order_item)
        num_order.append(len(x_order_item))
        v_order.append(v_order_item)

    return z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order



def import_dataset(year, datadir='/export/storage_cures/CURES/Processed/'):

    FULL = pd.read_csv(f'{datadir}FULL_OPIOID_{str(year)}_INPUT.csv', delimiter = ",",\
    dtype={'concurrent_MME': float, 'concurrent_methadone_MME': float, 'num_prescribers_past180': int,\
    'num_pharmacies_past180': int, 'concurrent_benzo': int, 'consecutive_days': int}).fillna(0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.001, random_state=42)
    for train_index, test_index in sss.split(FULL, FULL['long_term_180']):
        stratified_sample = FULL.iloc[test_index]

    # print(stratified_sample.shape, stratified_sample.columns.values.tolist())
    # df.dropna(axis=0, how='any', inplace=True)
    # df = df[df != "NA"].dropna()
    # for c in df.select_dtypes(include='object').columns:
    #     df[c] = pd.to_numeric(df[c], errors='coerce')
    # df = df.astype(float)

    '''
    ['patient_id', 'patient_gender', 'quantity', 'days_supply', 'daily_dose', 'total_dose', 
     'max_dose', 'age', 'concurrent_MME', 'concurrent_methadone_MME', 'num_prescribers_past180', 
     'num_pharmacies_past180', 'consecutive_days', 'concurrent_benzo', 'concurrent_benzo_same', 
     'concurrent_benzo_diff', 'days_to_long_term', 'long_term_180', 'num_prior_prescriptions', 
     'num_prior_prescriptions_past180', 'num_prior_prescriptions_past90', 'num_prior_prescriptions_past30', 
     'switch_drug', 'switch_payment', 'ever_switch_drug', 'ever_switch_payment', 'dose_diff', 
     'concurrent_MME_diff', 'quantity_diff', 'days_diff', 'avgMME_past180', 'avgDays_past180', 
     'avgMME_past90', 'avgDays_past90', 'avgMME_past30', 'avgDays_past30', 'HMFO', 'Codeine_MME', 
     'Hydrocodone_MME', 'Oxycodone_MME', 'Morphine_MME', 'Hydromorphone_MME', 'Methadone_MME', 'Fentanyl_MME', 
     'Oxymorphone_MME', 'gap', 'num_prior_prescriptions_benzo_past180', 'num_prior_prescriptions_benzo_past90', 
     'num_prior_prescriptions_benzo_past30', 'patient_HPIQuartile', 'prescriber_HPIQuartile', 'pharmacy_HPIQuartile', 
     'zip_pop', 'zip_pop_density', 'median_household_income', 'family_poverty_pct', 'unemployment_pct', 
     'patient_zip_num_prescriptions', 'patient_zip_num_patients', 'patient_zip_avg_days', 'patient_zip_avg_quantity', 
     'patient_zip_avg_MME', 'prescriber_monthly_prescriptions', 'prescriber_monthly_patients', 'prescriber_monthly_avg_days', 
     'prescriber_monthly_avg_quantity', 'prescriber_monthly_avg_MME', 'prescriber_yr_num_prescriptions', 
     'prescriber_yr_num_patients', 'prescriber_yr_num_pharmacies', 'prescriber_yr_avg_MME', 'prescriber_yr_avg_days', 
     'prescriber_yr_avg_quantity', 'prescriber_yr_num_prescriptions_quartile', 'prescriber_yr_num_patients_quartile', 
     'prescriber_yr_num_pharmacies_quartile', 'prescriber_yr_avg_MME_quartile', 'prescriber_yr_avg_days_quartile', 
     'prescriber_yr_avg_quantity_quartile', 'pharmacy_yr_num_prescriptions', 'pharmacy_yr_num_patients', 'pharmacy_yr_num_prescribers', 
     'pharmacy_yr_avg_MME', 'pharmacy_yr_avg_days', 'pharmacy_yr_avg_quantity', 'pharmacy_yr_num_prescriptions_quartile', 
     'pharmacy_yr_num_patients_quartile', 'pharmacy_yr_num_prescribers_quartile', 'pharmacy_yr_avg_MME_quartile', 
     'pharmacy_yr_avg_days_quartile', 'pharmacy_yr_avg_quantity_quartile', 'Codeine', 'Hydrocodone', 'Oxycodone', 'Morphine', 
     'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone', 'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit', 'MilitaryIns', 
     'WorkersComp', 'Other', 'IndianNation', 'zip_pop_density_quartile', 'median_household_income_quartile', 'family_poverty_pct_quartile', 
     'unemployment_pct_quartile', 'patient_zip_num_prescriptions_per_pop', 'patient_zip_num_patients_per_pop', 
     'patient_zip_num_prescriptions_per_pop_quartile', 'patient_zip_num_patients_per_pop_quartile']
    '''

    # feature_list = ["concurrent_MME", "concurrent_methadone_MME", 
    #                 "consecutive_days", "num_prescribers_past180", "num_pharmacies_past180", "concurrent_benzo",
    #                 "age", "num_prior_prescriptions", "concurrent_MME_diff", "quantity_diff", "days_diff",
    #                 "Codeine", "Hydrocodone", "Oxycodone", "Morphine", "HMFO",
    #                 'Medicaid', 'CommercialIns', 'Medicare', 'CashCredit',
    #                 "ever_switch_drug", "ever_switch_payment", "long_term_180"]
    
    feature_list = ["concurrent_MME", "consecutive_days", "num_prescribers_past180", "num_pharmacies_past180", "age",
                    # "concurrent_MME_diff", "quantity_diff", "days_diff",
                    # "median_household_income", "family_poverty_pct", "unemployment_pct",
                    "patient_zip_num_prescriptions", "patient_zip_num_patients", 
                    "patient_zip_avg_days", "patient_zip_avg_quantity", "patient_zip_avg_MME",
                    # "prescriber_yr_num_prescriptions", "prescriber_yr_num_patients", 
                    # "prescriber_yr_num_pharmacies", "prescriber_yr_avg_MME", 
                    # "prescriber_yr_avg_days", "prescriber_yr_avg_quantity",
                    # "pharmacy_yr_num_prescriptions", "pharmacy_yr_num_patients", 
                    # "pharmacy_yr_num_prescribers", "pharmacy_yr_avg_MME", 
                    # "pharmacy_yr_avg_days", "pharmacy_yr_avg_quantity",
                    "long_term_180"]

    df = stratified_sample[feature_list]

    z = df['long_term_180'].values
    counts = df['long_term_180'].value_counts()
    print(f"Number of 0s: {counts.get(0, 0)}; Number of 1s: {counts.get(1, 0)}\n")
    x = df.drop(columns=['long_term_180'])
    feature_list.remove('long_term_180')

    return df, x, z, feature_list
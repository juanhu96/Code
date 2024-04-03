#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 2023

Merging the data from CA overdose dashboard
THIS IS NOT FOR RISK SCORING PROJECT
@author: Jingyuan Hu
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd
import geopy

def main():
    # merge_data(level = 'Zip Code')
    compute_zip()




def merge_data(level, datadir = '/mnt/phd/jihu/opioid/Data/CA overdose'):

    if level == 'County':

        # ====================================================================

        # Note 1: 2022 is not available for all (cause, outcome)
        # Note 2: before 2013 also note available for all (cause, outcome)
        # Note 3: I think we should focus on pre-COVID
        start_year, end_year = 2006, 2021 
        quarters = ['', '_2', '_3', '_4']
        # cause_list = ['Any Opioid', 'All Drug', 'Heroin', 'Cocaine']
        cause_list = ['Any Opioid', 'All Drug']
        outcome_list = ['Death', 'EDVisit', 'Hosp']

        dfs_to_concat = []

        for cause in cause_list:
            for year in range(start_year, end_year+1):
                for i in range(4):
                    
                    quarter = quarters[i]
                    
                    ## NOTE: death prelimiary data for 2022, hosp missing for 2022

                    # if year == 2022 and cause == 'Any Opioid':
                    #     death_year_quarter = pd.read_csv(f'{datadir}/{cause}/CA_{cause}-Related Overdose_Death_by County_Prelim. {year}{quarter}.csv', delimiter = ",").iloc[2:-3]
                    #     ed_year_quarter = pd.read_csv(f'{datadir}/{cause}/CA_{cause}-Related Overdose_EDVisit_by County_{year}{quarter}.csv', delimiter = ",").iloc[2:-3]
                    #     hosp_year_quarter = pd.read_csv(f'{datadir}/{cause}/CA_{cause}-Related Overdose_Hosp_by County_{year}{quarter}.csv', delimiter = ",").iloc[2:-3]

                    death_year_quarter = pd.read_csv(f'{datadir}/{level}/{cause}/CA_{cause}-Related Overdose_Death_by {level}_{year}{quarter}.csv', delimiter = ",").iloc[2:-3]
                    ed_year_quarter = pd.read_csv(f'{datadir}/{level}/{cause}/CA_{cause}-Related Overdose_EDVisit_by {level}_{year}{quarter}.csv', delimiter = ",").iloc[2:-3]
                    hosp_year_quarter = pd.read_csv(f'{datadir}/{level}/{cause}/CA_{cause}-Related Overdose_Hosp_by {level}_{year}{quarter}.csv', delimiter = ",").iloc[2:-3]

                    # merge
                    death_year_quarter.columns = ['County', 'Death Rates', 'Death LCL', 'Death UCL']
                    ed_year_quarter.columns = ['County', 'EDVisit Rates', 'EDVisit LCL', 'EDVisit UCL']
                    hosp_year_quarter.columns = ['County', 'Hosp Rates', 'Hosp LCL', 'Hosp UCL']
                    df_year_quarter = pd.merge(pd.merge(death_year_quarter, ed_year_quarter, on='County'), hosp_year_quarter, on='County')

                    # add another column with year and quarter
                    df_year_quarter['Year'] = str(year)
                    df_year_quarter['Quarter'] = i+1
                    df_year_quarter['Cause'] = cause

                    # merge
                    dfs_to_concat.append(df_year_quarter)


        df_aggregated = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
        df_aggregated.to_csv(f'{datadir}/{level}/CA_aggregated{start_year}_{end_year}.csv', index=False)


    elif level == 'Zip Code':

        # ====================================================================

        start_year, end_year = 2010, 2021 # 2022 no hosp
        cause_list = ['Any Opioid']
        outcome_list = ['Death', 'EDVisit', 'Hosp']

        counties = pd.read_csv(f'{datadir}/CA_county_list.csv', delimiter = ",").county_name
        
        dfs_to_concat = []

        for cause in cause_list:
            for year in range(start_year, end_year+1):
                for county in counties:

                    death_year_county = pd.read_csv(f'{datadir}/{level}/{cause}/{county}_{cause}-Related Overdose_Death_by {level}_{year}.csv', delimiter = ",").iloc[2:-5]
                    ed_year_county = pd.read_csv(f'{datadir}/{level}/{cause}/{county}_{cause}-Related Overdose_ED Visit_by {level}_{year}.csv', delimiter = ",").iloc[2:-5]
                    hosp_year_county = pd.read_csv(f'{datadir}/{level}/{cause}/{county}_{cause}-Related Overdose_Hosp_by {level}_{year}.csv', delimiter = ",").iloc[2:-5]

                    # merge
                    death_year_county.columns = ['Zip', 'Death Rates', 'Death LCL', 'Death UCL']
                    ed_year_county.columns = ['Zip', 'EDVisit Rates', 'EDVisit LCL', 'EDVisit UCL']
                    hosp_year_county.columns = ['Zip', 'Hosp Rates', 'Hosp LCL', 'Hosp UCL']
                    df_year_county = pd.merge(pd.merge(death_year_county, ed_year_county, on='Zip'), hosp_year_county, on='Zip')

                    # add another column with year and quarter
                    df_year_county['Year'] = str(year)
                    df_year_county['County'] = county
                    df_year_county['Cause'] = cause

                    # merge
                    dfs_to_concat.append(df_year_county)

        df_aggregated = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
        df_aggregated.to_csv(f'{datadir}/{level}/CA_aggregated{start_year}_{end_year}.csv', index=False)


    else:
        print('Warning: Undefined!\n')



def compute_zip(datadir = '/mnt/phd/jihu/opioid/Data/CA overdose'):

    atm = pd.read_csv(f'{datadir}/bitcoin_atm_location_california.csv', delimiter = ",")
    
    geolocator = geopy.Nominatim(user_agent='1234')
    atm['zip'] = atm.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude').iloc[1:10]
    atm.to_csv(f'{datadir}/bitcoin_atm_location_california_computed.csv', index=False)



def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']




if __name__ == "__main__":
    main()
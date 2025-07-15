"""the goal is to merge all measurements into a single csv 
because reviewer2 said so
"""

import math

import pandas as pd

def rm_dashes(date_str):
    return "".join(date_str.split("-"))

def merge_biomass(row):
    if math.isnan(row['biomass per area grown (g/m2)']):
        row['biomass per area grown (g/m2)']=row["biomass per full plot (g/m2)"]
    return row

def merge_lai_des(row):
    if math.isnan(row['LAI per sqm of area grown']):
        row['LAI per sqm of area grown']=row["LAI per plot"]
    return row


if __name__ == '__main__':
    biomass_fp = "md_Biomass.csv"
    lai_des_fp = "md_Destructive_LAI.csv"
    sunscan_fp = "md_SunScan.csv"

    biomass_pd_ =pd.read_csv(biomass_fp)
    lai_des_pd_ =pd.read_csv(lai_des_fp)
    sunscan_pd_ =pd.read_csv(sunscan_fp)

    biomass_pd = biomass_pd_[~(biomass_pd_["Dont_use_psqm"]==1)]
    biomass_pd = biomass_pd.apply(merge_biomass, axis=1)
    biomass_pd = biomass_pd[['plot', 'date', 'crop', 'biomass per area grown (g/m2)']]
    biomass_pd.rename(columns={'biomass per area grown (g/m2)': 'Biomass (g/m2)'}, inplace=True)
    biomass_pd['date'] = biomass_pd['date'].astype(str)

    lai_des_pd = lai_des_pd_[~(lai_des_pd_["Dont_use_psqm"]==1)]
    lai_des_pd = lai_des_pd.apply(merge_lai_des, axis=1)
    lai_des_pd = lai_des_pd[['plot', 'date', 'crop', 'LAI per sqm of area grown']]
    lai_des_pd.rename(columns={'LAI per sqm of area grown': 'LAI_Destructive'}, inplace=True)
    lai_des_pd['date'] = lai_des_pd['date'].astype(str)

    sunscan_pd = sunscan_pd_[['plot_id', 'date', 'crop', 'LAI_mean']]
    sunscan_pd.rename(columns={'plot_id': 'plot'}, inplace=True)
    sunscan_pd.rename(columns={'LAI_mean': 'LAI_SunScan'}, inplace=True)
    sunscan_pd['date'] = sunscan_pd['date'].astype(str)

    merged_pd = pd.merge(lai_des_pd, sunscan_pd, on=["plot","date","crop"], how='outer')
    merged_pd = pd.merge(merged_pd, biomass_pd, on=["plot","date","crop"], how='outer')

    merged_pd['date'] = pd.to_datetime(merged_pd['date'])
    merged_pd = merged_pd.sort_values(by='date')
    merged_pd.to_csv("LAI_biomass_combined.csv", index=False)


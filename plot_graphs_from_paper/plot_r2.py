#!/usr/bin/env python3

import os
from datetime import datetime
import math

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from utils.sowing_dates import sowing_dates
from utils.graph_dicts import cult2color_dict
from utils.id_mapper import IdMapper

def correspond_des_ss(row, des_df):
    corr_df = des_df[des_df["plot"]==row["plot_id"]]
    corr_df = corr_df[abs(corr_df["date"] - row["days_sown"]) <= 2]
    if len(corr_df) > 0:
        row["des_lai"]=corr_df["LAI per plot"].iloc[0]
        row["delta_days"]=abs(corr_df["date"] - row["days_sown"]).item()
    row["cult_color"]=cult2color_dict[row["crop"]]
    return row

def LAI_minmax(row):
    n = int(row["num_scans"])
    min_lai = 100.0
    max_lai = 0.0
    for i in range(n):
        head = "LAI_meas_" + str(i+1)
        lai = float(row[head])
        min_lai = min(min_lai, lai)
        max_lai = max(max_lai, lai)

    row["LAI_min"] = min_lai
    row["LAI_max"] = max_lai

    return row



def use_area_grown(old_row):
    if not np.isnan(old_row["LAI per sqm of area grown"]):
        old_row["LAI per plot"] = old_row["LAI per sqm of area grown"]
    return old_row


def convert_cult(row):
    row["crop"] = IdMapper.get_cultivar(row["crop"])
    return row

# convert dates to days after sowing
def mew(old_row):
    cult = old_row["crop"]
    sow_date_str = sowing_dates[cult]
    sow_date = datetime.strptime(sow_date_str, '%Y%m%d')

    old_date_str = old_row["date"]
    old_date = datetime.strptime(str(old_date_str), '%Y%m%d')
    new_date = old_date - sow_date

    old_row["date"]= new_date.days
    new_row = old_row

    return new_row


def plot_r2_sunscan_des(sunscan_csv_fp, destructive_csv_fp, output_dir):
    sunscan_df = pd.read_csv(sunscan_csv_fp)

    des_df = pd.read_csv(destructive_csv_fp)
    des_df = des_df[des_df["Dont_use_psqm"]!=1]
    des_df=des_df.apply(mew, axis=1)  # get days sown
    des_df=des_df.apply(convert_cult, axis=1)  # make cult names the same for both df
    des_df=des_df.apply(use_area_grown, axis=1)  # use "LAI per sqm of area grown" if present
    sunscan_df=sunscan_df.apply(convert_cult, axis=1)  # make cult names the same for both df
    df_assigned = sunscan_df.apply(correspond_des_ss, axis=1, args=(des_df,))
    df_assigned = df_assigned[~pd.isna(df_assigned["des_lai"])]
     
    df_assigned = df_assigned.apply(LAI_minmax, axis=1)

    def plt_el(row):
        yerr=np.stack([row["LAI_mean"]-row["LAI_min"], row["LAI_max"]-row["LAI_mean"]])
        yerr = np.expand_dims(yerr, axis=1)
        # if row["LAI_mean"] > 5:
        #     import pdb; pdb.set_trace()
        el = plt.errorbar(
            row["des_lai"],
            row["LAI_mean"],
            #xerr=row["des_lai"]*0.04,
            yerr=yerr,
            color=row["cult_color"],
            fmt="x",
            capsize=5,
            )
        return row
    df_assigned = df_assigned.apply(plt_el, axis=1)
    

    max_max = max(df_assigned["des_lai"].max(), df_assigned["LAI_mean"].max() ) + 0.5

    slope, intercept = np.polyfit(df_assigned["des_lai"], df_assigned["LAI_mean"], 1)
    x=np.arange(0,max_max, 0.5) 
    best_fit_line = slope * x + intercept
    plt.plot(x, best_fit_line, color='black')
    plt.plot(x, x, color='black', ls="--")
    plt.xlim(0, max_max)
    plt.ylim(0, max_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Destructive LAI")
    plt.ylabel("SunScan LAI")

    custom_legend = []
    for cult in cult2color_dict:
        if cult == "Potato":
            continue
        custom_legend.append(
                plt.Line2D([0], [0], color=cult2color_dict[cult], marker='x', linestyle='', label=cult),
                )
    r_squared = r2_score(df_assigned["des_lai"],df_assigned["LAI_mean"])
    custom_legend.append(
            mpatches.Patch(color='white', label=f"$R^2$={r_squared:.2f}"),
            )
    rmse = math.sqrt(mean_squared_error(df_assigned["des_lai"],df_assigned["LAI_mean"]))
    custom_legend.append(
            mpatches.Patch(color='white', label=f"RMSE={rmse:.2f}"),
            )

    plt.legend(handles=custom_legend, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r2.pdf"), dpi=300)



@click.command()
@click.option('--parent_dir', "-p", required=True, help="path to parent directory of the MuST-C dataset")
@click.option('--output_dir', "-o", default=".", help="path to output dir. must exist")
def main(parent_dir, output_dir):

    sunscan_fp = os.path.join(parent_dir, "metadata", "md_SunScan.csv")
    des_fp = os.path.join(parent_dir, "metadata", "md_Destructive_LAI.csv")

    plot_r2_sunscan_des(
            sunscan_fp, 
            des_fp, 
            output_dir
            )


if __name__ == '__main__':
    main()

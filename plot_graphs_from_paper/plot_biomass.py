"""plot biomass based on csv
"""

import os
from datetime import datetime, time

import matplotlib.pyplot as plt
import pandas as pd
import click

from utils.sowing_dates import sowing_dates
from utils.graph_dicts import cult2color_dict, marker_dict 


def get_date_sown(df_entry):
    date_yyyymmdd = str(df_entry["date"])
    cultivar = df_entry["crop"]
    sow_date_str = sowing_dates[cultivar]
    sow_date = datetime.strptime(sow_date_str, '%Y%m%d')
    date_date = datetime.strptime(date_yyyymmdd, '%Y%m%d')

    delta = date_date - sow_date
    dates_sown = delta.days
    return dates_sown


@click.command()
@click.option('--parent_dir', "-p", required=True, help="path to parent directory of the MuST-C dataset")
@click.option('--output_dir', "-o", default=".", help="path to output dir. must exist")
def plot_scatter(parent_dir, output_dir):
    biomass_csv = os.path.join(parent_dir, "metadata", "md_Biomass.csv")
    full_df = pd.read_csv(biomass_csv)

    ft=18
    plt.figure(figsize=(8, 6))
    full_df = full_df[full_df["Dont_use_psqm"] !=1]

    full_df["days_sown"] = full_df.apply(get_date_sown, axis=1)
    for cult in full_df["crop"].unique():
        df = full_df[full_df["crop"]==cult]
        plt.scatter(
                    df["days_sown"],
                    df["biomass per full plot (g/m2)"],
                    marker=marker_dict[cult],
                    label=cult,
                    color=cult2color_dict[cult],
                    s=100,
                )

    plt.legend(fontsize=ft)
    plt.ylim(bottom=0)
    plt.xlabel('Days sown', fontsize=ft)
    plt.ylabel('Biomass (gm$^{-2}$)', fontsize=ft)
    plt.tick_params(axis='both', labelsize=ft)
    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, f"biomass.pdf"), dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_scatter()

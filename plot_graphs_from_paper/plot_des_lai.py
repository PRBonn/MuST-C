"""plot graphs from final destructive lai csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click

from utils.graph_dicts import cult2color_dict
from plot_biomass import get_date_sown


@click.command()
@click.option('--parent_dir', "-p", required=True, help="path to parent directory of the MuST-C dataset")
@click.option('--output_dir', "-o", default=".", help="path to output dir. must exist")
def plot_per_plant(parent_dir, output_dir):
    """plot variation of plants in the same plot
    """

    lai_csv = os.path.join(parent_dir, "metadata", "md_Destructive_LAI.csv")
    df = pd.read_csv(lai_csv)

    ft=16
    plt.figure(figsize=(16, 12))

    full_df = df[df["Dont_use_psqm"] ==1]
    full_df["days_sown"] = full_df.apply(get_date_sown, axis=1)

    for date in full_df["date"].unique():
        date_df = full_df[full_df["date"]==date]
        for plot in date_df["plot"].unique():
            df = date_df[date_df["plot"]==plot]
            x = np.arange(1,len(df)+1)
            # if crop == "Sugar Beet" or crop == "Maize":

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ind = np.arange(len(df))
            width = 0.35
            bar1 = ax1.bar(
                    ind - width/2, 
                    df["leaf_count(total)"], 
                    width, 
                    color="#473BF0",
                    label="Number of leaves"
                    )
           
            # plot the boxplots for each date
            bar2 = ax2.bar(
                    ind + width/2, 
                    df["total_leaf_area"], 
                    width, 
                    label="Leaf area (m$^2$)",
                    color="#FFA400",
                    yerr=df["total_leaf_area"]*0.04,
                    capsize=4
                    )
           
            ax1.legend(handles=[bar1], loc="upper left", fontsize=12)
            ax2.legend(handles=[bar2], loc="upper right", fontsize=12)

            ax1.set_ylabel('Number of leaves', fontsize=ft)
            ax2.set_ylabel('Leaf area (m$^{2}$)', fontsize=ft)
            ax1.set_ylim([0, df["leaf_count(total)"].max() + 4])

            ax1.set_xlabel("Plant id", fontsize=ft)
            ax1.set_xticks(ind)
            ax1.set_xticklabels(tuple(df["tag"].tolist()))
            ax2.get_xaxis().set_visible(False)

            plt.tight_layout() 
            plt.savefig(os.path.join(output_dir, f"per_plant_{date}_{df['crop'].iloc[0]}.pdf"), dpi=300)
            plt.close()


if __name__ == '__main__':
    plot_per_plant()


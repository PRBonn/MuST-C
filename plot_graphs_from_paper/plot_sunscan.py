import os

import click
import matplotlib.pyplot as plt 
import pandas as pd 

@click.command()
@click.option('--parent_dir', "-p", required=True, help="path to parent directory of the MuST-C dataset")
@click.option('--output_dir', "-o", default=".", help="path to output dir. must exist")
def plot_sunscan_graph(parent_dir, output_dir):
    sunscan_fp = os.path.join(parent_dir, "metadata", "md_SunScan.csv")
    full_df = pd.read_csv(sunscan_fp)

    ft = 20
    lw=3
    fmt_list = ["o-", "v--", "s-", "*--", "D-", ">--", "x-", "P--"]
    color_list = [
                "#EE4266", 
                "#FFA400",
                "#3BCEAC",
                "#473BF0",
                "#0EAD69",
                "#0D0630",
                "#B66D0D",
                "#9B5DE5"
                ]

    for cult in full_df["crop"].unique():
        plt.figure(figsize=(10, 6))
        df = full_df[full_df["crop"]==cult]
        for fmt_i, plot_id in enumerate(df["plot_id"].unique()):
            df_plot = df[df["plot_id"]==plot_id]

            plt.errorbar(
                    df_plot["days_sown"],
                    df_plot["LAI_mean"],
                    yerr=df_plot["LAI_stddev"],
                    label=plot_id,
                    fmt=fmt_list[fmt_i],
                    capsize=5,
                    color=color_list[fmt_i],
                    lw=lw
                    )
        plt.legend(fontsize=ft, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Days sown', fontsize=ft)
        plt.ylabel('LAI', fontsize=ft)
        plt.ylim(bottom=0) 
        plt.tick_params(axis='both', labelsize=ft)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sunscan_{cult}.pdf"))
        plt.close()

if __name__ == "__main__":
    plot_sunscan_graph()

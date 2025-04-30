"""get the dates when des or sunscan were made
for latex table 
"""


import pandas
from process_sunscan import IdMapper
cult_names = {
            "\maize" : ["maize", "sugar corn", "sugarcorn"],
            "\soybean": ["soybean", "soyab", "soya bean", "soy bean"],
            "\wheat": ["summerwheat", "sw", "sommerwheat", "summer wheat"],
            "\sugarbeet": ["sugarbeet", "sugarbeets", "sugar beet"],
            "\intercrop": ["mixture", "mix-faba", "mix-summerwheat", "mix-fabasummerwheat", "intercrop"],
            "\potato": ["potatoes", "potato"]
            }

def run_sunscan():
    csv_fp = "./sunscan_corrected.csv"  # sunscan 
    df = pandas.read_csv(csv_fp)
    cults = df["date"].unique()
    for cult in cults:
        df_cult =  df[df["date"] ==  cult]
        dates = df_cult["cultivar"].unique()
        converted = ""
        for date in dates:
            # converted = converted + date
            converted = converted + IdMapper.get_cultivar(date, cult_names)
            
        print(f"date: {cult}, cultivars: {converted}")



def run_des():
    csv_fp = "./total_17dec_scaled_consolidated.csv"  # destructive
    df = pandas.read_csv(csv_fp)
    cults = df["date"].unique()
    for cult in cults:
        df_cult =  df[df["date"] ==  cult]
        dates = df_cult["cultivar"].unique()
        converted = ""
        for date in dates:
            converted = converted + IdMapper.get_cultivar(date, cult_names)
            
        print(f"date: {cult}, cultivars: {converted}")

if __name__ == '__main__':
    run_des()
    # run_sunscan()

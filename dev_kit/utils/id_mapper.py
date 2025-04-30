import pandas as pd
import importlib.resources

# map all ids to the ex id (if not already)
class IdMapper:
    df = {}
    cult_names = {
            "Maize" : ["maize", "sugar corn", "sugarcorn"],
            "Soy Bean": ["soybean", "soyab", "soya bean", "soybean (minngold)", "soyabean", "soy bean"],
            "Wheat": ["summerwheat", "sw", "sommerwheat", "wheat", "summer wheat"],
            "Sugar Beet": ["sugarbeet", "sugarbeets", "sugar beet"],
            "Intercrop": ["mixture", "mix-faba", "mix-summerwheat", "mix-fabasummerwheat", "mixed", "faba", "mixture (faba-wheat)", "intercrop"],
            "Potato": ["potatoes", "potato"]
            }

    def set_df(csv_file =None):
        if csv_file is None:
            csv_file = importlib.resources.open_text('utils', 'map_ids.csv')
        IdMapper.df = pd.read_csv(csv_file, dtype={"ex_id": "Int32", "shp_id": "Int32"})

    def ex2cult(exid):
        df = IdMapper.df[IdMapper.df["ex_id"]==exid]
        cult = df["cultivar"].iloc[0]
        return cult

    def get_exid(plot):
        # check if its already
        try:
            plot = int(plot)
        except:
            pass

        df = IdMapper.df

        # todo check that its only one entry
        if IdMapper.df["ex_id"].isin([plot]).any():
            ex_id=plot
        elif IdMapper.df["shp_id"].isin([plot]).any():
            ex_id = df[df["shp_id"] == plot].iloc[0]["ex_id"]
        elif IdMapper.df["field_id"].isin([plot]).any():
            ex_id = df[df["field_id"] == plot].iloc[0]["ex_id"]
        else:
            raise ValueError(f"get_exid:Invalid plot id {plot}")
        return ex_id

    # map all ids to the final id
    def ex2shp_id(plot):
        df = IdMapper.df
        shp_id = df[df["ex_id"] == plot].iloc[0]["shp_id"]
        return shp_id

    def shp_id2cult(shp_id):
        cult = IdMapper.df[IdMapper.df["shp_id"]==shp_id]["cultivar"].item()
        return cult

    # keeping cultivar names consistent
    def get_cultivar(cult, cult_names_dict=None):
        if cult_names_dict is None:
            cult_names_dict = IdMapper.cult_names
        cult = cult.lower().strip()
        for keyer in cult_names_dict:
            for option in cult_names_dict[keyer]:
                # if (cult in option) or (option in cult):
                if cult == option:
                    return keyer
        return "Invalid crop"


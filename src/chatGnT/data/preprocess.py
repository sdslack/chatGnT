import pandas as pd

def clean_ingred(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pattern = r"^(\d*\s?\d/\d+|\d+(?:\.\d+)?)\s+(\S+)\s+(.+)$"
    df[["amt", "unit", "ingred"]] = df["ingredient_name"].str.extract(pattern)
    return df


# def clean_drinks(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df.columns = df.columns.str.strip()
#     df["strDrink"] = df["strDrink"].str.strip()
#     return df



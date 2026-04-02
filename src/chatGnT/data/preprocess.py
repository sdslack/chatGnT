import pandas as pd

def clean_ingred(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pattern = r"^(\d*\s?\d/\d+|\d+(?:\.\d+)?)\s+(\S+)\s+(.+)$"
    extracted = df["ingredient_name"].str.extract(pattern)

    valid = extracted.notnull().all(axis=1)
    valid_ids = df.groupby("id").apply(
        lambda x: valid[x.index].all()
    )
    valid_ids = valid_ids[valid_ids].index  # only ids with all valid rows
    df = df[df["id"].isin(valid_ids)]

    df["amt"] = extracted[0]
    df["unit"] = extracted[1]
    df["ingred"] = extracted[2]
    
    return df

# src/utils.py

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
# import re
# import matplotlib.pyplot as plt


# ----------- Data Loading -----------

def load_kagglehub_dataset(dataset_id: str, file_path: str = "", pandas_kwargs=None) -> pd.DataFrame:
    """Load dataset using KaggleHub."""
    pandas_kwargs = pandas_kwargs or {}
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        pandas_kwargs=pandas_kwargs
    )
    return df


def list_kagglehub_files(dataset_id: str):
    """List files inside the Kaggle dataset directory so you can pick your FILE_PATH."""
    path = kagglehub.dataset_download(dataset_id)
    return sorted([p.name for p in path.iterdir()])


# ----------- EDA Helpers -----------

# def schema_overview(df: pd.DataFrame) -> pd.DataFrame:
#     """Basic schema summary."""
#     return pd.DataFrame({
#         "dtype": df.dtypes.astype(str),
#         "missing_frac": df.isna().mean(),
#         "n_unique": df.nunique(dropna=True)
#     })


# def find_ingredient_columns(df: pd.DataFrame):
#     ing = [c for c in df.columns if re.match(r"strIngredient\\d+$", c)]
#     meas = [c for c in df.columns if re.match(r"strMeasure\\d+$", c)]

#     ing = sorted(ing, key=lambda x: int(re.findall(r"\\d+", x)[0]))
#     meas = sorted(meas, key=lambda x: int(re.findall(r"\\d+", x)[0]))
#     return ing, meas


# def extract_ingredients_long(df: pd.DataFrame, id_col=None):
#     """Convert wide ingredient columns into a long table."""
#     ing_cols, meas_cols = find_ingredient_columns(df)

#     temp = df.copy()
#     if id_col is None:
#         temp = temp.reset_index().rename(columns={"index": "drink_id"})
#         id_col = "drink_id"

#     rows = []
#     for i, ing_col in enumerate(ing_cols):
#         meas_col = meas_cols[i] if i < len(meas_cols) else None

#         chunk = temp[[id_col, ing_col] + ([meas_col] if meas_col else [])].copy()
#         chunk = chunk.rename(columns={ing_col: "ingredient"})
#         if meas_col:
#             chunk = chunk.rename(columns={meas_col: "measure"})
#         else:
#             chunk["measure"] = pd.NA
#         chunk["slot"] = i + 1
#         rows.append(chunk)

#     long_df = pd.concat(rows, ignore_index=True)
#     long_df = long_df[long_df["ingredient"].notna() & (long_df["ingredient"] != "")]
#     return long_df


# def ingredient_frequency(ing_long: pd.DataFrame):
#     return (ing_long["ingredient"]
#             .value_counts()
#             .rename_axis("ingredient")
#             .to_frame("count")
#             .reset_index())


# # ----------- Visualization -----------

# def plot_top_ingredients(freq_df, top_n=30, outpath=None):
#     top = freq_df.head(top_n).iloc[::-1]
#     fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.25)))
#     ax.barh(top["ingredient"], top["count"])
#     ax.set_title(f"Top {top_n} Ingredients")
#     ax.set_xlabel("Count")
#     plt.tight_layout()
#     if outpath:
#         fig.savefig(outpath, dpi=200, bbox_inches="tight")
#     return fig, ax
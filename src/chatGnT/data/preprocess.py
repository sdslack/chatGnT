import pandas as pd
import re

# def clean_ingred(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     pattern = r"^(\d*\s?\d/\d+|\d+(?:\.\d+)?)\s+(\S+)\s+(.+)$"
#     extracted = df["ingredient_name"].str.extract(pattern)

#     valid = extracted.notnull().all(axis=1)
#     valid_ids = df.groupby("id").apply(
#         lambda x: valid[x.index].all()
#     )
#     valid_ids = valid_ids[valid_ids].index  # only ids with all valid rows
#     df = df[df["id"].isin(valid_ids)]

#     df["amt"] = extracted[0]
#     df["unit"] = extracted[1]
#     df["ingred"] = extracted[2]
    
#     return df


# TODO: start over with new functions?
def filter_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function takes in a dataframe with columns 'id', 'ingredient_name', and
    'ingredient_link' where 'ingredient_name' contains both the amount and
    ingredient name and 'ingredient_link' is a more general link to the
    ingredient.

    Returns a dataframe with:
        - recipes where ALL ingredients have no numberic values in the
            'ingredient_name' column removed


    TODO: finish this!
    """
    df = df.copy()

    # Remove recipes that have no numeric values in the ingredient_name column
    recipes_with_num = df.groupby('id')['ingredient_name'].apply(lambda x: x.str.contains(r'\d').any())
    recipe_no_num = recipes_with_num[~recipes_with_num].index
    df_with_num = df[~df['id'].isin(recipe_no_num)]

    return df_with_num


def clean_recipts():
    pass
    #TODO: this could call function extract_amount unit
    #TODO: this could also call function extract ingredient name


def extract_amount_unit(ingredient_name):
    # From ingredient_name
    # extract either the amount
    # the amount and unit
    # or neither (case where ingredient_name doesn't contain a number)
    pattern = r"""
        ^\s*
        (?P<amount>
            \d+\s\d+/\d+ |     # mixed number (e.g., 1 1/2)
            \d+/\d+      |     # fraction (e.g., 1/2)
            \d+(\.\d+)?        # integer or decimal
        )
        \s*
        (?P<unit>[a-zA-Z]+)?   # optional unit
    """
    
    match = re.match(pattern, ingredient_name, re.VERBOSE)
    
    if not match:
        return None, None

    amount_str = match.group("amount")
    unit = match.group("unit")

    # Convert amount to float
    if " " in amount_str:  # mixed number
        whole, frac = amount_str.split()
        num, denom = frac.split("/")
        amount = float(whole) + float(num) / float(denom)
    elif "/" in amount_str:  # fraction
        num, denom = amount_str.split("/")
        amount = float(num) / float(denom)
    else:
        amount = float(amount_str)

    return amount, unit
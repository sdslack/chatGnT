import pandas as pd
import re

#TODO: revisit which of these work with dfs and which work with strings... what
# is best practice?

def clean_recipes(df):
    """
    Extract amount, unit, and name for each ingredient.
    """
    df = df.copy()
    df[["amt", "unit"]] = pd.DataFrame(
        df["ingredient_name"].apply(extract_amount_unit).tolist(),
        index=df.index
    )

    df["ingred"] = df["ingredient_link"].apply(extract_ingredient_name)

    return df


def extract_amount_unit(ingredient_name):
    """
    TODO: testing code for this

    print(preprocess.extract_amount_unit("1 cup sugar"))       # (1.0, "cup")
    print(preprocess.extract_amount_unit("1/2 tsp salt"))     # (0.5, "tsp")
    print(preprocess.extract_amount_unit("1 1/2 cups milk"))  # (1.5, "cups")
    print(preprocess.extract_amount_unit("salt to taste"))  # (None, None)
    print(preprocess.extract_amount_unit("5 strawberries"))  # (5, None)
    print(preprocess.extract_amount_unit("1/3  Grand Marnier"))  # (0.25, None)
    """
    # From ingredient_name
    # extract either the amount
    # the amount and unit
    # or neither (case where ingredient_name doesn't contain a number)
    #TODO: could revisit "juice of 1/2 lemon" formatting. Currently ignores juice
    # since ingredient_link in db also did.

    units_list = r"""
        \b(?:cups?|oz|ounces?|grams?|g|gr|kg|lbs?|pounds?|tbsp|tblsp|tsp|twist|slice|
        jiggers?|parts?|shots?|dl|cl|ml|l|dash(?:es)?|quart|qt|gal|bottles?|scoops?|top|
        fill|to\s+fill|glass|pint|piece|drops?|fifth|cans?|wedges?|pinch(?:es)?|garnish|spoons?|
        float|whole|packages?|sprigs?|strips?|cubes?|to\s+taste|splash(?:es)?)\b
    """

    pattern = fr"""
        ^\s*
        (?:(?:juice|add)\s+(?:of\s+)?)?      # "juice of ", "add", etc
        (?P<amount>
            (?:
                \d+\s\d+/\d+ |     # mixed number (1 1/2)
                \d+/\d+      |     # fraction (1/2)
                \d+(\.\d+)?        # integer or decimal
            )
            (?:\s?-\s?\d+)?  # match ranges (1-2)
            \b
            (?![-\s]*proof|-(?!\d))  # don't match if followed by "-" or "proof"
        )?
        \s*
        (?P<unit>(?:small\s+|large\s+|long\s+)?{units_list})?  # optional unit from list of units
    """
    
    match = re.match(pattern, ingredient_name, re.VERBOSE | re.IGNORECASE)

    amount_str = match.group("amount")
    unit = match.group("unit")

    if amount_str is None and unit is None:
        return None, None

    unit_clean = unit.lower() if unit else None

    # Convert amount to float
    if amount_str is None:
        return None, unit_clean
    if "-" in amount_str:  # take first number if range
        amount_str = amount_str.split("-")[0].strip()
    if " " in amount_str:  # mixed number
        whole, frac = amount_str.split()
        num, denom = frac.split("/")
        amount = float(whole) + float(num) / float(denom)
    elif "/" in amount_str:  # fraction
        num, denom = amount_str.split("/")
        amount = float(num) / float(denom)
    else:
        amount = float(amount_str)

    return amount, unit_clean


def extract_ingredient_name(ingredient_link):
    """
    TODO: code for testing this
    print(preprocess.extract_ingredient_name("/ingredent/-Carrot"))
    print(preprocess.extract_ingredient_name("/ingredent/1-Vodka"))
    """
    # from string, extract only after /###-
    match = re.search(r"/(\d+)?-([\w-]+)", ingredient_link)
    if match:
        ingredient_name = match.group(2).replace("^-", "")
        return ingredient_name.lower()
    else:
        return None



def filter_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function takes in a dataframe with columns 'id', 'ingredient_name', and
    'ingredient_link' where 'ingredient_name' contains both the amount and
    ingredient name and 'ingredient_link' is a more general link to the
    ingredient.

    Returns a dataframe with:
        - recipes where ALL ingredients have no numberic values in the
            'ingredient_name' column removed

    """
    df = df.copy()

    # Remove recipes that have no non-NaN values in the amt column
    recipes_with_amt = df.groupby('id')['amt'].apply(lambda x: x.notna().any())
    recipe_no_amt = recipes_with_amt[~recipes_with_amt].index
    df_with_num = df[~df['id'].isin(recipe_no_amt)]

    # Add units to ingredients not in countable list
    #TODO: may want to revisit and include fewer items in this list!
    countable = [
        'allspice', 'anise', 'apple', 'apricot', 'banana', 'blackberries',
        'caramel-sauce', 'cardamom', 'carrot', 'cherry', 'chocolate-sauce',
        'cinnamon', 'cloves', 'coffee', 'egg', 'egg-white', 'egg-yolk', 'figs',
        'ginger', 'ice', 'kiwi', 'lemon', 'lemon-peel', 'lime', 'mango',
        'maraschino-cherry', 'mini-snickers-bars', 'mint', 'olive', 'orange',
        'orange-peel', 'oreo-cookie', 'papaya', 'pineapple', 'red-chili-flakes',
        'strawberries', 'sugar', 'whipped-cream', 'wormwood',
        'bitter-lemon', 'blackcurrant-squash', 'caramel-coloring',
        'carbonated-water', 'cayenne-pepper', 'champagne', 'cherries',
        'chocolate-syrup', 'club-soda', 'cucumber', 'fruit', 'ginger-ale',
        'lime-peel', 'marshmallows', 'nutmeg', 'orange-spiral', 'pepper',
        'powdered-sugar', 'salt', 'salted-chocolate', 'schweppes-russchian',
        'soda-water', 'tonic-water', 'whipping-cream',
        'coca-cola', 'cranberry-juice', 'cream', 'fruit-juice',
        'lemon-lime-soda', 'lemonade', 'light-cream', 'milk', 'orange-juice',
        'passion-fruit-juice', 'sour-mix', 'tea', 'water']

    # Add "part" to rows where amt != NA & <= 1, unit = NA, & not in countable
    # and "parts" to rows where amt != NA & > 1, unit = NA, & not in countable
    add_part = (
        (df_with_num['amt'].notna()) &
        (df_with_num['amt'] <= 1) &
        (df_with_num['unit'].isna()) &
        (~df_with_num['ingred'].isin(countable)))
    add_parts = (
        (df_with_num['amt'].notna()) &
        (df_with_num['amt'] > 1) & 
        (df_with_num['unit'].isna()) &
        (~df_with_num['ingred'].isin(countable)))

    df_with_num.loc[add_part, 'unit'] = 'part'
    df_with_num.loc[add_parts, 'unit'] = 'parts'

    # For rows where amt == NA and unit == NA, set unit to "add" if in countable list
    add_add = (
        (df_with_num['amt'].isna()) &
        (df_with_num['unit'].isna()) &
        (df_with_num['ingred'].isin(countable))
    )
    df_with_num.loc[add_add, 'unit'] = 'add'

    # Remove recipes where any ingredients still have both amt & unit = NA
    rows_na = df_with_num[df_with_num["amt"].isna() & df_with_num["unit"].isna()]
    df_filt = df_with_num[~df_with_num["id"].isin(rows_na["id"].unique())]

    # Combine amt and unit into a single column "amt_unit"
    df_filt['amt_unit'] = df_filt.apply(lambda row: combine_amt_unit(row['amt'], row['unit']), axis=1)

    return df_filt



def combine_amt_unit(amt, unit):
    """
    Combine amt and unit into a single column "amt_unit".
    
    TODO: test cases
    print(preprocess.combine_amt_unit(5.0, "oz"))
    print(preprocess.combine_amt_unit(np.nan, "dash"))
    print(preprocess.combine_amt_unit(2, np.nan))
    """

    amt = str(amt) if pd.notna(amt) else ''
    unit = str(unit) if pd.notna(unit) else ''
    
    # Remove trailing .0 if you want "5" instead of "5.0"
    amt = amt.replace('.0', '') if amt.endswith('.0') else amt
    unit = unit.replace('.0', '') if unit.endswith('.0') else unit
    
    return f"{amt} {unit}".strip()


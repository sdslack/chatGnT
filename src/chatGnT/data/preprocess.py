import numbers
import pandas as pd
import re

#TODO: revisit which of these work with dfs and which work with strings... what
# is best practice?

PARTS_PER_UNIT = {
    "oz": 1.0,
    "ounce": 1.0,
    "ounces": 1.0,
    "cup": 8.0,
    "cups": 8.0,
    "tbsp": 0.5,
    "tblsp": 0.5,
    "tsp": 1 / 6,
    "jigger": 1.5,
    "jiggers": 1.5,
    "shot": 1.5,
    "shots": 1.5,
    "ml": 1 / 30,
    "cl": 1 / 3,
    "dl": 10 / 3,
    "l": 1000 / 30,
    "quart": 32.0,
    "qt": 32.0,
    "gal": 128.0,
    "pint": 16.0,
}

UNITS_LIST = r"""
    \b(?:cups?|oz|ounces?|grams?|g|gr|kg|lbs?|pounds?|tbsp|tblsp|tsp|twist|
    slice|jiggers?|parts?|shots?|dl|cl|ml|l|dash(?:es)?|quart|qt|gal|bottles?|
    scoops?|top|fill|to\s+fill|glass|pint|piece|drops?|fifth|cans?|wedges?|
    pinch(?:es)?|garnish|spoons?|float|whole|packages?|sprigs?|strips?|cubes?|
    to\s+taste|splash(?:es)?)\b
"""

AMOUNT_UNIT_PATTERN = re.compile(
    fr"""
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
    (?P<unit>(?:small\s+|large\s+|long\s+)?{UNITS_LIST})?  # optional unit from list of units
    """,
    re.VERBOSE | re.IGNORECASE
)

UNITLESS_INGREDIENTS = [
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
    'passion-fruit-juice', 'sour-mix', 'tea', 'water'
]

def clean_recipes(df):
    """
    Extract amount, unit, and name for each ingredient. Convert all units to
    parts where makes sense.
    """
    df = df.copy()
    df[["amt", "unit"]] = pd.DataFrame(
        df["ingredient_name"].apply(extract_amount_unit).tolist(),
        index=df.index
    )

    df[["amt", "unit"]] = pd.DataFrame(
        df.apply(lambda row: convert_to_parts(row["amt"], row["unit"]), axis=1).tolist(),
        index=df.index
    )

    df["ingred"] = df["ingredient_link"].apply(extract_ingredient_name)

    return df


def extract_amount_unit(ingredient_name):
    """
    Extract a numeric amount and unit from an ingredient string.
    """
    # From ingredient_name
        # extract either the amount
        # the amount and unit
        # or neither (case where ingredient_name doesn't contain a number)
    #TODO: could revisit "juice of 1/2 lemon" formatting. Currently ignores juice
    # since ingredient_link in db also did.

    match = AMOUNT_UNIT_PATTERN.match(ingredient_name)

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


def convert_to_parts(amount, unit):
    """
    Convert supported volumetric units into cocktail-style parts.

    Assumes 1 oz == 1 part and leaves ambiguous or non-volumetric units alone.
    """
    if pd.isna(amount) or pd.isna(unit):
        return amount, unit

    unit_clean = unit.lower().strip()
    unit_clean = re.sub(r"^(small|large|long)\s+", "", unit_clean)

    parts_per_unit = PARTS_PER_UNIT.get(unit_clean)
    if parts_per_unit is None:
        return amount, unit

    parts_amount = round(amount * parts_per_unit, 3)
    parts_unit = "part" if parts_amount <= 1 else "parts"

    return parts_amount, parts_unit


def extract_ingredient_name(ingredient_link):
    """
    Extract the normalized ingredient name from an ingredient link.
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
        - recipes whose total normalized parts exceed 8 removed

    """
    df = df.copy()

    # Remove recipes that have no non-NaN values in the amt column
    recipes_with_amt = df.groupby('id')['amt'].apply(lambda x: x.notna().any())
    recipe_no_amt = recipes_with_amt[~recipes_with_amt].index
    df_with_num = df[~df['id'].isin(recipe_no_amt)]

    # Add "part" to rows where amt != NA & <= 1, unit = NA, & not in UNITLESS_INGREDIENTS
    # and "parts" to rows where amt != NA & > 1, unit = NA, & not in UNITLESS_INGREDIENTS
    add_part = (
        (df_with_num['amt'].notna()) &
        (df_with_num['amt'] <= 1) &
        (df_with_num['unit'].isna()) &
        (~df_with_num['ingred'].isin(UNITLESS_INGREDIENTS)))
    add_parts = (
        (df_with_num['amt'].notna()) &
        (df_with_num['amt'] > 1) & 
        (df_with_num['unit'].isna()) &
        (~df_with_num['ingred'].isin(UNITLESS_INGREDIENTS)))

    df_with_num.loc[add_part, 'unit'] = 'part'
    df_with_num.loc[add_parts, 'unit'] = 'parts'

    # For rows where amt == NA and unit == NA, set unit to "add" if in UNITLESS_INGREDIENTS
    add_add = (
        (df_with_num['amt'].isna()) &
        (df_with_num['unit'].isna()) &
        (df_with_num['ingred'].isin(UNITLESS_INGREDIENTS))
    )
    df_with_num.loc[add_add, 'unit'] = 'add'

    # Remove recipes where any ingredients still have both amt & unit = NA
    rows_na = df_with_num[df_with_num["amt"].isna() & df_with_num["unit"].isna()]
    df_filt = df_with_num[~df_with_num["id"].isin(rows_na["id"].unique())]

    # Remove recipes that are likely batch and not for single drink
    total_parts = (
        df_filt[df_filt["unit"].isin(["part", "parts"])]
        .groupby("id")["amt"]
        .sum()
    )
    batch_recipe_ids = total_parts[total_parts > 8].index
    df_filt = df_filt[~df_filt["id"].isin(batch_recipe_ids)].copy()

    # Combine amt and unit into a single column "amt_unit"
    df_filt['amt_unit'] = df_filt.apply(lambda row: combine_amt_unit(row['amt'], row['unit']), axis=1)

    return df_filt


def combine_amt_unit(amt, unit):
    """
    Combine amt and unit into a single column "amt_unit".
    """
    if pd.notna(amt):
        if isinstance(amt, numbers.Real) and not isinstance(amt, bool):
            amt = f"{float(round(amt, 3)):.3f}".rstrip("0").rstrip(".")
        else:
            amt = str(amt)
    else:
        amt = ''
    unit = str(unit) if pd.notna(unit) else ''

    unit = unit.replace('.0', '') if unit.endswith('.0') else unit
    
    return f"{amt} {unit}".strip()

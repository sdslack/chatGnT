#NOTE: this script is in data/ not because it is dataset specific, but becasue
# would use it on all datasets as final step of preprocessing.


def recipe_to_tokens(df):
    """
    Convert a DataFrame with columns ['amount', 'unit', 'ingred'] into structured token sequences.

    Args:
        df (pd.DataFrame): Must have columns ['amount', 'unit', 'ingred']

    Returns:
        List[List[str]]: Each recipe as a list of tokens
    """
    recipes_tokens = []
    df = df.copy()

    # Assume df has a recipe identifier column; if not, treat all rows as one recipe
    grouped = df.groupby('id')  #TODO: this is file specific

    for _, recipe in grouped:
        tokens = []
        for _, row in recipe.iterrows():
            amt = str(row['amt']).strip()
            unit = str(row['unit']).strip()  # if pd.notna(row['unit']) else "unit"
            ingred = str(row['ingred']).strip()
            # if qty.lower() == 'nan' or ing.lower() == 'nan':
            #     continue
            tokens += [f"<amt>{amt}</amt>", f"<unit>{unit}</unit>", f"<ingred>{ingred}</ingred>", "<sep>"]
        recipes_tokens.append(tokens)

    return recipes_tokens


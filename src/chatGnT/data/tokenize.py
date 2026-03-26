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

def make_vocab(recipes_tokens):
    """
    Create a vocabulary from tokenized recipes.

    Args:
        recipes_tokens (List[List[str]]): List of tokenized recipes
    """
    # Flatten all tokens into a single list
    tokens_flat = [token for recipe in recipes_tokens for token in recipe]
    # Get unique tokens and assign an ID to each
    vocab = {token: i+1 for i, token in enumerate(sorted(set(tokens_flat)))}
    # Add a padding token for batching & a recipe end token
    vocab["<pad>"] = 0
    vocab["<end>"] = len(vocab)

    return vocab

def invert_vocab(vocab):
    """
    Create an inverse vocabulary mapping from ID to token.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping token to ID

    Returns:
        Dict[int, str]: Inverse vocabulary mapping ID to token
    """
    return {i: token for token, i in vocab.items()}

def embed_tokens(recipes_tokens, vocab):
    """
    Convert tokenized recipes into sequences of token IDs.

    Args:
        recipes_tokens (List[List[str]]): List of tokenized recipes
        vocab (Dict[str, int]): Vocabulary mapping token to ID

    Returns:
        List[List[int]]: List of recipes represented as sequences of token IDs
    """
    tokens_encoded = [[vocab[t] for t in recipe] for recipe in recipes_tokens]
    lengths = [len(r) for r in tokens_encoded]

    # Now add end token and pad to same length (seq_length = 48 + 1 for end token)
    max_len = max(len(r) for r in tokens_encoded)

    # Add end token and pad
    tokens_padded = [
        r + [vocab["<end>"]] + [vocab["<pad>"]] * (max_len - len(r))
        for r in tokens_encoded
    ]

    return tokens_padded

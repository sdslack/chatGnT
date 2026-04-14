#NOTE: this script is in data/ not because it is dataset specific, but becasue
# would use it on all datasets as final step of preprocessing.

# st = single task, mt = multi task

def recipe_to_tokens_st(df):
    """
    Convert a DataFrame with columns ['amt_unit', 'ingred'] into structured token sequences.

    Args:
        df (pd.DataFrame): Must have columns ['amt_unit', 'ingred']

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
            amt_unit = str(row['amt_unit']).strip()
            ingred = str(row['ingred']).strip()
            tokens += [f"<amt>{amt_unit}</amt>", f"<ingred>{ingred}</ingred>"]
        recipes_tokens.append(tokens)

    return recipes_tokens

def recipe_to_tokens_mt(df):
    """
    Convert a DataFrame with columns ['amt_unit', 'ingred'] into structured token sequences.

    Args:
        df (pd.DataFrame): Must have columns ['amt_unit', 'ingred']

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
            amt_unit = str(row['amt_unit']).strip()
            ingred = str(row['ingred']).strip()
            # tokens += [(f"<amt>{amt_unit}</amt>", f"<ingred>{ingred}</ingred>")]
            tokens.append((f"<amt>{amt_unit}</amt>", f"<ingred>{ingred}</ingred>"))
        recipes_tokens.append(tokens)

    return recipes_tokens

def make_vocab_st(recipes_tokens):
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

def make_vocab_mt(recipes_tokens):
    """
    Create a vocabulary from tokenized recipes.

    Args:
        recipes_tokens (List[List[str]]): List of tokenized recipes
    """
    amt_set = set()
    ingred_set = set()

    for recipe in recipes_tokens:
        for amt, ingred in recipe:
            amt_set.add(amt)
            ingred_set.add(ingred)

    vocab_amt = {amt: i+1 for i, amt in enumerate(sorted(amt_set))}
    vocab_ingred = {ingred: i+1 for i, ingred in enumerate(sorted(ingred_set))}

    # Add a padding token for batching & a recipe end token
    vocab_amt["<pad>"] = 0
    vocab_amt["<end>"] = len(vocab_amt)

    vocab_ingred["<pad>"] = 0
    vocab_ingred["<end>"] = len(vocab_ingred)

    return vocab_amt, vocab_ingred

def invert_vocab_st(vocab):
    """
    Create an inverse vocabulary mapping from ID to token.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping token to ID

    Returns:
        Dict[int, str]: Inverse vocabulary mapping ID to token
    """
    return {i: token for token, i in vocab.items()}

def invert_vocab_mt(vocab_amt, vocab_ingred):
    """
    Create an inverse vocabulary mapping from ID to token.

    Args:
        vocab (Dict[str, int]): Vocabulary mapping token to ID

    Returns:
        Dict[int, str]: Inverse vocabulary mapping ID to token
    """
    inv_vocab_amt = {i: token for token, i in vocab_amt.items()}
    inv_vocab_ingred = {i: token for token, i in vocab_ingred.items()}
    return inv_vocab_amt, inv_vocab_ingred

def embed_tokens_st(recipes_tokens, vocab):
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

def embed_tokens_mt(recipes_tokens, vocab_amt, vocab_ingred):
    """
    Convert tokenized recipes into sequences of token IDs.

    Args:
        recipes_tokens (List[List[str]]): List of tokenized recipes
        vocab (Dict[str, int]): Vocabulary mapping token to ID

    Returns:
        List[List[int]]: List of recipes represented as sequences of token IDs
    """
    recipes_encoded = []
    all_amt_seqs = []
    all_ingred_seqs = []

    for recipe in recipes_tokens:
        amt_ids = [vocab_amt[amt] for amt, _ in recipe]
        ingred_ids = [vocab_ingred[ingred] for _, ingred in recipe]

        # Add end token
        amt_ids.append(vocab_amt["<end>"])
        ingred_ids.append(vocab_ingred["<end>"])

        all_amt_seqs.append(amt_ids)
        all_ingred_seqs.append(ingred_ids)

    # Add padding
    max_len = max(len(seq) for seq in all_amt_seqs)
    for amt_ids, ingred_ids in zip(all_amt_seqs, all_ingred_seqs):
        pad_len = max_len - len(amt_ids)
        amt_ids_padded = amt_ids + [vocab_amt["<pad>"]] * pad_len
        ingred_ids_padded = ingred_ids + [vocab_ingred["<pad>"]] * pad_len
        recipes_encoded.append((amt_ids_padded, ingred_ids_padded))


    # Optional: pad sequences to max length in batch later in collate_fn
    return recipes_encoded

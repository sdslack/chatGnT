import re
import torch
from chatGnT.data import load, preprocess, tokenize
from chatGnT.models.structure import mask_single_task_next_logits

# st = single task, mt = multi task

def _normalize_lookup_text(text):
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def _normalize_ingredient_name(text):
    normalized = re.sub(r"[^a-z0-9]+", "-", str(text).strip().casefold())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("User input must contain an ingredient or recipe name.")
    return normalized


def _search_text_columns(df, value):
    matches = []
    normalized_recipe = _normalize_lookup_text(value)
    for column in df.columns:
        series = df[column]
        if not (
            getattr(series.dtype, "kind", None) in {"O", "U", "S"}
            or str(series.dtype).startswith("string")
        ):
            continue

        normalized_values = series.dropna().map(_normalize_lookup_text)
        if normalized_values.empty:
            continue

        matched_index = normalized_values[normalized_values == normalized_recipe].index
        if len(matched_index) > 0:
            matches.append((column, df.loc[matched_index].copy()))

    if not matches:
        return None

    matches.sort(key=lambda item: len(item[1]))
    return matches[0][1]


def _has_preprocess_columns(df):
    return {"ingredient_name", "ingredient_link"}.issubset(df.columns)


def _ensure_recipe_id_column(df):
    normalized = df.copy()
    if "id" in normalized.columns:
        return normalized

    for column in normalized.columns:
        if column in {"ingredient_name", "ingredient_link", "ingred", "amt", "unit", "amt_unit"}:
            continue
        unique_values = normalized[column].dropna().unique()
        if len(unique_values) == 1:
            normalized["id"] = normalized[column]
            return normalized

    raise ValueError(
        "Recipe rows were found, but no recipe identifier column could be normalized to 'id'."
    )


def _resolve_recipe_ingredient_rows(recipe_name, recipes_df):
    recipe_rows = _search_text_columns(recipes_df, recipe_name)
    if recipe_rows is None:
        return None

    if not _has_preprocess_columns(recipe_rows):
        raise ValueError(
            "Recipe rows were found, but they do not contain ingredient_name and ingredient_link."
        )

    return _ensure_recipe_id_column(recipe_rows)


def prepare_mt_start_tokens(user_input, drinks=None, ingred=None):
    """
    Convert a raw user string into the ``start_tokens`` format required by
    ``predict_mt``.

    Single ingredients are treated as one part, e.g. ``gin`` becomes:
    ``[('<amt>1 part</amt>', '<ingred>gin</ingred>')]``

    Recipe names are looked up in the raw drinks dataframe, then passed through
    the existing clean/filter/tokenize pipeline so the returned tokens match the
    training format.
    """
    raw_input = str(user_input).strip()
    if not raw_input:
        raise ValueError("User input must contain an ingredient or recipe name.")

    recipes_df = None
    if drinks is not None and _has_preprocess_columns(drinks):
        recipes_df = drinks
    elif ingred is not None and _has_preprocess_columns(ingred):
        recipes_df = ingred
    elif drinks is None and ingred is None:
        recipes_df = load.load_all()

    recipe_rows = None if recipes_df is None else _resolve_recipe_ingredient_rows(raw_input, recipes_df)
    if recipe_rows is not None:
        cleaned = preprocess.clean_recipes(recipe_rows)
        filtered = preprocess.filter_recipes(cleaned)
        if filtered.empty:
            raise ValueError(
                f"Recipe '{raw_input}' was found but was removed by preprocessing."
            )

        recipes_tokens = tokenize.recipe_to_tokens_mt(filtered)
        if not recipes_tokens:
            raise ValueError(
                f"Recipe '{raw_input}' was found but no tokens could be generated."
            )
        return recipes_tokens[0]

    ingredient = _normalize_ingredient_name(raw_input)
    return [(f"<amt>1 part</amt>", f"<ingred>{ingredient}</ingred>")]


def predict_st(model, device, pad_id, vocab, inv_vocab, start_ingred, max_len=50, temperature=0.8):
    model.eval()

    # Start sequence with group of tokens: amt, unit, ingred, sep
    ids = [vocab[i] for i in start_ingred]

    for _ in range(max_len):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)

        # padding mask
        pad_mask = (x == pad_id).transpose(0, 1)  # (batch, seq_len)

        # causal mask
        seq_len = x.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        with torch.no_grad():
            output = model(
                src=x,
                src_key_padding_mask=pad_mask,
                src_mask=src_mask
            )

            logits = output[-1, 0, :]
            logits = mask_single_task_next_logits(logits, vocab, current_length=len(ids))

            # Sampling for diversity
            # Convert to probabilities with temperature
            probs = torch.softmax(logits / temperature, dim=-1)
            # Sample for next token
            next_id = torch.multinomial(probs, num_samples=1).item()

        # Append prediction
        ids.append(next_id)

        # Stop if end token
        if inv_vocab[next_id] == "<end>":
            break

    # Convert back to tokens
    tokens = [inv_vocab[i] for i in ids]
    return tokens


def predict_mt(model, device, pad_id_amt, pad_id_ingred, vocab_amt, vocab_ingred, inv_vocab_amt, inv_vocab_ingred, start_tokens, max_len=50, temperature=0.8):
    model.eval()

    # Start sequence with group of tokens: amt, unit, ingred, sep
    amt_ids = [vocab_amt[amt] for amt, _ in start_tokens]
    ingred_ids = [vocab_ingred[ingred] for _, ingred in start_tokens]

    for _ in range(max_len):
        x_amt = torch.tensor(amt_ids, dtype=torch.long, device=device).unsqueeze(1)     # (seq_len, 1)
        x_ingred = torch.tensor(ingred_ids, dtype=torch.long, device=device).unsqueeze(1)

        # padding mask
        pad_mask = (
            (x_amt == pad_id_amt) |
            (x_ingred == pad_id_ingred)
        ).transpose(0, 1)
        
        # causal mask
        seq_len = x_amt.size(0)
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)

        with torch.no_grad():
            output_amt, output_ingred = model(
                src_amt=x_amt,
                src_ingred=x_ingred,
                src_key_padding_mask=pad_mask,
                src_mask=src_mask
            )

            logits_amt = output_amt[-1, 0, :]
            logits_ingred = output_ingred[-1, 0, :]

            # Sampling for diversity
            # Convert to probabilities with temperature
            probs_amt = torch.softmax(logits_amt / temperature, dim=-1)
            probs_ingred = torch.softmax(logits_ingred / temperature, dim=-1)
            # Sample for next token
            next_id_amt = torch.multinomial(probs_amt, num_samples=1).item()
            next_id_ingred = torch.multinomial(probs_ingred, num_samples=1).item()

        # Append prediction
        amt_ids.append(next_id_amt)
        ingred_ids.append(next_id_ingred)

        # Stop if end token
        if inv_vocab_amt[next_id_amt] == "<end>" or inv_vocab_ingred[next_id_ingred] == "<end>":
            break

    # Convert back to tokens
    tokens = [(inv_vocab_amt[i], inv_vocab_ingred[j]) for i, j in zip(amt_ids, ingred_ids)]
    return tokens

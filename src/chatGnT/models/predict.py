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


def _load_recipe_source(drinks=None, ingred=None):
    if drinks is not None and _has_preprocess_columns(drinks):
        return drinks
    if ingred is not None and _has_preprocess_columns(ingred):
        return ingred
    if drinks is None and ingred is None:
        return load.load_all()
    return None


def _prepare_recipe_tokens(user_input, recipe_tokenizer, drinks=None, ingred=None):
    raw_input = str(user_input).strip()
    if not raw_input:
        raise ValueError("User input must contain an ingredient or recipe name.")

    recipes_df = _load_recipe_source(drinks=drinks, ingred=ingred)
    if recipes_df is None:
        return raw_input, None

    recipe_rows = _resolve_recipe_ingredient_rows(raw_input, recipes_df)
    if recipe_rows is None:
        return raw_input, None

    cleaned = preprocess.clean_recipes(recipe_rows)
    filtered = preprocess.filter_recipes(cleaned)
    if filtered.empty:
        return raw_input, None

    recipes_tokens = recipe_tokenizer(filtered)
    if not recipes_tokens:
        return raw_input, None

    return raw_input, recipes_tokens[0]


def _prepare_single_ingredient_mt(raw_input):
    ingredient = _normalize_ingredient_name(raw_input)
    return [("<amt>1 part</amt>", f"<ingred>{ingredient}</ingred>")]


def _prepare_single_ingredient_st(raw_input):
    ingredient = _normalize_ingredient_name(raw_input)
    return ["<amt>1 part</amt>", f"<ingred>{ingredient}</ingred>"]


def _strip_token_tag(token, tag):
    prefix = f"<{tag}>"
    suffix = f"</{tag}>"
    if isinstance(token, str) and token.startswith(prefix) and token.endswith(suffix):
        return token[len(prefix):-len(suffix)].strip()
    return str(token).strip()


def _humanize_ingredient_name(name):
    return str(name).replace("-", " ").strip()


def _trim_terminal_tokens_st(tokens):
    trimmed = list(tokens)
    while trimmed and trimmed[-1] in {"<end>", "<pad>"}:
        trimmed.pop()
    return trimmed


def _trim_terminal_tokens_mt(tokens):
    trimmed = list(tokens)
    while trimmed and (
        trimmed[-1][0] in {"<end>", "<pad>"} or trimmed[-1][1] in {"<end>", "<pad>"}
    ):
        trimmed.pop()
    return trimmed


def _truncate_recipe_prompt_mt(tokens, recipe_prefix_len=None):
    if recipe_prefix_len is None:
        return tokens
    if recipe_prefix_len <= 0:
        raise ValueError("recipe_prefix_len must be a positive integer or None.")
    return list(tokens[:recipe_prefix_len])


def _truncate_recipe_prompt_st(tokens, recipe_prefix_len=None):
    if recipe_prefix_len is None:
        return tokens
    if recipe_prefix_len <= 0:
        raise ValueError("recipe_prefix_len must be a positive integer or None.")
    return list(tokens[: recipe_prefix_len * 2])


def prepare_mt_start_tokens(user_input, drinks=None, ingred=None, recipe_prefix_len=None):
    """
    Convert a raw user string into the ``start_tokens`` format required by
    ``predict_mt``.

    Single ingredients are treated as one part, e.g. ``gin`` becomes:
    ``[('<amt>1 part</amt>', '<ingred>gin</ingred>')]``

    Recipe names are looked up in the raw drinks dataframe, then passed through
    the existing clean/filter/tokenize pipeline so the returned tokens match the
    training format.

    If ``recipe_prefix_len`` is set, recipe-name matches are truncated to that
    many ingredient pairs before generation.
    """
    raw_input, recipe_tokens = _prepare_recipe_tokens(
        user_input,
        tokenize.recipe_to_tokens_mt,
        drinks=drinks,
        ingred=ingred,
    )
    if recipe_tokens is not None:
        return _truncate_recipe_prompt_mt(recipe_tokens, recipe_prefix_len)
    return _prepare_single_ingredient_mt(raw_input)


def prepare_st_start_tokens(user_input, drinks=None, ingred=None, recipe_prefix_len=None):
    """
    Convert a raw user string into the token sequence required by ``predict_st``.

    Single ingredients are treated as one part, e.g. ``gin`` becomes:
    ``['<amt>1 part</amt>', '<ingred>gin</ingred>']``

    Recipe names are looked up in the raw drinks dataframe, then passed through
    the existing clean/filter/tokenize pipeline so the returned tokens match the
    training format.

    If ``recipe_prefix_len`` is set, recipe-name matches are truncated to that
    many ingredient pairs before generation.
    """
    raw_input, recipe_tokens = _prepare_recipe_tokens(
        user_input,
        tokenize.recipe_to_tokens_st,
        drinks=drinks,
        ingred=ingred,
    )
    if recipe_tokens is not None:
        return _truncate_recipe_prompt_st(recipe_tokens, recipe_prefix_len)
    return _prepare_single_ingredient_st(raw_input)


def format_prediction_st(tokens=None, ids=None, inv_vocab=None):
    """
    Convert a single-task prediction into readable ingredient lines.

    Pass either decoded ``tokens`` or token ``ids`` with ``inv_vocab``.
    """
    if tokens is None:
        if ids is None or inv_vocab is None:
            raise ValueError("Pass either tokens or ids with inv_vocab.")
        tokens = [inv_vocab[i] for i in ids]

    lines = []
    current_amt = None
    for token in tokens:
        if token in {"<end>", "<pad>"}:
            break
        if token.startswith("<amt>"):
            current_amt = _strip_token_tag(token, "amt")
        elif token.startswith("<ingred>"):
            ingred = _humanize_ingredient_name(_strip_token_tag(token, "ingred"))
            if current_amt:
                lines.append(f"{current_amt} {ingred}")
            else:
                lines.append(ingred)
            current_amt = None

    return lines


def format_prediction_mt(tokens=None, amt_ids=None, ingred_ids=None, inv_vocab_amt=None, inv_vocab_ingred=None):
    """
    Convert a multi-task prediction into readable ingredient lines.

    Pass either decoded ``tokens`` or paired token id lists with inverse vocabs.
    """
    if tokens is None:
        if None in {amt_ids, ingred_ids, inv_vocab_amt, inv_vocab_ingred}:
            raise ValueError("Pass either tokens or amt_ids/ingred_ids with inverse vocabs.")
        tokens = [(inv_vocab_amt[i], inv_vocab_ingred[j]) for i, j in zip(amt_ids, ingred_ids)]

    lines = []
    for amt_token, ingred_token in tokens:
        if amt_token in {"<end>", "<pad>"} or ingred_token in {"<end>", "<pad>"}:
            break
        amt = _strip_token_tag(amt_token, "amt")
        ingred = _humanize_ingredient_name(_strip_token_tag(ingred_token, "ingred"))
        lines.append(f"{amt} {ingred}".strip())

    return lines


def _resolve_prediction_device(model, device=None):
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_st_from_input(
    model,
    vocab,
    user_input,
    drinks=None,
    ingred=None,
    recipe_prefix_len=None,
    device=None,
    max_len=50,
    temperature=0.8,
):
    """
    Notebook-friendly wrapper that prepares the prompt, runs single-task
    generation, and formats the output.
    """
    device = _resolve_prediction_device(model, device)
    model = model.to(device)
    inv_vocab = {v: k for k, v in vocab.items()}
    start_tokens = prepare_st_start_tokens(
        user_input,
        drinks=drinks,
        ingred=ingred,
        recipe_prefix_len=recipe_prefix_len,
    )
    tokens = predict_st(
        model,
        device,
        vocab["<pad>"],
        vocab,
        inv_vocab,
        start_tokens,
        max_len=max_len,
        temperature=temperature,
    )
    return {
        "start_tokens": start_tokens,
        "tokens": tokens,
        "lines": format_prediction_st(tokens=tokens),
    }


def generate_mt_from_input(
    model,
    vocab_amt,
    vocab_ingred,
    user_input,
    drinks=None,
    ingred=None,
    recipe_prefix_len=None,
    device=None,
    max_len=50,
    temperature=0.8,
):
    """
    Notebook-friendly wrapper that prepares the prompt, runs multi-task
    generation, and formats the output.
    """
    device = _resolve_prediction_device(model, device)
    model = model.to(device)
    inv_vocab_amt = {v: k for k, v in vocab_amt.items()}
    inv_vocab_ingred = {v: k for k, v in vocab_ingred.items()}
    start_tokens = prepare_mt_start_tokens(
        user_input,
        drinks=drinks,
        ingred=ingred,
        recipe_prefix_len=recipe_prefix_len,
    )
    tokens = predict_mt(
        model,
        device,
        vocab_amt["<pad>"],
        vocab_ingred["<pad>"],
        vocab_amt,
        vocab_ingred,
        inv_vocab_amt,
        inv_vocab_ingred,
        start_tokens,
        max_len=max_len,
        temperature=temperature,
    )
    return {
        "start_tokens": start_tokens,
        "tokens": tokens,
        "lines": format_prediction_mt(tokens=tokens),
    }


def predict_st(model, device, pad_id, vocab, inv_vocab, start_ingred, max_len=50, temperature=0.8):
    model.eval()

    start_ingred = _trim_terminal_tokens_st(start_ingred)

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

        # Stop if end token
        if inv_vocab[next_id] == "<end>":
            break

        # Append prediction
        ids.append(next_id)

    # Convert back to tokens
    tokens = [inv_vocab[i] for i in ids]
    return tokens


def predict_mt(model, device, pad_id_amt, pad_id_ingred, vocab_amt, vocab_ingred, inv_vocab_amt, inv_vocab_ingred, start_tokens, max_len=50, temperature=0.8):
    model.eval()

    start_tokens = _trim_terminal_tokens_mt(start_tokens)

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

        # Stop if end token
        if inv_vocab_amt[next_id_amt] == "<end>" or inv_vocab_ingred[next_id_ingred] == "<end>":
            break

        # Append prediction
        amt_ids.append(next_id_amt)
        ingred_ids.append(next_id_ingred)

    # Convert back to tokens
    tokens = [(inv_vocab_amt[i], inv_vocab_ingred[j]) for i, j in zip(amt_ids, ingred_ids)]
    return tokens

def load_model_mt(path, model_class):
    checkpoint = torch.load(path, map_location="cpu")

    model = model_class(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return (
        model,
        checkpoint["vocab_amt"],
        checkpoint["vocab_ingred"],
        checkpoint["config"],
    )

def load_model_st(path, model_class):
    checkpoint = torch.load(path, map_location="cpu")

    model = model_class(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return (
        model,
        checkpoint["vocab"],
        checkpoint["config"],
    )

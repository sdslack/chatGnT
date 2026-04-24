import pandas as pd
import pytest
import torch

from chatGnT.models.predict import (
    format_prediction_mt,
    format_prediction_st,
    predict_mt,
    predict_st,
    prepare_mt_start_tokens,
    prepare_st_start_tokens,
)


EMPTY_DRINKS = pd.DataFrame({"placeholder": []})
EMPTY_INGRED = pd.DataFrame({"placeholder": []})


def test_prepare_mt_start_tokens_for_single_ingredient():
    tokens = prepare_mt_start_tokens("gin", drinks=EMPTY_DRINKS, ingred=EMPTY_INGRED)

    assert tokens == [("<amt>1 part</amt>", "<ingred>gin</ingred>")]


def test_prepare_mt_start_tokens_normalizes_single_ingredient():
    tokens = prepare_mt_start_tokens(
        " Cranberry Juice ",
        drinks=EMPTY_DRINKS,
        ingred=EMPTY_INGRED,
    )

    assert tokens == [
        ("<amt>1 part</amt>", "<ingred>cranberry-juice</ingred>")
    ]


def test_prepare_mt_start_tokens_for_recipe_name():
    drinks = pd.DataFrame(
        {
            "id": [7, 7],
            "strDrink": ["Cosmopolitan", "Cosmopolitan"],
            "ingredient_name": ["1 1/2 oz vodka", "1 oz cranberry juice"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Cranberry-Juice",
            ],
        }
    )

    tokens = prepare_mt_start_tokens("Cosmopolitan", drinks=drinks)

    assert tokens == [
        ("<amt>1.5 parts</amt>", "<ingred>vodka</ingred>"),
        ("<amt>1 part</amt>", "<ingred>cranberry-juice</ingred>"),
    ]


def test_prepare_mt_start_tokens_can_truncate_recipe_prompt():
    drinks = pd.DataFrame(
        {
            "id": [7, 7],
            "strDrink": ["Cosmopolitan", "Cosmopolitan"],
            "ingredient_name": ["1 1/2 oz vodka", "1 oz cranberry juice"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Cranberry-Juice",
            ],
        }
    )

    tokens = prepare_mt_start_tokens("Cosmopolitan", drinks=drinks, recipe_prefix_len=1)

    assert tokens == [("<amt>1.5 parts</amt>", "<ingred>vodka</ingred>")]


def test_prepare_mt_start_tokens_finds_recipe_name_without_hard_coded_name_column():
    recipes = pd.DataFrame(
        {
            "recipe_key": [42, 42],
            "title": ["Cosmopolitan", "Cosmopolitan"],
            "notes": ["classic sour-style cocktail", "classic sour-style cocktail"],
            "ingredient_name": ["1 1/2 oz vodka", "1 oz cranberry juice"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Cranberry-Juice",
            ],
        }
    )

    tokens = prepare_mt_start_tokens("Cosmopolitan", drinks=recipes)

    assert tokens == [
        ("<amt>1.5 parts</amt>", "<ingred>vodka</ingred>"),
        ("<amt>1 part</amt>", "<ingred>cranberry-juice</ingred>"),
    ]


def test_prepare_mt_start_tokens_recipe_lookup_is_case_insensitive():
    drinks = pd.DataFrame(
        {
            "id": [9],
            "name": ["Cosmopolitan"],
            "ingredient_name": ["2 oz vodka"],
            "ingredient_link": ["/ingredient/1-Vodka"],
        }
    )

    tokens = prepare_mt_start_tokens(" cosmopolitan ", drinks=drinks)

    assert tokens == [("<amt>2 parts</amt>", "<ingred>vodka</ingred>")]


def test_prepare_mt_start_tokens_falls_back_to_ingredient_when_recipe_is_filtered_out():
    drinks = pd.DataFrame(
        {
            "id": [3, 3],
            "strDrink": ["Mystery Drink", "Mystery Drink"],
            "ingredient_name": ["vodka", "lime"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Lime",
            ],
        }
    )

    tokens = prepare_mt_start_tokens("Mystery Drink", drinks=drinks)

    assert tokens == [("<amt>1 part</amt>", "<ingred>mystery-drink</ingred>")]


def test_prepare_mt_start_tokens_falls_back_to_ingredient_when_no_recipe_rows_are_usable():
    drinks = pd.DataFrame(
        {
            "idDrink": [42],
            "strDrink": ["Cosmopolitan"],
        }
    )

    tokens = prepare_mt_start_tokens("Cosmopolitan", drinks=drinks, ingred=EMPTY_INGRED)

    assert tokens == [("<amt>1 part</amt>", "<ingred>cosmopolitan</ingred>")]


def test_prepare_mt_start_tokens_rejects_empty_input():
    with pytest.raises(ValueError, match="must contain an ingredient or recipe name"):
        prepare_mt_start_tokens("   ", drinks=EMPTY_DRINKS, ingred=EMPTY_INGRED)


def test_prepare_st_start_tokens_for_single_ingredient():
    tokens = prepare_st_start_tokens("gin", drinks=EMPTY_DRINKS, ingred=EMPTY_INGRED)

    assert tokens == ["<amt>1 part</amt>", "<ingred>gin</ingred>"]


def test_prepare_st_start_tokens_for_recipe_name():
    drinks = pd.DataFrame(
        {
            "id": [7, 7],
            "strDrink": ["Cosmopolitan", "Cosmopolitan"],
            "ingredient_name": ["1 1/2 oz vodka", "1 oz cranberry juice"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Cranberry-Juice",
            ],
        }
    )

    tokens = prepare_st_start_tokens("Cosmopolitan", drinks=drinks)

    assert tokens == [
        "<amt>1.5 parts</amt>",
        "<ingred>vodka</ingred>",
        "<amt>1 part</amt>",
        "<ingred>cranberry-juice</ingred>",
    ]


def test_prepare_st_start_tokens_can_truncate_recipe_prompt():
    drinks = pd.DataFrame(
        {
            "id": [7, 7],
            "strDrink": ["Cosmopolitan", "Cosmopolitan"],
            "ingredient_name": ["1 1/2 oz vodka", "1 oz cranberry juice"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Cranberry-Juice",
            ],
        }
    )

    tokens = prepare_st_start_tokens("Cosmopolitan", drinks=drinks, recipe_prefix_len=1)

    assert tokens == [
        "<amt>1.5 parts</amt>",
        "<ingred>vodka</ingred>",
    ]


def test_prepare_st_start_tokens_falls_back_to_ingredient_when_recipe_is_filtered_out():
    drinks = pd.DataFrame(
        {
            "id": [3, 3],
            "strDrink": ["Mystery Drink", "Mystery Drink"],
            "ingredient_name": ["vodka", "lime"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Lime",
            ],
        }
    )

    tokens = prepare_st_start_tokens("Mystery Drink", drinks=drinks)

    assert tokens == ["<amt>1 part</amt>", "<ingred>mystery-drink</ingred>"]


def test_format_prediction_st_from_tokens():
    lines = format_prediction_st(
        tokens=[
            "<amt>1.5 parts</amt>",
            "<ingred>cranberry-juice</ingred>",
            "<amt>1 part</amt>",
            "<ingred>lime</ingred>",
            "<end>",
        ]
    )

    assert lines == ["1.5 parts cranberry juice", "1 part lime"]


def test_format_prediction_st_from_ids_and_inverse_vocab():
    inv_vocab = {
        1: "<amt>2 parts</amt>",
        2: "<ingred>gin</ingred>",
        3: "<end>",
    }

    lines = format_prediction_st(ids=[1, 2, 3], inv_vocab=inv_vocab)

    assert lines == ["2 parts gin"]


def test_format_prediction_mt_from_tokens():
    lines = format_prediction_mt(
        tokens=[
            ("<amt>2 parts</amt>", "<ingred>gin</ingred>"),
            ("<amt>1 part</amt>", "<ingred>tonic-water</ingred>"),
            ("<end>", "<end>"),
        ]
    )

    assert lines == ["2 parts gin", "1 part tonic water"]


def test_format_prediction_mt_from_ids_and_inverse_vocab():
    inv_vocab_amt = {
        1: "<amt>1 part</amt>",
        2: "<end>",
    }
    inv_vocab_ingred = {
        1: "<ingred>vodka</ingred>",
        2: "<end>",
    }

    lines = format_prediction_mt(
        amt_ids=[1, 2],
        ingred_ids=[1, 2],
        inv_vocab_amt=inv_vocab_amt,
        inv_vocab_ingred=inv_vocab_ingred,
    )

    assert lines == ["1 part vodka"]


class DummySingleTaskModel:
    def __init__(self, logits_sequence):
        self.logits_sequence = logits_sequence
        self.calls = []

    def eval(self):
        return self

    def generate_square_subsequent_mask(self, seq_len):
        return torch.zeros((seq_len, seq_len))

    def __call__(self, src, src_key_padding_mask, src_mask):
        self.calls.append(src.squeeze(1).tolist())
        seq_len = src.size(0)
        vocab_size = self.logits_sequence[0].numel()
        output = torch.full((seq_len, 1, vocab_size), float("-inf"))
        output[-1, 0, :] = self.logits_sequence[len(self.calls) - 1]
        return output


class DummyMultiTaskModel:
    def __init__(self, amt_logits_sequence, ingred_logits_sequence):
        self.amt_logits_sequence = amt_logits_sequence
        self.ingred_logits_sequence = ingred_logits_sequence
        self.calls = []

    def eval(self):
        return self

    def generate_square_subsequent_mask(self, seq_len):
        return torch.zeros((seq_len, seq_len))

    def __call__(self, src_amt, src_ingred, src_key_padding_mask, src_mask):
        self.calls.append((src_amt.squeeze(1).tolist(), src_ingred.squeeze(1).tolist()))
        seq_len = src_amt.size(0)
        amt_vocab_size = self.amt_logits_sequence[0].numel()
        ingred_vocab_size = self.ingred_logits_sequence[0].numel()
        output_amt = torch.full((seq_len, 1, amt_vocab_size), float("-inf"))
        output_ingred = torch.full((seq_len, 1, ingred_vocab_size), float("-inf"))
        idx = len(self.calls) - 1
        output_amt[-1, 0, :] = self.amt_logits_sequence[idx]
        output_ingred[-1, 0, :] = self.ingred_logits_sequence[idx]
        return output_amt, output_ingred


def test_predict_st_trims_end_from_start_and_omits_generated_end():
    vocab = {
        "<pad>": 0,
        "<amt>1 part</amt>": 1,
        "<ingred>gin</ingred>": 2,
        "<end>": 3,
    }
    inv_vocab = {token_id: token for token, token_id in vocab.items()}
    model = DummySingleTaskModel(
        logits_sequence=[
            torch.tensor([float("-inf"), float("-inf"), float("-inf"), 0.0]),
        ]
    )

    tokens = predict_st(
        model=model,
        device="cpu",
        pad_id=0,
        vocab=vocab,
        inv_vocab=inv_vocab,
        start_ingred=["<amt>1 part</amt>", "<ingred>gin</ingred>", "<end>"],
        temperature=1.0,
    )

    assert tokens == ["<amt>1 part</amt>", "<ingred>gin</ingred>"]
    assert model.calls == [[1, 2]]


def test_predict_mt_trims_end_from_start_and_omits_generated_end():
    vocab_amt = {
        "<pad>": 0,
        "<amt>1 part</amt>": 1,
        "<end>": 2,
    }
    vocab_ingred = {
        "<pad>": 0,
        "<ingred>gin</ingred>": 1,
        "<end>": 2,
    }
    inv_vocab_amt = {token_id: token for token, token_id in vocab_amt.items()}
    inv_vocab_ingred = {token_id: token for token, token_id in vocab_ingred.items()}
    model = DummyMultiTaskModel(
        amt_logits_sequence=[torch.tensor([float("-inf"), float("-inf"), 0.0])],
        ingred_logits_sequence=[torch.tensor([float("-inf"), float("-inf"), 0.0])],
    )

    tokens = predict_mt(
        model=model,
        device="cpu",
        pad_id_amt=0,
        pad_id_ingred=0,
        vocab_amt=vocab_amt,
        vocab_ingred=vocab_ingred,
        inv_vocab_amt=inv_vocab_amt,
        inv_vocab_ingred=inv_vocab_ingred,
        start_tokens=[("<amt>1 part</amt>", "<ingred>gin</ingred>"), ("<end>", "<end>")],
        temperature=1.0,
    )

    assert tokens == [("<amt>1 part</amt>", "<ingred>gin</ingred>")]
    assert model.calls == [([1], [1])]

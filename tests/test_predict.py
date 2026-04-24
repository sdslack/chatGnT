import pandas as pd
import pytest

from chatGnT.models.predict import prepare_mt_start_tokens


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


def test_prepare_mt_start_tokens_raises_when_recipe_was_filtered_out():
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

    with pytest.raises(ValueError, match="removed by preprocessing"):
        prepare_mt_start_tokens("Mystery Drink", drinks=drinks)


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

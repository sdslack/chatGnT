import math

import numpy as np
import pandas as pd

from chatGnT.data.preprocess import (
    clean_recipes,
    combine_amt_unit,
    convert_to_parts,
    extract_amount_unit,
    extract_ingredient_name,
    filter_recipes,
)


def test_extract_amount_unit_examples():
    assert extract_amount_unit("1 cup sugar") == (1.0, "cup")
    assert extract_amount_unit("1/2 tsp salt") == (0.5, "tsp")
    assert extract_amount_unit("1 1/2 cups milk") == (1.5, "cups")
    assert extract_amount_unit("salt to taste") == (None, None)
    assert extract_amount_unit("5 strawberries") == (5.0, None)
    assert round(extract_amount_unit("1/3 Grand Marnier")[0], 3) == 0.333
    assert extract_amount_unit("1/3 Grand Marnier")[1] is None


def test_extract_ingredient_name_examples():
    assert extract_ingredient_name("/ingredient/-Carrot") == "carrot"
    assert extract_ingredient_name("/ingredient/1-Vodka") == "vodka"


def test_convert_to_parts_converts_supported_units():
    assert convert_to_parts(2.0, "oz") == (2.0, "parts")
    assert convert_to_parts(0.5, "tbsp") == (0.25, "part")
    assert convert_to_parts(30.0, "ml") == (1.0, "part")
    assert convert_to_parts(1.0, "jigger") == (1.5, "parts")


def test_convert_to_parts_leaves_unsupported_units_unchanged():
    assert convert_to_parts(2.0, "dashes") == (2.0, "dashes")
    assert convert_to_parts(1.0, "splash") == (1.0, "splash")


def test_combine_amt_unit_formats_values():
    assert combine_amt_unit(5.0, "oz") == "5 oz"
    assert combine_amt_unit(pd.NA, "dash") == "dash"
    assert combine_amt_unit(2, pd.NA) == "2"
    assert combine_amt_unit(1 / 6, "glass") == "0.167 glass"
    assert combine_amt_unit(np.float64(1 / 6), "glass") == "0.167 glass"


def test_clean_recipes_extracts_and_normalizes_parts():
    df = pd.DataFrame(
        {
            "ingredient_name": ["2 oz vodka", "1 splash soda"],
            "ingredient_link": ["/ingredient/1-Vodka", "/ingredient/2-Soda-Water"],
        }
    )

    cleaned = clean_recipes(df)

    assert cleaned.loc[0, "amt"] == 2.0
    assert cleaned.loc[0, "unit"] == "parts"
    assert cleaned.loc[0, "ingred"] == "vodka"
    assert cleaned.loc[1, "amt"] == 1.0
    assert cleaned.loc[1, "unit"] == "splash"
    assert cleaned.loc[1, "ingred"] == "soda-water"


def test_filter_recipes_assigns_part_to_fractional_unitless_spirit():
    df = pd.DataFrame(
        {
            "id": [1],
            "ingredient_name": ["1/3 Grand Marnier"],
            "ingredient_link": ["/ingredient/1-Grand-Marnier"],
            "amt": [1 / 3],
            "unit": [None],
            "ingred": ["grand-marnier"],
        }
    )

    filtered = filter_recipes(df).reset_index(drop=True)

    assert round(filtered.loc[0, "amt"], 3) == 0.333
    assert filtered.loc[0, "unit"] == "part"
    assert filtered.loc[0, "amt_unit"].startswith("0.333")


def test_filter_recipes_assigns_part_and_add_units():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "ingredient_name": ["rum", "sugar", "bitters"],
            "ingredient_link": [
                "/ingredient/1-Rum",
                "/ingredient/2-Sugar",
                "/ingredient/3-Bitters",
            ],
            "amt": [1.0, math.nan, 2.0],
            "unit": [None, None, None],
            "ingred": ["rum", "sugar", "bitters"],
        }
    )

    filtered = filter_recipes(df).sort_values(["id", "ingred"]).reset_index(drop=True)

    assert len(filtered) == 3
    assert filtered.loc[0, "unit"] == "part"
    assert filtered.loc[0, "amt_unit"] == "1 part"
    assert filtered.loc[1, "unit"] == "add"
    assert filtered.loc[1, "amt_unit"] == "add"
    assert filtered.loc[2, "unit"] == "parts"
    assert filtered.loc[2, "amt_unit"] == "2 parts"


def test_filter_recipes_removes_batch_sized_recipes_over_eight_parts():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "ingredient_name": ["vodka", "lime-juice", "gin", "vermouth"],
            "ingredient_link": [
                "/ingredient/1-Vodka",
                "/ingredient/2-Lime-Juice",
                "/ingredient/3-Gin",
                "/ingredient/4-Vermouth",
            ],
            "amt": [6.0, 2.0, 6.0, 3.0],
            "unit": ["parts", "parts", "parts", "parts"],
            "ingred": ["vodka", "lime-juice", "gin", "vermouth"],
        }
    )

    filtered = filter_recipes(df)

    assert set(filtered["id"]) == {1}
    assert filtered[filtered["id"] == 1]["amt"].sum() == 8.0

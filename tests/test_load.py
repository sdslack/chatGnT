import pandas as pd

from chatGnT.data.load import merge_drinks_into_ingred


def test_merge_drinks_into_ingred_uses_shared_recipe_key():
    ingred = pd.DataFrame(
        {
            "id": [10, 10, 11],
            "ingredient_name": ["1 oz gin", "1 oz tonic water", "2 oz vodka"],
            "ingredient_link": [
                "/ingredient/1-Gin",
                "/ingredient/2-Tonic-Water",
                "/ingredient/3-Vodka",
            ],
        }
    )
    drinks = pd.DataFrame(
        {
            "id": [10, 11],
            "name": ["Gin and Tonic", "Vodka Martini"],
        }
    )

    merged = merge_drinks_into_ingred(ingred, drinks)

    assert list(merged["name"]) == ["Gin and Tonic", "Gin and Tonic", "Vodka Martini"]


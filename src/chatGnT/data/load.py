from chatGnT.config import CFG

def load_ingred():
    from chatGnT.data import utils
    return utils.load_kagglehub_dataset(
        CFG.dataset_id,
        CFG.ingred_path
    )

def load_drinks():
    from chatGnT.data import utils
    return utils.load_kagglehub_dataset(
        CFG.dataset_id,
        CFG.drinks_path
    )

def merge_drinks_into_ingred(ingred, drinks, on=None):
    merge_column = "id"
    drink_columns = [column for column in drinks.columns if column != merge_column]

    return ingred.merge(
        drinks[[merge_column] + drink_columns],
        on=merge_column,
        how="left",
        validate="many_to_one",
    )


def load_all():
    ingred = load_ingred()
    drinks = load_drinks()

    return merge_drinks_into_ingred(ingred, drinks)

from chatGnT.config import CFG
from chatGnT import utils

def load_ingred():
    return utils.load_kagglehub_dataset(
        CFG.dataset_id,
        CFG.ingred_path
    )

def load_drinks():
    return utils.load_kagglehub_dataset(
        CFG.dataset_id,
        CFG.drinks_path
    )

def load_all():
    return {
        "ingred": load_ingred(),
        "drinks": load_drinks(),
    }

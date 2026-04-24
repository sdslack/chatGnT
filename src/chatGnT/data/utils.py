# src/utils.py

#TODO: update python format
#TODO: keep this file generic, reusable, dataset-agnostic

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ----------- Data Loading -----------
 
def load_kagglehub_dataset(dataset_id: str, file_path: str = "", pandas_kwargs=None) -> pd.DataFrame:
    """Load dataset using KaggleHub."""
    pandas_kwargs = pandas_kwargs or {}
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        pandas_kwargs=pandas_kwargs
    )
    return df


def list_kagglehub_files(dataset_id: str):
    """List files inside the Kaggle dataset directory so you can pick your FILE_PATH."""
    path = kagglehub.dataset_download(dataset_id)
    return sorted([p.name for p in path.iterdir()])



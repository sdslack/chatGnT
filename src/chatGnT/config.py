# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Data
    dataset_id: str = "pxxthik/the-cocktail-db-recipe-collection"
    ingred_path: str = "ingredients.csv"
    drinks_path: str = "drinks.csv"

    # Reproducibility
    seed: int = 42

    # Output paths
    project_root: Path = Path(__file__).resolve().parents[2]
    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"

    # EDA controls
    top_n_ingredients: int = 30

def ensure_dirs(cfg: Config) -> None:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

CFG = Config()

"""Data loader and persistence manager for cryptocurrency market telemetry."""

from pathlib import Path
import pandas as pd
from src.core.config import DataConfig
from src.data_pipeline.schema_validator import validate_raw_telemetry


def load_raw_telemetry(path: Path, config: DataConfig) -> pd.DataFrame:
    """Load raw CSV telemetry and validate integrity against the data schema.

    Returns:
        pd.DataFrame: Validated raw telemetry dataframe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw telemetry file not found at: {path}")

    df = pd.read_csv(path)
    validate_raw_telemetry(df, config)
    return df


def save_processed_telemetry(df: pd.DataFrame, path: Path) -> None:
    """Save processed feature-engineered dataset to Parquet format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_processed_telemetry(path: Path) -> pd.DataFrame:
    """Load cached processed telemetry from Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Processed telemetry file not found at: {path}")
    return pd.read_parquet(path)

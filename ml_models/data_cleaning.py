import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing null/NaN values.

    @param df: pandas DataFrame to clean
    @return: Cleaned DataFrame with null values removed
    """
    if df is None or df.empty:
        return df

    initial_rows = len(df)
    df_cleaned = df.dropna()
    removed_rows = initial_rows - len(df_cleaned)

    if removed_rows > 0:
        print(f"Data Cleaning: Removed {removed_rows} rows with null values")
        print(f"Original dataset size: {initial_rows} rows")
        print(f"Cleaned dataset size: {len(df_cleaned)} rows")

    return df_cleaned

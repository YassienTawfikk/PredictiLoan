import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.__00__paths import raw_data_dir


def download_dataset():
    # List of files to check
    raw_dataset = raw_data_dir / "loan_approval_dataset.csv"

    # Check and download
    if raw_dataset.exists():
        print("✔️ Dataset is already downloaded.")
    else:
        # Download dataset
        dataset_path = Path(kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset"))

        if not dataset_path.exists():
            raise FileNotFoundError("⚠ Dataset not found.")

        # Check for an extra "Data" folder
        data_root = dataset_path / "Data" if (dataset_path / "Data").exists() else dataset_path

        # Copy files/folders to raw_data_dir
        for item in data_root.iterdir():
            target = raw_data_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)

        print("✔️ Dataset successfully downloaded.")


def preprocess_dataset(df):
    # Strip Spaces in Feature title
    df.columns = df.columns.str.strip()

    # Drop ID
    df = df.drop(columns=['loan_id'])

    # Label Encoding
    df['education'] = df['education'].map({" Graduate": 1, " Not Graduate": 0})
    df['self_employed'] = df['self_employed'].map({" Yes": 1, " No": 0})
    df['loan_status'] = df['loan_status'].map({" Approved": 1, " Rejected": 0})

    # Normalizing Numerical Features
    numerical_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                          'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                          'bank_asset_value']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


def load_dataset(path):
    return pd.read_csv(path)


def save_dataset(df, path):
    return df.to_csv(path, index=False)


def split_dataset(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=42)

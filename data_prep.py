import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from packaging import version


# Columns
CAT_COLS = ["Available Globally?", "Language Indicator", "Content Type"]
NUM_COLS = ["Hours Viewed", "Release_Year"]

def load_and_clean(input_path):
    df = pd.read_csv(input_path)

    # Remove commas and convert Hours Viewed to int
    df["Hours Viewed"] = df["Hours Viewed"].str.replace(",", "", regex=False).astype("int64")

    # Drop missing/duplicate titles
    df.dropna(subset=["Title"], inplace=True)
    df.drop_duplicates(subset=["Title"], inplace=True)

    # Assign Content_ID
    df["Content_ID"] = df.reset_index().index.astype("int32")
    # Save the full DataFrame (with Content_ID)
    df.to_parquet("./models/metadata.parquet", index=False)

    # Handle Release Date → Release_Year
    release = pd.to_datetime(df["Release Date"], errors="coerce")  # removed infer_datetime_format
    release_year = release.dt.year

    # Fill missing years using regex extraction
    extracted_year = df["Release Date"].astype(str).str.extract(r"(19\d{2}|20\d{2}|21\d{2})").iloc[:, 0]
    release_year = release_year.fillna(extracted_year)

    # Convert to float (safe future behavior)
    release_year = release_year.astype(float, errors="ignore")
    df["Release_Year"] = release_year.infer_objects(copy=False)  # avoids silent downcasting warning

    return df


def build_features(df):
    # Extract numeric year from Release Date
    df["Release Year"] = pd.to_datetime(df["Release Date"], errors="coerce").dt.year

    # Define numeric and categorical columns
    NUM_COLS = ["Hours Viewed", "Release Year"]
    CAT_COLS = ["Available Globally?", "Language Indicator", "Content Type"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS)
        ]
    )

    X = preprocessor.fit_transform(df)
    meta = df[["Title", "Release Date","Content_ID"]]  # keep for later reference
    return X, preprocessor, meta

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from data_prep import load_and_clean, build_features

DEFAULT_INPUT = "./data/netflix_content.csv"
DEFAULT_OUTDIR = "./models"


def main(input_csv: str, outdir: str, n_neighbors: int = 50):
    os.makedirs(outdir, exist_ok=True)

    # 1) Load + basic cleaning
    df = load_and_clean(input_csv)

    # 2) Feature engineering (X) + retain metadata (for display)
    X, transformer, meta = build_features(df)

    # 3) Fit a cosine k-NN index (unsupervised retrieval)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, X.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(X)

    # 4) Persist artifacts
    joblib.dump(nn, os.path.join(outdir, "nn_model.joblib"))
    joblib.dump(transformer, os.path.join(outdir, "transformer.joblib"))
    meta.to_parquet(os.path.join(outdir, "metadata.parquet"), index=False)

    print(f"Saved: {outdir}/nn_model.joblib, transformer.joblib, metadata.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to netflix_content_2023.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Artifacts output directory")
    parser.add_argument("--neighbors", type=int, default=50, help="Max neighbors to store in k-NN index")
    args = parser.parse_args()
    main(args.input, args.outdir, args.neighbors)
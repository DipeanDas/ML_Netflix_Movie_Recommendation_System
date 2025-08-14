import os

# Adjust defaults as needed
DATA_CSV = os.getenv("DATA_CSV", "./netflix_content_2023.csv")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "nn_model.joblib")
TRANSFORMER_PATH = os.path.join(ARTIFACT_DIR, "transformer.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "metadata.parquet")
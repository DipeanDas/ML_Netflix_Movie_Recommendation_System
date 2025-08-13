import joblib
import pandas as pd
import numpy as np
import difflib
from typing import List, Dict

class RecommenderService:
    def __init__(self, model_path: str, transformer_path: str, meta_path: str):
        # Load artifacts
        self.nn = joblib.load(model_path)
        self.transformer = joblib.load(transformer_path)
        self.meta = pd.read_parquet(meta_path)

        # Build convenience maps
        self.title_to_id = {t.lower(): cid for t, cid in zip(self.meta["Title"], self.meta["Content_ID"])}
        self.id_to_row = self.meta.set_index("Content_ID")

    # --- Public API ---
    def suggest_titles(self, query: str, k: int = 10) -> List[str]:
        titles = self.meta["Title"].tolist()
        # quick fuzzy suggestions
        return difflib.get_close_matches(query, titles, n=k, cutoff=0.3)

    def recommend_by_title(self, title: str, k: int = 10) -> List[Dict]:
        cid = self._resolve_title_to_id(title)
        return self.recommend_by_id(int(cid), k=k)

    def recommend_by_id(self, content_id: int, k: int = 10) -> List[Dict]:
        row = self.id_to_row.loc[content_id]
        Xq = self._vectorize_row(row)
        distances, indices = self.nn.kneighbors(Xq, n_neighbors=min(k+1, len(self.meta)))

        # Flatten
        distances, indices = distances[0], indices[0]

        # Map neighbor indices back to Content_ID via the meta index (which aligns with row order)
        # Our NearestNeighbors was fit on transformer.transform(df[features]) in the same row order as meta
        neighbor_rows = self.meta.iloc[indices].copy()

        # Exclude the query item if present as first neighbor (distance 0)
        neighbor_rows = neighbor_rows[neighbor_rows["Content_ID"] != content_id]

        results = []
        for _, r in neighbor_rows.head(k).iterrows():
            results.append({
                "Title": r["Title"],
                "Similarity": float(1.0 - distances[_] if _ < len(distances) else np.nan),
                "Language": r["Language Indicator"],
                "ContentType": r["Content Type"],
                "AvailableGlobally": r["Available Globally?"],
                "ReleaseDate": r["Release Date"],
                "ReleaseYear": int(r["Release_Year"]),
                "HoursViewed": int(r["Hours Viewed"]),
            })
        return results

    # --- Helpers ---
    def _resolve_title_to_id(self, title: str) -> int:
        key = title.lower()
        if key in self.title_to_id:
            return self.title_to_id[key]
        # fuzzy fallback
        close = self.suggest_titles(title, k=1)
        if not close:
            raise ValueError(f"Title '{title}' not found.")
        return self.title_to_id[close[0].lower()]

    def _vectorize_row(self, row) -> np.ndarray:
        # Build a single-row DataFrame with the expected feature columns
        features = {
            "Language Indicator": [row["Language Indicator"]],
            "Content Type": [row["Content Type"]],
            "Available Globally?": [row["Available Globally?"]],
            "Release_Year": [row["Release_Year"]],
            "Log_Hours_Viewed": [np.log1p(row["Hours Viewed"])],
        }
        Xq = self.transformer.transform(pd.DataFrame(features))
        return Xq
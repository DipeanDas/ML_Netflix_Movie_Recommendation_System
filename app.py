from flask import Flask, render_template, request, jsonify
from recommender import RecommenderService
import os

app = Flask(__name__)

# Initialize recommender service once at startup
reco = RecommenderService(
    model_path=os.getenv("MODEL_PATH", "models/nn_model.joblib"),
    transformer_path=os.getenv("TRANSFORMER_PATH", "models/transformer.joblib"),
    meta_path=os.getenv("META_PATH", "models/metadata.parquet")
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/suggest")
def suggest():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    suggestions = reco.suggest_titles(q, k=10)
    return jsonify(suggestions)

@app.route("/api/recommend", methods=["POST"]) 
def recommend():
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    k = int(data.get("k", 10))
    if not title:
        return jsonify({"error": "title is required"}), 400

    try:
        recs = reco.recommend_by_title(title, k=k)
        return jsonify({"query": title, "results": recs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    # For production, consider using: waitress-serve --call app:app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
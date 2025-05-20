from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load models
vectorizer = joblib.load("vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
df = pd.read_csv("products.csv")
df.fillna("", inplace=True)

@app.route('/', methods=["GET"])
def home():
    return (
        "<h1>Welcome to Abolfazl shop</h1>"
    )

@app.route("/products", methods=["GET"])
def get_all_products():
    products = df[["product_id", "name", "category", "price", "rating"]]
    return jsonify(products.to_dict(orient="records"))


@app.route("/categories", methods=["GET"])
def get_categories():
    categories = sorted(df["category"].dropna().unique().tolist())
    return jsonify(categories)


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", default="", type=str)
    category = request.args.get("category", default=None, type=str)
    min_price = request.args.get("min_price", default=0.0, type=float)
    max_price = request.args.get("max_price", default=float("inf"), type=float)
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)

    if not query.strip():
        return jsonify({"error": "Query parameter is required"}), 400

    # Compute cosine similarity
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity"] = similarity_scores

    if category:
        results = results[results["category"].str.lower() == category.lower()]
    results = results[(results["price"] >= min_price) & (results["price"] <= max_price)]

    # Boosted score
    results["boosted_score"] = results["similarity"] * 0.7 + (results["rating"] / 5.0) * 0.3
    results = results.sort_values(by="boosted_score", ascending=False)

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    paginated = results.iloc[start:end]

    # Output
    output = paginated[["product_id", "name", "category", "price", "rating", "similarity", "boosted_score"]]
    return jsonify(output.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)



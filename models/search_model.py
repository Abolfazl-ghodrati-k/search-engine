import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.vectorizer_loader import vectorizer, tfidf_matrix

df = pd.read_csv("data/products.csv").fillna("")

def get_all_products():
    return df[["product_id", "name", "category", "price", "rating"]].to_dict(orient="records")

def get_product_by_id(product_id):
    product = df[df["product_id"] == product_id]
    return None if product.empty else product.iloc[0].to_dict()

def get_categories():
    return sorted(df["category"].dropna().unique().tolist())

def search_products(query, category, min_price, max_price, page, per_page):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity"] = similarity_scores

    if category:
        results = results[results["category"].str.lower() == category.lower()]
    results = results[(results["price"] >= min_price) & (results["price"] <= max_price)]

    results["boosted_score"] = results["similarity"] * 0.7 + (results["rating"] / 5.0) * 0.3
    results = results.sort_values(by="boosted_score", ascending=False)

    start = (page - 1) * per_page
    end = start + per_page
    paginated = results.iloc[start:end]

    return paginated[["product_id", "name", "category", "price", "rating", "similarity", "boosted_score"]].to_dict(orient="records")

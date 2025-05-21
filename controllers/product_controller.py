from flask import Blueprint, request, jsonify
from services.gpt_service import optimize_query_with_cohere
from models.search_model import search_products, get_all_products, get_product_by_id, get_categories

product_blueprint = Blueprint('product', __name__)

@product_blueprint.route('/', methods=["GET"])
def home():
    return "<h1>Welcome to Abolfazl shop</h1>"

@product_blueprint.route("/products", methods=["GET"])
def all_products():
    return jsonify(get_all_products())

@product_blueprint.route("/product/<int:product_id>", methods=["GET"])
def product_by_id(product_id):
    product = get_product_by_id(product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(product)

@product_blueprint.route("/categories", methods=["GET"])
def categories():
    return jsonify(get_categories())

@product_blueprint.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", default="", type=str)
    category = request.args.get("category", default=None, type=str)
    min_price = request.args.get("min_price", default=0.0, type=float)
    max_price = request.args.get("max_price", default=float("inf"), type=float)
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)

    if not query.strip():
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        optimized_query = optimize_query_with_cohere(query)
    except Exception as e:
        return jsonify({"error": f"AI optimization failed: {str(e)}"}), 500

    results = search_products(optimized_query, category, min_price, max_price, page, per_page)

    return jsonify({
        "original_query": query,
        "optimized_query": optimized_query,
        "results": results
    })

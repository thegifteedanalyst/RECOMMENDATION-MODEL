from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="Hybrid Recommendation API")

MODEL_PATH = "recommend_model.pkl"

# ----------------------------
# LOAD MODEL
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

user_item = model["user_item"]
products = model["products"]
vectorizer = model["vectorizer"]
content_matrix = model["content_matrix"]

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_user_index(user_id):
    try:
        return list(user_item.index).index(user_id)
    except ValueError:
        return None

# ----------------------------
# COLLABORATIVE FILTERING
# ----------------------------
def collaborative_recommend(user_id, top_n=5):
    user_index = get_user_index(user_id)
    if user_index is None:
        return []

    user_vector = user_item.iloc[user_index].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, user_item)[0]
    similar_users_idx = similarity_scores.argsort()[::-1][1:6]
    similar_users = user_item.index[similar_users_idx]

    recs = user_item.loc[similar_users].sum().sort_values(ascending=False)
    product_ids = recs.head(top_n).index
    return products[products['product_id'].isin(product_ids)]['product_name'].tolist()

# ----------------------------
# CONTENT-BASED FILTERING
# ----------------------------
def content_recommend(product_name, top_n=5):
    indices = pd.Series(products.index, index=products['product_name']).drop_duplicates()
    if product_name not in indices:
        return []

    idx = indices[product_name]
    sim_scores = cosine_similarity(content_matrix[idx], content_matrix)[0]
    sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return products.iloc[sim_indices]['product_name'].tolist()

# ----------------------------
# HYBRID RECOMMENDATION
# ----------------------------
def hybrid_recommend(user_id, product_name, top_n=5):
    collab = collaborative_recommend(user_id, top_n)
    content = content_recommend(product_name, top_n)
    combined = list(dict.fromkeys(collab + content))  # remove duplicates
    return combined[:top_n]

# ----------------------------
# API ROUTES
# ----------------------------
@app.get("/")
def home():
    return {"message": "🚀 API is running successfully"}

@app.get("/recommend")
def recommend(user_id: int, product_name: str):
    results = hybrid_recommend(user_id, product_name)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return {
        "user_id": user_id,
        "product": product_name,
        "recommendations": results
    }
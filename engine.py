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

    # User similarity
    user_sim = cosine_similarity(user_item.values)[user_index]
    sim_scores = list(enumerate(user_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    similar_users = [user_item.index[i[0]] for i in sim_scores]
    recs = user_item.loc[similar_users]
    top_products = recs.sum().sort_values(ascending=False).head(top_n).index.tolist()
    return top_products

# ----------------------------
# CONTENT-BASED FILTERING
# ----------------------------
def content_recommend(product_name, top_n=5):
    try:
        product_name = product_name.strip().lower()
        prod_indices = pd.Series(products.index, index=products['product_name'].str.lower()).drop_duplicates()

        if product_name not in prod_indices:
            return []

        idx = prod_indices[product_name]
        sim_scores = cosine_similarity(content_matrix[idx], content_matrix)[0]
        sim_indices = sim_scores.argsort()[::-1][1:top_n+1]

        # Safety check for out-of-bounds
        sim_indices = [i for i in sim_indices if i < len(products)]
        return products.iloc[sim_indices]['product_name'].tolist()
    except Exception as e:
        print("Content error:", e)
        return []

# ----------------------------
# HYBRID RECOMMENDATION
# ----------------------------
def hybrid_recommend(user_id, product_name, top_n=5):
    collab = collaborative_recommend(user_id, top_n)
    content = content_recommend(product_name, top_n)
    combined = list(dict.fromkeys(collab + content))
    return combined[:top_n]

# ----------------------------
# API ROUTES
# ----------------------------
@app.get("/")
def home():
    return {"message": "🚀 Hybrid Recommendation API Running"}

@app.get("/recommend")
def recommend(user_id: int, product_name: str):
    results = hybrid_recommend(user_id, product_name)
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return {"user_id": user_id, "product": product_name, "recommendations": results}

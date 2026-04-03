from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import os

app = FastAPI(title="Hybrid Recommendation API")

MODEL_PATH = "recomm_model.pkl"

# ----------------------------
# LOAD MODEL
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = model["df"]
user_item = model["user_item"]
user_similarity = model["user_similarity"]
content_similarity = model["content_similarity"]

# ----------------------------
# HELPER FUNCTION
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

    similarity_scores = list(enumerate(user_similarity[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    similar_users = [user_item.index[i[0]] for i in similarity_scores]

    recs = df[df["customer_id"].isin(similar_users)]

    return recs["product_name"].value_counts().head(top_n).index.tolist()

# ----------------------------
# CONTENT-BASED FILTERING
# ----------------------------
def content_recommend(product_name, top_n=5):
    indices = pd.Series(df.index, index=df["product_name"]).drop_duplicates()

    if product_name not in indices:
        return []

    idx = indices[product_name]

    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    product_indices = [i[0] for i in sim_scores]

    return df["product_name"].iloc[product_indices].tolist()

# ----------------------------
# HYBRID MODEL
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
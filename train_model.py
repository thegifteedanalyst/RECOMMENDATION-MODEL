import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = "advanced_ecommerce_dataset.csv"
MODEL_PATH = "recommend_model.pkl"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().dropna()

# ----------------------------
# REDUCE DATA SIZE (IMPORTANT)
# ----------------------------
df = df[['customer_id', 'product_id', 'product_name', 'category', 'product_description', 'rating']]

# ----------------------------
# USER-ITEM MATRIX (SMALL)
# ----------------------------
user_item = df.pivot_table(
    index='customer_id',
    columns='product_id',
    values='rating'
).fillna(0)

# ----------------------------
# CONTENT FEATURES
# ----------------------------
df['combined_features'] = (
    df['product_name'] + " " +
    df['category'] + " " +
    df['product_description']
)

vectorizer = CountVectorizer(max_features=500)  # 🔥 LIMIT SIZE
content_matrix = vectorizer.fit_transform(df['combined_features'])

# ----------------------------
# KEEP ONLY UNIQUE PRODUCTS
# ----------------------------
products = df[['product_id', 'product_name']].drop_duplicates()

# ----------------------------
# SAVE LIGHT MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)

model_data = {
    "user_item": user_item,
    "products": products,
    "vectorizer": vectorizer,
    "content_matrix": content_matrix
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print("✅ Lightweight model saved (<25MB expected)")
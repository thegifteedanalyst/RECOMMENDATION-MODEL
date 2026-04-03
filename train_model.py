import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

print("🚀 Script started...")

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = "advanced_ecommerce_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.pkl")

print("📂 Current directory:", os.getcwd())

# ----------------------------
# LOAD DATA
# ----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

print("✅ Dataset found")

df = pd.read_csv(DATA_PATH)

# ----------------------------
# CLEAN DATA
# ----------------------------
df = df.drop_duplicates()
df = df.dropna()

print("✅ Data cleaned")

# ----------------------------
# COLLABORATIVE FILTERING
# ----------------------------
user_item = df.pivot_table(
    index="customer_id",
    columns="product_id",
    values="rating"
).fillna(0)

print("✅ User-item matrix created")

user_similarity = cosine_similarity(user_item)

# ----------------------------
# CONTENT-BASED FILTERING
# ----------------------------
df["combined_features"] = (
    df["product_name"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["product_description"].astype(str)
)

vectorizer = CountVectorizer(stop_words="english")
content_matrix = vectorizer.fit_transform(df["combined_features"])

content_similarity = cosine_similarity(content_matrix)

print("✅ Content model built")

# ----------------------------
# SAVE MODEL
# ----------------------------
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("📁 Models folder ready")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "df": df,
            "user_item": user_item,
            "user_similarity": user_similarity,
            "content_similarity": content_similarity
        }, f)

    print("🎉 MODEL SAVED SUCCESSFULLY!")
    print(f"📦 Saved at: {MODEL_PATH}")

except Exception as e:
    print("❌ ERROR SAVING MODEL:", e)
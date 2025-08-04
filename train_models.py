import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import os

# Load CSV (correct name)
df = pd.read_csv("BBC_News_Train.csv")

# Clean the 'tweet' column
df = df.dropna(subset=['tweet'])
text_data = df['tweet'].astype(str)

# Vectorize
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(text_data)

# Train LDA model WITHOUT random_state
lda_model = LatentDirichletAllocation(n_components=5)  # ← No random_state!

# Fit the model
lda_model.fit(X)

# Save to models/
os.makedirs("models", exist_ok=True)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/lda_model.pkl", "wb") as f:
    pickle.dump(lda_model, f)

print("✅ Models saved fresh with compatible format.")

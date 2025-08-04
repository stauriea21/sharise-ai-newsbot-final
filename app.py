from flask import Flask, render_template, request, jsonify
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation

# 🧠 Initialize Flask App
app = Flask(__name__)

# 📁 Load Models from /models folder
MODELS_DIR = "models"

with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'category_model.pkl'), 'rb') as f:
    category_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'lda_vectorizer.pkl'), 'rb') as f:
    lda_vectorizer = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'lda_model.pkl'), 'rb') as f:
    lda_model = pickle.load(f)

# 🏷️ Category Labels
CATEGORY_LABELS = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

# 🧸 Emoji Map
category_emoji_map = {
    'Business': '💼',
    'Entertainment': '🎭',
    'Politics': '🏛️',
    'Sport': '🏅',
    'Tech': '💻'
}

# 🧠 Extract top keywords for each topic from the LDA model
topic_keywords_dict = {}
feature_names = lda_vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[:-6:-1]  # top 5 keywords
    top_keywords = [feature_names[i] for i in top_indices]
    topic_keywords_dict[topic_idx] = top_keywords

# 🌐 Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# 🧠 Predict Category & Topic (Form)
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']

    # Predict Category
    transformed_text = vectorizer.transform([input_text])
    category_pred = category_model.predict(transformed_text)[0]
    confidence_score = max(category_model.predict_proba(transformed_text)[0]) * 100
    category_label = CATEGORY_LABELS[category_pred]
    category_icon = category_emoji_map.get(category_label, '🗞️')

    # Predict Topic
    lda_input = lda_vectorizer.transform([input_text])
    topic_dist = lda_model.transform(lda_input)
    topic_index = topic_dist.argmax()
    topic_score = topic_dist[0][topic_index] * 100
    keywords = topic_keywords_dict.get(topic_index, [])
    keyword_str = ", ".join(keywords)

    return render_template('index.html',
                           prediction="📊 Prediction Result",
                           category=f"{category_icon} {category_label}",
                           confidence=f"{confidence_score:.2f}%",
                           topic=f"{keyword_str} ({topic_score:.2f}%)")

# ⚡ API Endpoint for Real-Time Feedback (JavaScript)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    input_text = request.form['text']

    # Category Prediction
    transformed_text = vectorizer.transform([input_text])
    category_pred = category_model.predict(transformed_text)[0]
    confidence_score = max(category_model.predict_proba(transformed_text)[0]) * 100
    category_label = CATEGORY_LABELS[category_pred]
    category_icon = category_emoji_map.get(category_label, '🗞️')

    # Topic Prediction
    lda_input = lda_vectorizer.transform([input_text])
    topic_dist = lda_model.transform(lda_input)
    topic_index = topic_dist.argmax()
    topic_score = topic_dist[0][topic_index] * 100
    keywords = topic_keywords_dict.get(topic_index, [])
    keyword_str = ", ".join(keywords)

    return jsonify({
        'prediction': "📊 Prediction Result",
        'category': f"{category_icon} {category_label}",
        'confidence': f"{confidence_score:.2f}%",
        'topic': f"{keyword_str} ({topic_score:.2f}%)"
    })

# 🚀 Run Flask App
if __name__ == '__main__':
    app.run(debug=True)

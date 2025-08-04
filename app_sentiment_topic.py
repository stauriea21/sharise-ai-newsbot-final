from flask import Flask, render_template, request
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# ✅ Load trained vectorizer and LDA model
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

# ✅ Topic labels with emojis
TOPIC_LABELS = {
    0: "🌍 World & Politics",
    1: "💼 Business & Economy",
    2: "⚙️ Tech & Innovation",
    3: "🎭 Entertainment & Culture",
    4: "⚽ Sports & Recreation"
}

# ✅ Topic prediction function
def predict_topic(text):
    X_input = vectorizer.transform([text])
    topic_dist = lda_model.transform(X_input)
    topic_index = topic_dist.argmax()
    confidence = topic_dist[0][topic_index]
    return topic_index, confidence

# ✅ Flask app route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    user_input = ""

    if request.method == 'POST':
        user_input = request.form['text']
        topic_id, topic_score = predict_topic(user_input)

        # 🧠 Debug print to console
        print(f"Predicted topic ID: {topic_id}, score: {topic_score}")

        # 🔍 Convert topic number to label with emoji
        prediction = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
        confidence = round(topic_score * 100, 2)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        user_input=user_input
    )

if __name__ == '__main__':
    app.run(debug=True)

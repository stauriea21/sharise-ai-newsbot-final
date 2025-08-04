from flask import Flask, render_template, request
import pickle
import numpy as np

# ✅ Load the Naive Bayes model and TF-IDF vectorizer
with open('models/naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# ✅ Create the Flask app
app = Flask(__name__)

# ✅ Define label mapping
label_map = {
    0: "Business 🏛️",
    1: "Entertainment 🎬",
    2: "Politics 🗳️",
    3: "Sport ⚽",
    4: "Tech 🖥️"
}

# ✅ Home route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news_text']
        vectorized_text = tfidf.transform([input_text])
        prediction = nb_model.predict(vectorized_text)[0]
        prediction_proba = nb_model.predict_proba(vectorized_text)[0]
        confidence = np.max(prediction_proba) * 100

        result = f"Predicted Category: {label_map[prediction]} (Confidence: {confidence:.2f}%)"
        return render_template('index.html', prediction_text=result)

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)

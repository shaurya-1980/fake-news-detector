from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)

CORS(app)

# LOAD FILES
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("LATEST MODEL LOADED")

# CLEANING
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        news = data['text']

        news = clean_text(news)

        vector_input = vectorizer.transform([news])

        prediction = model.predict(vector_input)[0]

        result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

        confidence = 95 if result == "REAL NEWS" else 92

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000)
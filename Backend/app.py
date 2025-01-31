from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "fine_tuned_bert_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Personality labels
PERSONALITY_LABELS = ["Extrovert", "Introvert", "Ambivert"]

# Initialize Firebase
cred = credentials.Certificate("lerny-fc320-firebase-adminsdk-qb8fk-8773f7e764.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Helper function to predict personality
def predict_personality(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return PERSONALITY_LABELS[prediction]

# Default route
@app.route("/")
def home():
    return "Welcome to the Flask API! Use /api/predict_personality or /api/save_conversation for operations."

# Endpoint to predict personality
@app.route('/api/predict_personality', methods=['POST'])
def predict_personality_endpoint():
    try:
        data = request.json
        user_text = data.get("text")
        if not user_text:
            return jsonify({"error": "Input text is required"}), 400
        personality = predict_personality(user_text)
        return jsonify({"personality": personality})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to save user data to Firebase
@app.route('/api/save_conversation', methods=['POST'])
def save_conversation():
    try:
        data = request.json
        user_id = data.get("user_id")
        conversation = data.get("conversation")
        if not user_id or not conversation:
            return jsonify({"error": "user_id and conversation are required"}), 400

        # Save data to Firebase
        db.collection("conversations").document(user_id).set({"conversation": conversation})
        return jsonify({"message": "Data saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)


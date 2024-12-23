from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os
import torch

app = Flask(__name__)

# Global variables for models
sentiment_analysis = None
classification = None
cache_dir = '/tmp/model_cache'  # Use /tmp in Cloud Run

def initialize_models():
    global sentiment_analysis, classification
    
    try:
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize sentiment analysis model
        print("Loading sentiment analysis model...")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "siebert/sentiment-roberta-large-english", 
            cache_dir=cache_dir
        )
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "siebert/sentiment-roberta-large-english", 
            cache_dir=cache_dir
        )
        sentiment_analysis = pipeline(
            "sentiment-analysis", 
            model=sentiment_model, 
            tokenizer=sentiment_tokenizer
        )

        # Initialize classification model
        print("Loading zero-shot classification model...")
        classification = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli", 
            cache_dir=cache_dir
        )
        
        return True
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return False

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Main analysis endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    global sentiment_analysis, classification
    
    if sentiment_analysis is None or classification is None:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500

    try:
        # Get sentences from request
        data = request.get_json()
        sentences = data.get('sentences', [])
        
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        categories = ["Compensation and Benefits", "Work Life Balance", 
                     "Job Security", "Culture", "Career Path"]
        
        results = []
        
        # Process each sentence
        for i, sentence in enumerate(sentences, 1):
            # Sentiment analysis
            sentiment_result = sentiment_analysis(sentence)
            sentiment = sentiment_result[0]["label"]
            sentiment_confidence = sentiment_result[0]["score"] * 100

            # Classification
            classification_result = classification(
                sentence, 
                candidate_labels=categories
            )

            # Store results
            classifications = {
                label: score * 100 
                for label, score in zip(
                    classification_result["labels"], 
                    classification_result["scores"]
                )
            }
            
            result = {
                "id": i,
                "sentence": sentence.strip(),
                "sentiment": {
                    "label": sentiment,
                    "confidence": sentiment_confidence
                },
                "classifications": classifications
            }
            
            results.append(result)

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(_name_)

# Load or train model
MODEL_PATH = "models/csic_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading existing model and vectorizer...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Training new model...")
        # Load data
        df = pd.read_csv("data/csic_ready.csv")
        
        # Features and labels
        X = df["request"]
        y = df["threat"]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = vectorizer.fit_transform(X)
        
        # Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_vec, y)
        
        # Save model and vectorizer
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return model, vectorizer

model, vectorizer = load_or_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'request' not in data:
            return jsonify({"error": "No request provided"}), 400
            
        # Vectorize the input
        X = vectorizer.transform([data['request']])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get top 3 predictions with probabilities
        classes = model.classes_
        top3 = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:3]
        
        return jsonify({
            "prediction": prediction,
            "probabilities": {k: float(v) for k, v in top3}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Read and process the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
            
        # Basic analysis
        if 'threat' not in df.columns:
            return jsonify({"error": "File must contain 'threat' column"}), 400
            
        threat_counts = df['threat'].value_counts().to_dict()
        total_threats = len(df)
        threat_types = len(threat_counts)
        
        # Get timeline data if timestamp column exists
        timeline = {}
        if 'timestamp' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                timeline = df.groupby('date').size().to_dict()
            except:
                pass
                
        return jsonify({
            "total_threats": total_threats,
            "threat_types": threat_types,
            "threat_distribution": threat_counts,
            "timeline": timeline
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)
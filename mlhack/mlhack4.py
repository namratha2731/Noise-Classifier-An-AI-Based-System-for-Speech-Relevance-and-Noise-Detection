import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import joblib
import os

# File paths
dataset_path = './processed_voice_to_text_dataset.csv'
rf_model_path = './rf_model.pkl'
lstm_model_path = './lstm_model.h5'
tfidf_vectorizer_path = './tfidf_vectorizer.pkl'
tokenizer_path = './tokenizer.pkl'

# Training and saving models
def train_and_save_models(dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path)
    data['label'] = data['label'].map({'noise': 0, 'sound': 1})
    
    # Split into features and target
    X = data['processed_text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    joblib.dump(tfidf, tfidf_vectorizer_path)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_tfidf, y_train)
    joblib.dump(rf, rf_model_path)

    # Tokenizer and Padding for LSTM
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    max_len = max(len(seq) for seq in X_train_seq)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    joblib.dump(tokenizer, tokenizer_path)

    # Define and Train LSTM Model
    embedding_dim = 50
    model = Sequential([
        Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_len),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_padded, y_train, epochs=5, batch_size=32)
    model.save(lstm_model_path)

    # Evaluate models
    lstm_preds = (model.predict(X_train_padded) > 0.5).astype("int32").flatten()
    rf_preds = rf.predict(X_train_tfidf)
    print(f"Random Forest Accuracy: {accuracy_score(y_train, rf_preds)}")
    print(f"LSTM Accuracy: {accuracy_score(y_train, lstm_preds)}")

# Train models if files are missing
if not (os.path.exists(rf_model_path) and os.path.exists(lstm_model_path)):
    train_and_save_models(dataset_path)

# Load models and pre-processors
rf_model = joblib.load(rf_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
tokenizer = joblib.load(tokenizer_path)
lstm_model = load_model(lstm_model_path)

# Constants
max_len = 100  # Set to the maximum sequence length from training
weights = [0.5, 0.5]  # Hybrid model weights

# Tkinter GUI
def classify_text():
    input_text = input_field.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    try:
        # Preprocess for Random Forest
        input_tfidf = tfidf_vectorizer.transform([input_text]).toarray()
        rf_pred = rf_model.predict(input_tfidf)

        # Preprocess for LSTM
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')
        lstm_pred = (lstm_model.predict(input_padded) > 0.5).astype("int32").flatten()

        # Combine predictions
        hybrid_pred = (weights[0] * rf_pred + weights[1] * lstm_pred).round().astype("int32")[0]
        result = "Sound" if hybrid_pred == 1 else "Noise"
        result_label.config(text=f"Prediction: {result}", fg="green")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="red")

# Create Tkinter window
root = tk.Tk()
root.title("Noise vs Sound Classification")

# Input Text Field
tk.Label(root, text="Enter text for classification:", font=("Arial", 14)).pack(pady=10)
input_field = tk.Text(root, height=10, width=50, font=("Arial", 12))
input_field.pack(pady=10)

# Predict Button
classify_button = tk.Button(root, text="Classify", font=("Arial", 14), bg="blue", fg="white", command=classify_text)
classify_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()

import sounddevice as sd
import wave
import whisper
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

# Whisper model for transcription
whisper_model = whisper.load_model("base")

# Function to classify an audio file
def classify_audio_file(audio_file_path):
    try:
        # Transcribe the audio file
        print("Transcribing audio...")
        transcription_result = whisper_model.transcribe(audio_file_path)
        input_text = transcription_result['text'].strip()
        print(f"Transcription: {input_text}")
        
        if not input_text:
            print("No text detected in the audio. Classification failed.")
            return

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
        print(f"Classification Result: {result}")
    except Exception as e:
        print(f"Error during classification: {e}")

# Example usage
if _name_ == "_main_":
    audio_file = "path_to_your_audio_file.wav"  # Replace with your audio file path
    classify_audio_file(audio_file)
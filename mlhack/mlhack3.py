import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed dataset
file_path = './processed_voice_to_text_dataset.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Encode target labels
data['label'] = data['label'].map({'noise': 0, 'sound': 1})

# Split data into features and target
X = data['processed_text']
y = data['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text for ML model using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Tokenize and pad text for DL model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = max(len(seq) for seq in X_train_seq)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Random Forest Classifier (ML Component)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", rf_acc)

# LSTM Model (DL Component)
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
model.summary()

# Train LSTM model
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_data=(X_test_padded, y_test))

# Evaluate LSTM model
dl_preds = (model.predict(X_test_padded) > 0.5).astype("int32").flatten()
dl_acc = accuracy_score(y_test, dl_preds)
print("LSTM Model Accuracy:", dl_acc)

# Combine Predictions Using Weighted Averaging
final_preds = (0.5 * rf_preds + 0.5 * dl_preds).round().astype("int32")
final_acc = accuracy_score(y_test, final_preds)
print("Hybrid Model Accuracy:", final_acc)

# Calculate Additional Metrics
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
roc_auc = roc_auc_score(y_test, final_preds)
conf_matrix = confusion_matrix(y_test, final_preds)

print("\nAdditional Evaluation Metrics for Hybrid Model:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
print("Confusion Matrix:\n", conf_matrix)

# Classification report for Hybrid Model
report = classification_report(y_test, final_preds, target_names=['Noise', 'Sound'])
print("Classification Report:\n", report)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, final_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Hybrid Model")
plt.legend(loc="lower right")
plt.show()

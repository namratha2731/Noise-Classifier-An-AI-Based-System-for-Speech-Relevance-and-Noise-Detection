
# Noise Classifier: An AI-Based System for Speech Relevance and Noise Detection

This repository presents an AI-driven system designed to classify speech relevance and detect noise in audio or voice-to-text data. The project tackles the challenge of distinguishing meaningful human speech from background noise or irrelevant audio by employing a combination of Natural Language Processing (NLP) techniques alongside advanced machine learning models, including both traditional methods and deep learning architectures. It aims to improve the accuracy and efficiency of voice-to-text applications, call center analytics, and voice assistant systems by effectively filtering out noise and irrelevant conversational elements.

---

## ✨ Features

- **Speech Relevance Classification:** Classifies voice-to-text data as either relevant speech or noise, improving downstream applications’ quality.
- **Dual Model Approach:**  
  - *Deep Learning (LSTM):* Uses a Long Short-Term Memory neural network (`lstm_model.h5`) well-suited for sequential text data.  
  - *Traditional ML (Random Forest):* Employs a Random Forest (`rf_model.pkl`), a powerful ensemble learning algorithm.
- **Natural Language Processing (NLP):**  
  - *Tokenization:* Implements a pre-trained tokenizer (`tokenizer.pkl`) for splitting text into tokens.  
  - *TF-IDF Vectorization:* Utilizes a pre-fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`) to convert text into numerical features.
- **Comprehensive Data Handling:** Includes raw and processed voice-to-text datasets for training and validation (`voice_to_text_dataset_10000_non_redundant.csv`, `processed_voice_to_text_dataset.csv`, `analyzed_dataset.csv`).
- **Pre-trained Models:** Provides models and vectorizers pre-trained for immediate inference without retraining.

---

## 🧠 How It Works

1. **Data Collection:** Voice-to-text transcripts are collected and simulated via CSV datasets.
2. **Preprocessing:** Raw text undergoes cleaning, tokenization, and transformation into TF-IDF vectors compatible with machine learning models.
3. **Model Training:** Both LSTM and Random Forest models are trained on processed data to learn to distinguish relevant speech from noise.
4. **Classification/Prediction:** Incoming voice-to-text inputs are preprocessed and fed into the trained models to predict relevance or noise presence.
5. **Output:** The system outputs a classification label indicating relevant speech or noise, enabling filtering or further processing.

---

## 💻 Technologies Used

- **Python:** Core programming language.
- **TensorFlow/Keras:** Framework for building and running the LSTM deep learning model.
- **Scikit-learn:** For Random Forest implementation and preprocessing utilities.
- **Pandas:** Data loading and manipulation.
- **NumPy:** Numerical computations.
- **Joblib:** For saving/loading models and vectorizers.

---

## 📁 Project Structure

```
noise-classifier-an-ai-based-system-for-speech-relevance-and-noise-detection/
├── .gitignore
└── mlhack/
    ├── analyzed_dataset.csv
    ├── CaseStudy_Template.pptx
    ├── lstm_model.h5
    ├── mlhack1.py
    ├── mlhack2.py
    ├── mlhack21.py
    ├── mlhack3.py
    ├── mlhack4.py
    ├── mlhackfinal.py         # Main classification pipeline
    ├── mlhacktrans.py
    ├── mlhacktrans2.py
    ├── processed_voice_to_text_dataset.csv
    ├── rf_model.pkl
    ├── solution for hackathon.docx
    ├── tfidf_vectorizer.pkl
    ├── tokenizer.pkl
    └── voice_to_text_dataset_10000_non_redundant.csv
```

---

## ⚙️ Setup and Installation

1. **Clone the Repository:**
   ```
   git clone 
   cd noise-classifier-an-ai-based-system-for-speech-relevance-and-noise-detection/mlhack
   ```

2. **Install Dependencies:**
   ```
   pip install pandas numpy scikit-learn tensorflow joblib
   ```

---

## ▶️ Usage

- Run the main classification script:
  ```
  python mlhackfinal.py
  ```
- Ensure all required models (`lstm_model.h5`, `rf_model.pkl`), vectorizers (`tfidf_vectorizer.pkl`), tokenizer (`tokenizer.pkl`), and datasets are present in the `mlhack` folder.

- Additional scripts (`mlhack1.py`, `mlhack2.py`, etc.) may contain preprocessing or model training routines; refer to inline comments for specific instructions.

---

## 💡 Conclusion

The **Noise Classifier: An AI-Based System for Speech Relevance and Noise Detection** project embodies a comprehensive application of machine learning and natural language processing to a real-world challenge. By integrating both deep learning and traditional ML approaches, it delivers a robust solution for enhancing the quality of voice-to-text data through effective noise filtering. This system highlights the significance of meticulous data preprocessing and appropriate model selection in achieving high-performance classification, establishing a solid foundation for future developments in voice analytics, conversational AI, and audio filtering technologies.


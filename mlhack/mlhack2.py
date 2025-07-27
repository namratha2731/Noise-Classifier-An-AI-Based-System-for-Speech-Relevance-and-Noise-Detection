import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = "./voice_to_text_dataset_10000_non_redundant.csv"
data = pd.read_csv(file_path)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function for text preprocessing
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join the words back into a single string
    return ' '.join(words)

# Apply preprocessing to the text column
data['processed_text'] = data['text'].apply(preprocess_text)

# Display the first few rows of the processed dataset
print(data.head())

# Save the preprocessed data to a new file
output_file = './processed_voice_to_text_dataset.csv'
data.to_csv(output_file, index=False)
print(f"Preprocessed dataset saved to {output_file}")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')

# Load the preprocessed dataset
file_path = './processed_voice_to_text_dataset.csv'
data = pd.read_csv(file_path)

# Functions for analyses and visualizations

# Bag of Words Analysis
def bag_of_words_analysis(text_series, top_n=20):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(text_series)
    word_counts = bow_matrix.sum(axis=0).A1
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
    return Counter(word_freq).most_common(top_n)

# N-Gram Analysis
def get_ngrams(text_series, n=2, top_n=20):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngram_matrix = vectorizer.fit_transform(text_series)
    ngram_counts = ngram_matrix.sum(axis=0).A1
    ngram_freq = dict(zip(vectorizer.get_feature_names_out(), ngram_counts))
    return Counter(ngram_freq).most_common(top_n)

# Generate Word Cloud
def generate_wordcloud(text_series):
    text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud", fontsize=20)
    plt.show()

# Plot Word Frequency
def plot_word_frequency(words, title):
    labels, values = zip(*words)
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# Sentiment Analysis
def calculate_sentiment(text_series):
    sentiments = text_series.apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    plt.hist(sentiments, bins=20, color='purple', alpha=0.7)
    plt.title('Sentiment Polarity Distribution')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.show()
    return sentiments

# TF-IDF Analysis
def tfidf_analysis(text_series, top_n=20):
    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    words = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    return dict(zip(words, scores))

# POS Tagging
def pos_tag_analysis(text_series):
    tagged = text_series.apply(lambda x: nltk.pos_tag(x.split()))
    return tagged

# Word Length Analysis
def average_word_length(text_series):
    return text_series.apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))

# Text Length Analysis
def text_length_analysis(text_series):
    lengths = text_series.apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, color='orange', alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
    return lengths

# Perform Analyses
print("Performing analyses and generating graphs...")

# Bag of Words
top_words = bag_of_words_analysis(data['processed_text'])
plot_word_frequency(top_words, "Top 20 Words (Bag of Words)")

# Bigrams and Trigrams
top_bigrams = get_ngrams(data['processed_text'], n=2)
plot_word_frequency(top_bigrams, "Top 20 Bigrams")

top_trigrams = get_ngrams(data['processed_text'], n=3)
plot_word_frequency(top_trigrams, "Top 20 Trigrams")

# Word Cloud
generate_wordcloud(data['processed_text'])

# Sentiment Analysis
data['sentiment'] = calculate_sentiment(data['processed_text'])

# TF-IDF Analysis
tfidf_scores = tfidf_analysis(data['processed_text'])
plot_word_frequency(tfidf_scores.items(), "Top Words by TF-IDF")

# Text Statistics
data['avg_word_length'] = average_word_length(data['processed_text'])
data['text_length'] = text_length_analysis(data['processed_text'])

# POS Tagging
data['pos_tags'] = pos_tag_analysis(data['processed_text'])

# Save the dataset with new metrics
data.to_csv('./analyzed_dataset.csv', index=False)
print("Analysis complete. Results saved to './analyzed_dataset.csv'")

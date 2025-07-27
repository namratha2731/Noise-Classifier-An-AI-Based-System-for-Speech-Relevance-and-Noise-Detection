import random
import pandas as pd

# Technical terms for sound
sound_technical_terms = [
    "machine learning", "quantum computing", "data structures", "cloud computing",
    "natural language processing", "cybersecurity", "blockchain technology", "big data analytics",
    "neural networks", "computer vision", "artificial intelligence", "linear regression",
    "decision trees", "support vector machines", "reinforcement learning", "genetic algorithms",
    "gradient descent", "optimization techniques", "IoT devices", "robotics", "digital signal processing",
    "microcontrollers", "databases", "software engineering", "agile methodologies", "operating systems",
    "network protocols", "encryption algorithms", "wireless communication", "3D printing",
    "virtual reality", "augmented reality", "deep learning", "semantic analysis", "topic modeling",
    "cloud storage", "edge computing", "sensors and actuators", "bioinformatics", "data preprocessing",
    "time complexity analysis", "space complexity", "algorithm design", "finite state machines",
    "distributed systems", "parallel processing", "multi-threading", "serverless computing"
]

# Non-technical terms for noise
noise_non_technical_phrases = [
    "background chatter", "inaudible murmurs", "mic feedback", "random laughter",
    "coughing and sneezing", "static noises", "chair scraping", "footsteps", 
    "projector humming", "pen tapping", "door slamming", "outside honking", 
    "people talking", "late arrivals", "announcements", "ambient noise", 
    "random whispers", "unintelligible murmurs", "phone ringing", 
    "air conditioner noise", "echoing sound"
]

# Sound templates (technical speech)
sound_templates = [
    "In today's lecture, we will discuss {} in detail.",
    "The concept of {} is essential for understanding {}.",
    "Can anyone explain the principle of {}?",
    "Let's calculate the {} using the formula provided.",
    "The application of {} in {} is critical for this field.",
    "Next, we'll explore how {} works in {} scenarios.",
    "Real-world examples of {} include {}.",
    "Remember, {} is a key concept in {}.",
    "For {} to work, we need to understand {} first.",
    "The relationship between {} and {} is fundamental."
]

# Noise templates (distractions)
noise_templates = [
    "Uh, like, there was some {} in the background.",
    "Can someone mute the {}? It's really loud.",
    "Random noises like {} made it hard to focus.",
    "The {} was too distracting during the session.",
    "Background sounds of {} disrupted the lecture.",
    "Someone was {}—it was impossible to hear anything.",
    "The echo from {} made the voice unclear.",
    "Ambient noises like {} were very distracting.",
    "I couldn't hear the lecture because of the {}.",
    "Unintelligible murmurs about {} kept interfering."
]

# Function to generate a row
def generate_text_row(template_list, placeholder_list, category):
    """Generate a single row of text using a random template and placeholder."""
    template = random.choice(template_list)
    placeholders = random.sample(placeholder_list, template.count("{}"))
    text = template.format(*placeholders)
    return {"text": text, "label": category}

# Create dataset
rows = []
num_rows_per_category = 5000  # Total 10,000 rows (5,000 each)

# Generate sound rows
for _ in range(num_rows_per_category):
    rows.append(generate_text_row(sound_templates, sound_technical_terms, "sound"))

# Generate noise rows
for _ in range(num_rows_per_category):
    rows.append(generate_text_row(noise_templates, noise_non_technical_phrases, "noise"))

# Shuffle the dataset
random.shuffle(rows)

# Save to CSV
dataset = pd.DataFrame(rows)
dataset.to_csv("voice_to_text_dataset_10000_non_redundant.csv", index=False)
print("Dataset saved as 'voice_to_text_dataset_10000_non_redundant.csv'")

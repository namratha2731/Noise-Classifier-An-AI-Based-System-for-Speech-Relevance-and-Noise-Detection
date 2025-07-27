import whisper

# Load the Whisper model (you can choose 'tiny', 'base', 'small', 'medium', or 'large')
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("your_audio_file.wav")  # Replace with the path to your audio file

print("Transcription:", result['text'])
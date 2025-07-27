
import sounddevice as sd
import wave
import whisper

# Function to record audio
def record_audio(filename, duration, samplerate=16000):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    
    # Save the audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

# File and recording parameters
output_file = "recorded_audio.wav"
record_duration = 10  # Duration of recording in seconds

# Record audio from the microphone
record_audio(output_file, record_duration)

# Load the Whisper model (you can choose 'tiny', 'base', 'small', 'medium', or 'large')
model = whisper.load_model("base")
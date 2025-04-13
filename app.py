from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import speech_recognition as sr
import librosa
import numpy as np
import re
from mistralai import Mistral
from langchain.prompts import PromptTemplate
import concurrent.futures


app = Flask(__name__)



# Initialize Mistral
os.environ["MISTRAL_API_KEY"] = "tw7BGvT5iE0duZIryKL0W18GotNRKVlB"
llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        word_count = len(text.split())
        print(f"Word count: {word_count}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        text = ""
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
        text = ""
    return text

def analyze_audio(audio_file, transcript):
    y, sr = librosa.load(audio_file)
    rms = librosa.feature.rms(y=y).mean()
    volume = "Too Loud" if rms > 0.6 else "Too Soft" if rms <= 0.009 else "Moderate and Balanced"
    words = len(transcript.split())
    duration = librosa.get_duration(y=y, sr=sr)
    wpm = words / (duration / 60)
    pause_threshold = 0.5
    intervals = librosa.effects.split(y, top_db=30)
    pause_durations = [(intervals[i][0] - intervals[i - 1][1]) / sr for i in range(1, len(intervals))]
    long_pauses = sum(1 for p in pause_durations if p > pause_threshold)
    pauses = "Frequent Long Pauses" if long_pauses > 2 else "Appropriate Pauses"
    filler_words = len(re.findall(r'\b(uh|um|like|you know)\b', transcript))
    clarity = "Clear and Fluent" if filler_words == 0 else "Needs Improvement in Fluency"
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch_variance = np.var(pitches) if len(pitches) > 0 else 0
    pitch_range = np.max(pitches) - np.min(pitches) if len(pitches) > 0 else 0
    pitch_description = "Dynamic Pitch Range" if pitch_range > 100 else "Limited Pitch Range"
    intonation_slope = (pitches[-1] - pitches[0]) / len(pitches) if len(pitches) > 1 else 0
    intonation = "Engaging Intonation" if intonation_slope > 0.3 else "Flat Intonation and monotone"
    return {
        "volume": volume,
        "wpm": round(wpm, 2),
        "pauses": pauses,
        "clarity": clarity,
        "pitch": pitch_description,
        "intonation": intonation,
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    tmp_folder = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(tmp_folder, filename)
    file.save(file_path)

    transcript = transcribe_audio(file_path)
    metrics = analyze_audio(file_path, transcript)

    summary_prompt = PromptTemplate(
        input_variables=["transcript", "metrics"],
        template=(
            "You are a supportive AI speech coach to help introverts. Based on the transcript:\n{transcript}\n"
            "and these detailed metrics:\n{metrics}\n"
            "Evaluate the speaker's confidence and performance in a friendly and encouraging manner, making them aware of their strengths and areas for improvement with tips on how to improve each aspect. Use the following format:\n\n"
            "1. **Overall Confidence Analysis**:\n"
            "- Provide an overall score out of 100, with a positive, friendly summary of the speaker's performance. Include encouragement and reassurance.\n\n"
            "2. **Detailed Metrics Evaluation**:\n"
            "- **Volume**\n"
            "- **Rate of Speech (WPM)** (normal is between 120–150)\n"
            "- **Pauses**\n"
            "- **Clarity**\n"
            "- **Pitch**\n"
            "- **Intonation**\n"
        )
    )

    formatted_prompt = summary_prompt.format(transcript=transcript, metrics=metrics)

    def call_mistral():
        return llm.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": formatted_prompt}]
        )

    # Threaded timeout wrapper
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_mistral)
            response = future.result(timeout=30)  # Timeout in seconds

        feedback = response.choices[0].message.content

        return jsonify({
            "transcript": transcript,
            "metrics": metrics,
            "feedback": feedback
        })

    except concurrent.futures.TimeoutError:
        return jsonify({
            "error": "Mistral API call timed out after 30 seconds. Please try again later."
        }), 504

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

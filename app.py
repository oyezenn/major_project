from flask import Flask, render_template, request
import pickle
import numpy as np
import librosa
import os

app = Flask(__name__)

# Load models
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
audio_model = pickle.load(open("audio_model.pkl", "rb"))

# ── Module 2: Text Analysis ──────────────────────────────────────────
def predict_text(text):
    vector = vectorizer.transform([text]).toarray()
    prob = model.predict_proba(vector)[0][1]
    critical_words = ["kill", "suicide", "die", "end my life"]
    if any(word in text.lower() for word in critical_words):
        return "High Risk (Critical Keywords Detected)"
    if prob > 0.7:
        return f"High Risk ({prob:.2f})"
    elif prob > 0.4:
        return f"Moderate Risk ({prob:.2f})"
    else:
        return f"Low Risk ({prob:.2f})"

# ── Module 3: Audio Analysis ─────────────────────────────────────────
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_audio(file_path):
    features = extract_features(file_path)
    prediction = audio_model.predict([features])[0]
    labels = ["Minimal Risk", "Mild Risk", "Moderate Risk", "Severe Risk"]
    return labels[prediction]

# ── Module 4: Questionnaire Scoring ──────────────────────────────────
def score_phq9(responses):
    total = sum(responses)
    if total <= 4:    return "Minimal Depression", total
    elif total <= 9:  return "Mild Depression", total
    elif total <= 14: return "Moderate Depression", total
    elif total <= 19: return "Moderately Severe Depression", total
    else:             return "Severe Depression", total

def score_gad7(responses):
    total = sum(responses)
    if total <= 4:    return "Minimal Anxiety", total
    elif total <= 9:  return "Mild Anxiety", total
    elif total <= 14: return "Moderate Anxiety", total
    else:             return "Severe Anxiety", total

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        user_input = request.form["text"]
        result = predict_text(user_input)
    return render_template("index.html", result=result)

@app.route("/audio", methods=["GET", "POST"])
def audio():
    result = ""
    error = ""
    if request.method == "POST":
        if "audio" not in request.files or request.files["audio"].filename == "":
            error = "Please upload a WAV audio file."
        else:
            audio_file = request.files["audio"]
            save_path = "temp_audio.wav"
            audio_file.save(save_path)
            try:
                result = predict_audio(save_path)
            except Exception as e:
                error = f"Error processing audio: {str(e)}"
            finally:
                if os.path.exists(save_path):
                    os.remove(save_path)
    return render_template("audio.html", result=result, error=error)

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    phq_result = gad_result = phq_score = gad_score = None
    if request.method == "POST":
        phq = [int(request.form.get(f"phq_{i}", 0)) for i in range(1, 10)]
        gad = [int(request.form.get(f"gad_{i}", 0)) for i in range(1, 8)]
        phq_result, phq_score = score_phq9(phq)
        gad_result, gad_score = score_gad7(gad)
    return render_template("questionnaire.html",
                           phq_result=phq_result, phq_score=phq_score,
                           gad_result=gad_result, gad_score=gad_score)

if __name__ == "__main__":
    app.run(debug=True)
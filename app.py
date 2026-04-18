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
    if not text or len(text.strip()) < 5:
        return "error: Please enter at least one sentence."
    if len(text.strip()) > 5000:
        return "error: Text too long. Please keep it under 5000 characters."

    vector = vectorizer.transform([text]).toarray()
    prob = model.predict_proba(vector)[0][1]

    critical_words = ["kill", "suicide", "die", "end my life", "hurt myself"]
    if any(word in text.lower() for word in critical_words):
        return "High Risk (Critical Keywords Detected)"

    if prob > 0.7:
        return f"High Risk ({prob:.2f})"
    elif prob > 0.4:
        return f"Moderate Risk ({prob:.2f})"
    else:
        return f"Low Risk ({prob:.2f})"

# ── Module 3: Audio Analysis ─────────────────────────────────────────
ALLOWED_EXTENSIONS = {"wav"}
MAX_AUDIO_SIZE_MB = 10

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    if len(y) < sr * 1:  # less than 1 second
        raise ValueError("Audio too short. Please record at least 1 second.")
    if len(y) > sr * 120:  # more than 2 minutes
        raise ValueError("Audio too long. Please keep it under 2 minutes.")
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
    error = ""
    if request.method == "POST":
        user_input = request.form.get("text", "").strip()
        if not user_input:
            error = "Please enter some text before submitting."
        else:
            result = predict_text(user_input)
            if result.startswith("error:"):
                error = result.replace("error: ", "")
                result = ""
    return render_template("index.html", result=result, error=error)

@app.route("/audio", methods=["GET", "POST"])
def audio():
    result = ""
    error = ""
    if request.method == "POST":
        if "audio" not in request.files:
            error = "No file received. Please upload a WAV file."
        else:
            audio_file = request.files["audio"]
            if audio_file.filename == "":
                error = "No file selected. Please choose a WAV file."
            elif not allowed_file(audio_file.filename):
                error = "Invalid file type. Only .wav files are supported."
            else:
                # Check file size
                audio_file.seek(0, 2)
                size_mb = audio_file.tell() / (1024 * 1024)
                audio_file.seek(0)
                if size_mb > MAX_AUDIO_SIZE_MB:
                    error = f"File too large ({size_mb:.1f} MB). Maximum size is {MAX_AUDIO_SIZE_MB} MB."
                else:
                    save_path = "temp_audio.wav"
                    audio_file.save(save_path)
                    try:
                        result = predict_audio(save_path)
                    except ValueError as ve:
                        error = str(ve)
                    except Exception as e:
                        error = f"Could not process audio. Make sure it is a valid WAV recording."
                    finally:
                        if os.path.exists(save_path):
                            os.remove(save_path)
    return render_template("audio.html", result=result, error=error)

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    phq_result = gad_result = phq_score = gad_score = None
    error = ""
    if request.method == "POST":
        try:
            phq = [int(request.form.get(f"phq_{i}", -1)) for i in range(1, 10)]
            gad = [int(request.form.get(f"gad_{i}", -1)) for i in range(1, 8)]

            if -1 in phq or -1 in gad:
                error = "Please answer all questions before submitting."
            elif not all(0 <= v <= 3 for v in phq + gad):
                error = "Invalid response values detected."
            else:
                phq_result, phq_score = score_phq9(phq)
                gad_result, gad_score = score_gad7(gad)
        except (ValueError, TypeError):
            error = "Something went wrong. Please try again."

    return render_template("questionnaire.html",
                           phq_result=phq_result, phq_score=phq_score,
                           gad_result=gad_result, gad_score=gad_score,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
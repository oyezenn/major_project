from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved text model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

# ── Module 4: Questionnaire Scoring ──────────────────────────────────
def score_phq9(responses):
    total = sum(responses)
    if total <= 4:   return "Minimal Depression", total
    elif total <= 9:  return "Mild Depression", total
    elif total <= 14: return "Moderate Depression", total
    elif total <= 19: return "Moderately Severe Depression", total
    else:             return "Severe Depression", total

def score_gad7(responses):
    total = sum(responses)
    if total <= 4:   return "Minimal Anxiety", total
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
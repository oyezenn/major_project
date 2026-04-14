from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    vector = vectorizer.transform([text]).toarray()
    prob = model.predict_proba(vector)[0][1]

    critical_words = ["kill", "suicide", "die", "end my life"]
    text_lower = text.lower()

    if any(word in text_lower for word in critical_words):
        return "High Risk (Critical Keywords Detected)"

    if prob > 0.7:
        return f"High Risk ({prob:.2f})"
    elif prob > 0.4:
        return f"Moderate Risk ({prob:.2f})"
    else:
        return f"Low Risk ({prob:.2f})"

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        user_input = request.form["text"]
        result = predict(user_input)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
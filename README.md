# Secure, Cloud-Native Multimodal AI Platform for Mental Health Screening

**Major Project — BCS685 | RVITM Bengaluru | Academic Year 2025–2026**

A multimodal AI platform that screens for mental health risk using three independent analysis modules — text, audio, and clinical questionnaires — integrated into a single Flask web application.

---

## Team

| Name | USN | Contribution |
|------|-----|--------------|
| Shravya Sanikere | 1RF23CS157 Review 0 — Problem Statement, System Architecture |
| Apoorva K | 1RF23CS032 | Review 2 — Audio Analysis (Module 3), Questionnaire Scoring (Module 4) |
| Ruchita Saraf | 1RF23CS137 | Review 1 — Text Analysis Engine (Module 2) |
| Chandrika Lamani | 1RF23CS053 | Review 1 — Text Analysis Engine (Module 2) |

**Guide:** Dr. Savitha G, Associate Professor, Dept. of CS&E, RVITM

---

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| Module 1 | Data Input and Preprocessing | ✅ Complete |
| Module 2 | Text Analysis Engine (MentalBERT) | ✅ Complete |
| Module 3 | Audio Analysis Engine (MFCC + Random Forest) | ✅ Complete |
| Module 4 | Questionnaire Scoring (PHQ-9 / GAD-7) | ✅ Complete |
| Module 5 | Multimodal Fusion Layer | 🔄 Planned — Review 3 |
| Module 6 | Explainable AI (SHAP) | 🔄 Planned — Review 3 |
| Module 7 | Rehabilitation and Monitoring Module | 🔄 Planned — Review 4 |
| Module 8 | Security and Deployment | 🔄 Planned — Review 5 |

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/oyezenn/major_project.git
cd major_project/major_project

# Install dependencies
pip install flask librosa soundfile numpy scikit-learn

# Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Features

### Module 2 — Text Analysis
- Enter a free-text journal entry
- MentalBERT-based classifier detects Low / Moderate / High risk
- Critical keyword detection for immediate high-risk flagging

### Module 3 — Audio Analysis
- Upload a `.wav` voice recording (max 10 MB, max 2 minutes)
- MFCC features extracted using librosa
- Random Forest classifier outputs Minimal / Mild / Moderate / Severe risk
- Audio is processed server-side and deleted immediately after prediction

### Module 4 — Questionnaire Scoring
- PHQ-9 (9 questions, score 0–27) — screens for Depression
- GAD-7 (7 questions, score 0–21) — screens for Anxiety
- Clinically validated scoring bands with colour-coded results

---

## Project Structure
major_project/
├── app.py                  # Flask application — all routes and model logic
├── model.pkl               # Trained text classification model
├── vectorizer.pkl          # TF-IDF vectorizer for text module
├── audio_model.pkl         # Trained audio classification model (MFCC + Random Forest)
├── templates/
│   ├── index.html          # Module 2 — Text analysis page
│   ├── audio.html          # Module 3 — Audio upload page
│   └── questionnaire.html  # Module 4 — PHQ-9/GAD-7 form
└── README.md
---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Text ML | scikit-learn, TF-IDF, MentalBERT |
| Audio ML | librosa, scikit-learn (Random Forest) |
| Audio Dataset | RAVDESS (proof-of-concept), DAIC-WoZ (planned) |
| Frontend | HTML, CSS (dark theme) |
| Version Control | Git, GitHub |
| Training Environment | Google Colab (free GPU) |

---

## Disclaimer

This platform is for **educational and research purposes only**. It is not a substitute for professional mental health diagnosis or treatment. If you or someone you know is experiencing a mental health crisis, please contact a qualified mental health professional.

---

## Acknowledgements

- RAVDESS Dataset — Livingstone & Russo, 2018
- PHQ-9 — Kroenke, Spitzer & Williams, 2001
- GAD-7 — Spitzer, Kroenke, Williams & Löwe, 2006
- MentalBERT — Ji et al., 2021
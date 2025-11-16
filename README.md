# Speech Impact Analyzer

A Natural Language Processing project that analyzes political speeches and tweets to understand how rhetorical style (logos, pathos, ethos) impacts audience engagement.

## 🧠 Overview
- **Goal:** Predict audience engagement (High/Medium/Low)
- **Data:** EMPOLITICON dataset + political tweets
- **Model:** TF-IDF + Rhetorical Features + Logistic Regression
- **Web App:** Flask-based interface to test new texts

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
cd code
python train_model.py
python app.py

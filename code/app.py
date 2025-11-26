from flask import Flask, render_template, request
import pickle
import numpy as np
from preprocess import clean_text
from features import rhetorical_features

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
        if request.method == "POST":
        user_text = request.form["user_input"]

        clean = clean_text(user_text)
        rhet = rhetorical_features(clean).reshape(1, -1)
        tfidf = vectorizer.transform([clean]).toarray()
        combined = np.hstack([tfidf, rhet])
        pred = model.predict(combined)[0]

        # --- NEW: compute nicer scores for display ---
        raw_logos  = float(rhet[0][0])
        raw_pathos = float(rhet[0][1])
        raw_ethos  = float(rhet[0][2])

        total = raw_logos + raw_pathos + raw_ethos

        if total > 0:
            logos_score  = (raw_logos  / total) * 100
            pathos_score = (raw_pathos / total) * 100
            ethos_score  = (raw_ethos  / total) * 100
        else:
            logos_score = pathos_score = ethos_score = 0.0

        return render_template(
            "index.html",
            user_text=user_text,
            pred=pred,
            logos=logos_score,
            pathos=pathos_score,
            ethos=ethos_score,
        )

    # GET request
    return render_template(
        "index.html",
        user_text="",
        pred=None,
        logos=None,
        pathos=None,
        ethos=None,
    )


if __name__ == "__main__":
    app.run(debug=True)

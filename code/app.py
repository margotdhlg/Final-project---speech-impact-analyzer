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

        # --- preprocessing ---
        clean = clean_text(user_text)
        rhet = rhetorical_features(clean).reshape(1, -1)
        tfidf = vectorizer.transform([clean]).toarray()
        combined = np.hstack([tfidf, rhet])
        pred = model.predict(combined)[0]

        # --- RAW scores coming from the feature extractor ---
        logos_raw = float(rhet[0][0])
        pathos_raw = float(rhet[0][1])
        ethos_raw = float(rhet[0][2])

        print("RAW SCORES:", logos_raw, pathos_raw, ethos_raw)

        # --- Display scaling (only affects UI, not model) ---
        SCALE_LOGOS = 100.0
        SCALE_PATHOS = 200.0   
        SCALE_ETHOS = 100.0

        logos_display = logos_raw * SCALE_LOGOS
        pathos_display = pathos_raw * SCALE_PATHOS
        ethos_display = ethos_raw * SCALE_ETHOS

        print("DISPLAY SCORES:", logos_display, pathos_display, ethos_display)

        return render_template(
            "index.html",
            user_text=user_text,
            pred=pred,
            logos=logos_display,
            pathos=pathos_display,
            ethos=ethos_display,
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

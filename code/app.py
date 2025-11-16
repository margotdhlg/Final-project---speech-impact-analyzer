from flask import Flask, render_template, request
import pickle
import numpy as np
from preprocess import clean_text
from features import rhetorical_features

app = Flask(__name__)

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
        
        return render_template("index.html", 
                               user_text=user_text,
                               pred=pred,
                               logos=rhet[0][0],
                               pathos=rhet[0][1],
                               ethos=rhet[0][2])
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

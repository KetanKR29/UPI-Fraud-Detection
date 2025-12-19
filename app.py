from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])

        prediction = model.predict(final_features)

        result = "Fraud Transaction 🚨" if prediction[0] == 1 else "Safe Transaction ✅"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)


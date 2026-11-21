import pickle
from pathlib import Path

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder="templates/static")

with (BASE_DIR / "model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

with (BASE_DIR / "label.pkl").open("rb") as label_file:
    label = pickle.load(label_file)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    password = request.form.get("password", "")

    if not password:
        return render_template(
            "index.html",
            prediction="Please enter a password.",
        ), 400

    features = [[
        len(password),
        sum(character.isupper() for character in password),
        sum(character.islower() for character in password),
        sum(character.isdigit() for character in password),
        sum(not character.isalnum() for character in password),
    ]]

    try:
        result = model.predict(features)
        prediction = label.inverse_transform(result)[0]
    except (ValueError, AttributeError) as error:
        app.logger.exception("Could not predict password strength")
        return render_template(
            "index.html",
            prediction="Prediction failed. Please check the model files.",
        ), 500

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port= int(__import__("os").environ.get("PORT", 5000)))

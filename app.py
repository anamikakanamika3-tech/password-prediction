import pickle

from flask import Flask, render_template, request

app = Flask(__name__, static_folder="templates/static")

model = pickle.load(open("model.pkl", "rb"))
label = pickle.load(open("label.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    password = request.form["password"]

    features = [[
        len(password),
        sum(c.isupper() for c in password),
        sum(c.islower() for c in password),
        sum(c.isdigit() for c in password),
        sum(not c.isalnum() for c in password),
    ]]
    result = model.predict(features)
    prediction = label.inverse_transform(result)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Pclass = int(request.form["Pclass"])
    Age = float(request.form["Age"])
    SibSp = int(request.form["SibSp"])
    Parch = int(request.form["Parch"])
    Fare = float(request.form["Fare"])
    HasCabin = int(request.form["HasCabin"])
    Embarked_Q = int(request.form["Embarked_Q"])
    Embarked_S = int(request.form["Embarked_S"])
    Sex_male = int(request.form["Sex_male"])

    input_data = np.array([[Pclass, Age, SibSp, Parch, Fare, HasCabin, Embarked_Q, Embarked_S, Sex_male]])

    numeric_cols = [1, 2, 3, 4]
    input_data[:, numeric_cols] = scaler.transform(input_data[:, numeric_cols])

    prediction = model.predict(input_data)[0]

    result = "Survived" if prediction == 1 else "Did not survive"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

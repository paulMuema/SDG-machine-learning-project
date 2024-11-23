import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("./diabetes-prediction-model", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collecting user inputs from the form
    gender = request.form["gender"]
    fpg = float(request.form["fpg"])  # Fasting Plasma Glucose
    hba1c = float(request.form["hba1c"])  # HbA1c level
    nocturia = int(request.form["nocturia"])  # Excessive urination at night (0/1)
    polyuria = int(request.form["polyuria"])  # Excessive urination (0/1)
    weight_loss = int(request.form["weight_loss"])  # Weight loss (0/1)
    vomiting = int(request.form["vomiting"])  # Vomiting (0/1)
    nausea = int(request.form["nausea"])  # Nausea (0/1)
    polydipsia = int(request.form["polydipsia"])  # Excessive thirst (0/1)
    polyphagia = int(request.form["polyphagia"])  # Excessive hunger (0/1)
    headache = int(request.form["headache"])  # Headache (0/1)
    bmi = float(request.form["bmi"])  # BMI

    # Convert gender to numeric value (Male=1, Female=0)
    gender_numeric = 1 if gender == "Male" else 0

    # Creating the feature array
    features = [gender_numeric, fpg, hba1c, nocturia, polyuria, weight_loss, vomiting,
                nausea, polydipsia, polyphagia, headache, bmi]

    # Convert the features into a numpy array for prediction
    final_features = [np.array(features)]

    # Making the prediction
    prediction = model.predict(final_features)

    # If the prediction is 1, it means the person is likely to have heart disease
    if prediction[0] == 1:
        prediction_text = "The person is likely to have heart disease."
    else:
        prediction_text = "The person is likely not to have heart disease."

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
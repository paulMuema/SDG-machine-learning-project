import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize the Flask app and specify the templates folder
app = Flask(__name__, template_folder="web-app/templates")

# Load the trained diabetes prediction model
model = pickle.load(open("./diabetes-prediction-model.pkl", "rb"))

@app.route("/")
def home():
    # Render the index.html for the home page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect user inputs from the form
    age = int(request.form["age"])  # Age
    gender = request.form["gender"]  # Gender (Male/Female)
    fpg = float(request.form["fpg"])  # Fasting Plasma Glucose
    hba1c = float(request.form["hba1c"])  # HbA1c level
    nocturia = int(request.form["nocturia"])  # Excessive urination at night (0/1)
    polyuria = int(request.form["polyuria"])  # Excessive urination (0/1)
    weight_loss = int(request.form["weight_loss"])  # Weight loss (0/1)
    vomiting = int(request.form["vomiting"])  # Vomiting episodes (0/1)
    nausea = int(request.form["nausea"])  # Frequent nausea (0/1)
    polydipsia = int(request.form["polydipsia"])  # Excessive thirst (0/1)
    polyphagia = int(request.form["polyphagia"])  # Excessive hunger (0/1)
    headache = int(request.form["headache"])  # Frequent headaches (0/1)
    bmi = float(request.form["bmi"])  # BMI (Body Mass Index)

    # Convert gender to numeric value (Male=1, Female=0)
    gender_numeric = 1 if gender == "Male" else 0

    # Creating the feature array
    features = [age, gender_numeric, fpg, hba1c, nocturia, polyuria, weight_loss, vomiting,
                nausea, polydipsia, polyphagia, headache, bmi]

    # Convert the features into a numpy array for prediction
    final_features = [np.array(features)]

    # Making the prediction
    prediction = model.predict(final_features)

    # Prepare the result text
    if prediction[0] == 1:
        prediction_text = "The person is likely to have diabetes."
    else:
        prediction_text = "The person is unlikely to have diabetes."

    # Render the result on the home page
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    # Run the app in debug mode
    app.run(debug=True)
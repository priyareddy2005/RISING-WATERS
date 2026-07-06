#import required libraries for flask and machine learning
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

#initialize your flask application
app = Flask(__name__)

#load the required training model
MODEL_PATH = os.path.join("model", "flood_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = None
scaler = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

#Home page route
@app.route('/')
def home():
    return render_template("index.html")

#Predict Flood risk based on input
@app.route('/predict', methods=['POST'])
def predict():
#input input values from html page
    try:
        annual_rainfall = float(request.form['annual_rainfall'])
        cloud_visibility = float(request.form['cloud_visibility'])
        jan = float(request.form['jan'])
        feb = float(request.form['feb'])
        mar = float(request.form['mar'])
        apr = float(request.form['apr'])
        may = float(request.form['may'])
        jun = float(request.form['jun'])
        jul = float(request.form['jul'])
        aug = float(request.form['aug'])
        sep = float(request.form['sep'])
        octo = float(request.form['oct'])
        nov = float(request.form['nov'])
        dec = float(request.form['dec'])
 
#import data into dataframe
        data = pd.DataFrame([[

            annual_rainfall,
            cloud_visibility,
            jan,
            feb,
            mar,
            apr,
            may,
            jun,
            jul,
            aug,
            sep,
            octo,
            nov,
            dec

        ]], columns=[

            "Annual_Rainfall",
            "Cloud_Visibility",
            "Jan_Rainfall",
            "Feb_Rainfall",
            "Mar_Rainfall",
            "Apr_Rainfall",
            "May_Rainfall",
            "Jun_Rainfall",
            "Jul_Rainfall",
            "Aug_Rainfall",
            "Sep_Rainfall",
            "Oct_Rainfall",
            "Nov_Rainfall",
            "Dec_Rainfall"

        ])

        if model is None:
            return render_template(
                "index.html",
                prediction="Model not found. Train the model first."
            )

        try:
            prediction = model.predict(data)
        except:
            prediction = model.predict(scaler.transform(data))

        result = "⚠ Flood Likely" if prediction[0] == 1 else "✅ No Flood Expected"

        return render_template(
            "index.html",
            prediction=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )

#Run the flask development server
if __name__ == "__main__":
    app.run(debug=True)
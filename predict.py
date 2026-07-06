#import required libraries
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model/flood_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Example input data
sample_data = {
    "Annual_Rainfall": [2200],
    "Cloud_Visibility": [70],
    "Jan_Rainfall": [120],
    "Feb_Rainfall": [100],
    "Mar_Rainfall": [150],
    "Apr_Rainfall": [200],
    "May_Rainfall": [300],
    "Jun_Rainfall": [450],
    "Jul_Rainfall": [600],
    "Aug_Rainfall": [550],
    "Sep_Rainfall": [400],
    "Oct_Rainfall": [250],
    "Nov_Rainfall": [180],
    "Dec_Rainfall": [130]
}


input_data = pd.DataFrame(sample_data)


scaled_data = scaler.transform(input_data)


prediction = model.predict(scaled_data)

if prediction[0] == 1:
    print("Flood Likely")
else:
    print("No Flood Expected")
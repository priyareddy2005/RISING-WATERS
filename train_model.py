# Import required libraries for data processing,
# machine learning, and model serialization.
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load the flood prediction dataset.
data = pd.read_csv("data/flood_data.csv")

X = data.drop("Flood", axis=1)
y = data["Flood"]

# Display basic information about the dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_model = None
best_accuracy = 0

print("\nModel Accuracy\n")

# Split the dataset into training and testing sets.
for name, model in models.items():

    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name}: {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


joblib.dump(best_model, "model/flood_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nBest Accuracy:", round(best_accuracy*100, 2), "%")
print("Model saved successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T.dot(y))
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Load dataset
df = pd.read_csv("housing_price_india.csv")
df["number of bathrooms"] = df["number of bathrooms"].astype(int)

# Separate features and target
features = ["number of bedrooms", "number of bathrooms", "living area", "Postal Code"]
X_raw = df[features]
y = df["Price"]

# Apply OneHotEncoding on Postal Code
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
location_encoded = encoder.fit_transform(X_raw[["Postal Code"]])

# Combine with other features
X_numeric = X_raw.drop(columns=["Postal Code"]).values
X_final = np.hstack([X_numeric, location_encoded])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = MyLinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump({"weights": model.weights, "bias": model.bias, "encoder": encoder}, f)

# Evaluate
train_rmse = np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2))
test_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
print(f"Training RMSE: ₹{train_rmse:,.2f}")
print(f"Testing RMSE: ₹{test_rmse:,.2f}")

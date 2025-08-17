import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data.csv")

# Drop unnecessary columns
data = data.drop(columns=["date", "street", "city", "statezip", "country"])

# Define features and target
X = data.drop(columns=["price"]).values
y = data["price"].values.reshape(-1, 1)

# Train-test split
def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2, random_state=42)

# Add bias (intercept) column
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Gradient Descent Implementation
def gradient_descent(X, y, lr=0.00000001, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))  # initialize weights
    
    for epoch in range(epochs):
        y_pred = X.dot(theta)
        error = y_pred - y
        gradients = (2/m) * X.T.dot(error)
        theta -= lr * gradients

        if epoch % 100 == 0:
            mse = np.mean(error**2)
            print(f"Epoch {epoch}: MSE = {mse:.2f}")
    
    return theta

# Train the model
theta = gradient_descent(X_train_bias, y_train, lr=1e-7, epochs=1000)

# Predict on test data
y_pred = X_test_bias.dot(theta)

# Evaluation
mse = np.mean((y_test - y_pred) ** 2)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print(f"\nFinal Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example prediction
new_data = np.array([[3.0, 2.0, 1500, 5000, 1.0, 0, 0, 3, 1500, 0, 1980, 2005]])
new_data_bias = np.c_[np.ones((new_data.shape[0], 1)), new_data]
predicted_price = new_data_bias.dot(theta)
print(f"Predicted Price: {predicted_price[0][0]}")

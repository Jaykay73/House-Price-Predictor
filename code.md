# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Drop unnecessary columns (e.g., date, street, city, statezip, country)
data = data.drop(columns=['date', 'street', 'city', 'statezip', 'country'])

# Check for missing values and handle them if necessary
print(data.isnull().sum())

# Define features (X) and target variable (y)
X = data.drop(columns=['price'])  # Features: all columns except 'price'
y = data['price']  # Target variable: 'price'

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example of making predictions with new data
new_data = [[3.0, 2.0, 1500, 5000, 1.0, 0, 0, 3, 1500, 0, 1980, 2005]]  # Replace with actual values
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")

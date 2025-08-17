# Linear Regression from Scratch with NumPy

## Overview

This project demonstrates how to implement **Linear Regression** from scratch using only **NumPy**, without relying on machine learning libraries such as scikit-learn. The model is trained using **Gradient Descent**, and performance is evaluated using Mean Squared Error (MSE) and the R-squared (R²) metric.

The dataset used contains housing data with features such as the number of bedrooms, bathrooms, square footage, lot size, and other attributes. The target variable is the house price.

---

## Features

* Train-test split implemented manually with NumPy
* Bias (intercept) term added explicitly
* Linear regression implemented with:

  * **Gradient Descent optimization**
* Evaluation using:

  * Mean Squared Error (MSE)
  * R-squared (R²) score
* Supports predictions on new data

---

## Project Structure

```
.
├── data.csv          # Dataset file
├── house_price_prediction.py   # Main implementation
└── README.md         # Project documentation
```

---

## How It Works

1. Load and preprocess the dataset:

   * Drop irrelevant columns such as `date`, `street`, `city`, `statezip`, and `country`.
   * Separate features (X) and target variable (y).
   * Split data into training and testing sets.

2. Add a bias term (intercept) to the feature matrix.

3. Train the model using **Gradient Descent**:

   * Initialize weights to zeros.
   * Iteratively update weights to minimize the Mean Squared Error.

4. Evaluate the trained model using:

   * Mean Squared Error (MSE)
   * R-squared (R²)

5. Predict housing prices for new samples.

---

## Example Usage

```bash
python linear_regression.py
```

Example output during training:

```
Epoch 0: MSE = 45000000000.00
Epoch 100: MSE = 32000000000.00
...
Final Mean Squared Error: 29000000000.00
R-squared: 0.74
Predicted Price: 450000.23
```

---

## Prediction Example

To predict the price of a new house, provide its features:

```python
new_data = np.array([[3.0, 2.0, 1500, 5000, 1.0, 0, 0, 3, 1500, 0, 1980, 2005]])
new_data_bias = np.c_[np.ones((new_data.shape[0], 1)), new_data]
predicted_price = new_data_bias.dot(theta)
```

Output:

```
Predicted Price: 450000.23
```

---

## Requirements

* Python 3.x
* NumPy
* Pandas

Install dependencies:

```bash
pip install numpy pandas
```


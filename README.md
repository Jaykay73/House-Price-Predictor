# Linear Regression Model for House Price Prediction

This repository contains a Python script that implements a Linear Regression model to predict house prices based on a dataset. The script uses the `pandas` library for data manipulation, `scikit-learn` for model training and evaluation, and includes steps for data preprocessing, model training, and prediction.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Evaluation](#evaluation)
- [Example Prediction](#example-prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to predict house prices using a Linear Regression model. The dataset used in this project contains various features related to houses, such as the number of bedrooms, square footage, and year built. The script preprocesses the data, trains a Linear Regression model, and evaluates its performance using metrics like Mean Squared Error (MSE) and R-squared.

## Requirements

To run this code, you need the following Python libraries installed:

- `pandas`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install pandas scikit-learn
```

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/house-price-prediction.git
```

2. Navigate to the project directory:

```bash
cd house-price-prediction
```

3. Ensure you have the required libraries installed (see [Requirements](#requirements)).

## Usage

1. Place your dataset in the project directory and name it `data.csv`.

2. Run the script:

```bash
python house_price_prediction.py
```

3. The script will output the following:
   - The first few rows of the dataset.
   - The number of missing values in each column.
   - The Mean Squared Error (MSE) and R-squared values of the model.
   - An example prediction for a new set of data.

## Code Overview

The script performs the following steps:

1. **Import Libraries**: The necessary libraries are imported, including `pandas` for data manipulation and `scikit-learn` for model training and evaluation.

2. **Load Dataset**: The dataset is loaded from `data.csv` using `pandas`.

3. **Data Preprocessing**:
   - Unnecessary columns (e.g., `date`, `street`, `city`, `statezip`, `country`) are dropped.
   - Missing values are checked and handled if necessary.

4. **Feature and Target Selection**:
   - Features (`X`) are selected by dropping the `price` column.
   - The target variable (`y`) is set to the `price` column.

5. **Train-Test Split**: The data is split into training and testing sets (80% train, 20% test).

6. **Model Training**: A Linear Regression model is initialized and trained on the training data.

7. **Model Evaluation**: The model's performance is evaluated on the test set using Mean Squared Error (MSE) and R-squared metrics.

8. **Prediction**: An example prediction is made using new data.

## Evaluation

The model's performance is evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values. Lower values indicate better performance.
- **R-squared**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher values indicate better performance.

The script outputs these metrics after evaluating the model on the test set.

## Example Prediction

The script includes an example of making predictions with new data. You can replace the values in the `new_data` variable with actual values to make predictions for new houses.

```python
new_data = [[3.0, 2.0, 1500, 5000, 1.0, 0, 0, 3, 1500, 0, 1980, 2005]]  # Replace with actual values
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")
```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this README to better suit your project's needs. Happy coding! 🚀

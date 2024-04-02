# Stock Predictor

## Overview
Stock Predictor is a machine learning model designed to forecast stock prices using Support Vector Machines (SVM). This project aims to provide a tool for investors and traders to make informed decisions based on predicted stock prices.

## Features
- **SVM for Stock Price Prediction**: Support Vector Machines are utilized for their effectiveness in regression tasks, providing reliable predictions for stock prices.
- **Flask Backend**: The Flask backend enables easy deployment of the model, allowing users to interact with it through API endpoints.
- **Data Retrieval with Yahoo Finance**: Historical stock data is fetched from Yahoo Finance using the `yfinance` library, ensuring accurate and up-to-date information for analysis.
- **Preprocessing Techniques**: Data preprocessing is applied to optimize the input data, including feature engineering and scaling using Min-Max normalization.
- **Model Evaluation**: The performance of the model is evaluated using standard regression metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), providing insights into its accuracy.

## Installation
To set up the Stock Predictor on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your_repository.git
    cd stock-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Prepare Data**: Before training the model, ensure that you have historical stock data available. You can specify the ticker symbol and the forecast period to make predictions.

2. **Train and Test Model**: Use the provided functions to preprocess the data, train the SVM model, and evaluate its performance.

3. **Evaluate Model**: After training the model, evaluate its performance using the chosen metrics. Adjust parameters or preprocessing techniques as necessary to improve accuracy.

4. **Interact with Flask Backend**: Once the model is trained and evaluated, deploy it using the Flask backend. Users can make predictions by sending HTTP requests to the provided API endpoints.

## Example
```python
ticker_symbol = 'AAPL'
forecast_days = 30

# Evaluate the model for the given stock symbol and forecast period
predicted_prices = model_evaluation_callback(ticker_symbol, forecast_days)

print("Predicted Prices:", predicted_prices)


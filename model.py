import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import pandas as pd 

def preprocess_data(stock_data, forecast_days):
   
    to_be_used = stock_data.tail(forecast_days)
    X_solve = to_be_used
    stock_data['pred'] = stock_data['Close'].shift(-forecast_days)
    
    stock_data = stock_data[:-forecast_days]
    
    X = stock_data.drop(columns=['pred'])
    y = stock_data['pred']
    
    return X,y,X_solve

def train_test_model(X, y, X_solve):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_solve_scaled = scaler.transform(X_solve)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, shuffle=False)
    
    
    svr = SVR(kernel='rbf')
    
   
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1]
    }
    
    
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    
    svr_best = SVR(kernel='rbf',**best_params)
    svr_best.fit(X_train, y_train)
    
   
    y_pred = svr_best.predict(X_test)
    y_pred_solve = svr_best.predict(X_solve_scaled)
    
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mse, mae, y_pred_solve

def evaluate_model(ticker_symbol, forecast_days):
    
    stock_data = yf.download(ticker_symbol, period='365d', interval='1d')
 
    
    new_df = pd.DataFrame()
    for i in stock_data.columns:
        for j in stock_data.columns:
            for k in stock_data.columns:
                new_df[i+j+k] = stock_data[i] * stock_data[j] * stock_data[k]

    for i in stock_data.columns:
         for j in stock_data.columns:
            new_df[i+j] = stock_data[i] * stock_data[j]
            


    stock_data = pd.concat([stock_data, new_df], axis=1)

  
    
    X, y, X_solve = preprocess_data(stock_data, forecast_days)
    
    mse, mae, predicted = train_test_model(X, y, X_solve)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    return predicted


def model_evaluation_callback(ticker_symbol, forecast_days):
    predicted = evaluate_model(ticker_symbol, forecast_days)
    return predicted

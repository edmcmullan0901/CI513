# src/model_training.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def train_polynomial_model(df, degree=2, target_col='penalized_energy'):
   
    #Trains a polynomial regression model on air_temperature and the target column.
    
    X = df[['air_temperature']].values
    y = df[target_col].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly, X_poly, y

def find_optimal_temperature(model, poly, temp_min=10, temp_max=35, steps=200):
    
    #finds the optimal temp and the corresponding predicted minimum

    temp_range = np.linspace(temp_min, temp_max, steps).reshape(-1, 1)
    temp_poly = poly.transform(temp_range)
    preds = model.predict(temp_poly)

    min_index = np.argmin(preds)
    optimal_temp = temp_range[min_index][0]
    predicted_value = preds[min_index]

    return optimal_temp, predicted_value

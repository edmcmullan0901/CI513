from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred):
    #creates dictionary of evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)



    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def compare_models(results_dict):
    #compares evaluation metrics across different models
    rows = []

    for model_name, (y_true, y_pred) in results_dict.items():
        metrics = evaluate_model(y_true, y_pred)
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

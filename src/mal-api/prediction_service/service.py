import os
import joblib
import numpy as np

def predict_with_rf_and_mwa(iot_data, model_6_hour_steps=5, total_6_hour_steps=28):
    # Set up path
    if os.path.exists('/app'):
        models_dir = '/app/models'
    else:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'models')

    #load model and scalers
    model = joblib.load(os.path.join(models_dir, "soil_moisture_model.pkl"))
    scaler_X = joblib.load(os.path.join(models_dir, "scaler_X.save"))
    scaler_y = joblib.load(os.path.join(models_dir, "scaler_y.save"))

    #input preparation
    input_vector = np.array([[iot_data["Soil_Moisture"],
                              iot_data["Ambient_Temperature"],
                              iot_data["Humidity"]]])
    input_scaled = scaler_X.transform(input_vector)

    #predict with model
    y_pred_scaled = model.predict(input_scaled)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)[0]

    input_moisture = iot_data["Soil_Moisture"]
    y_pred_real = np.minimum(y_pred_real, input_moisture)

    if total_6_hour_steps <= model_6_hour_steps:
        return y_pred_real[:total_6_hour_steps]

    #MWA extension for more than 5 steps
    deltas = np.diff(y_pred_real)
    weights = np.arange(1, len(deltas) + 1)
    weighted_drop = -np.dot(deltas, weights) / weights.sum() if len(deltas) > 0 else 0.5

    last_known = y_pred_real[-1]
    extension = [max(last_known - weighted_drop * (j + 1), 0)
                 for j in range(total_6_hour_steps - model_6_hour_steps)]

    full_forecast = np.concatenate([y_pred_real, extension])
    return np.clip(full_forecast, 0, input_moisture)

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Cleaned_Soil_Moisture"] = df["Soil_Moisture"]
    return df


def extract_drying_cycles(df, min_cycle_length=2, spike_threshold=3.0):
    cycles = []
    current_cycle = []

    for i in range(1, len(df)):
        prev = df["Cleaned_Soil_Moisture"].iloc[i - 1]
        curr = df["Cleaned_Soil_Moisture"].iloc[i]

        if curr <= prev:
            current_cycle.append(i)
        elif curr - prev >= spike_threshold:
            if len(current_cycle) >= min_cycle_length:
                cycles.append(current_cycle.copy())
            current_cycle = [i]
        else:
            if len(current_cycle) >= min_cycle_length:
                cycles.append(current_cycle.copy())
            current_cycle = []

    if len(current_cycle) >= min_cycle_length:
        cycles.append(current_cycle)

    return cycles

def analyze_drying_cycles(df):
    #explains why dataset is not ideal for this case
    print("Drying Cycle Analysis")

    cycles = extract_drying_cycles(df)
    cycle_lengths = [len(c) for c in cycles]

    print(f"Total drying cycles: {len(cycles)}")
    print(f"Top 5 longest cycles: {sorted(cycle_lengths, reverse=True)[:5]}")

    #extract start and end moisture values
    start_vals = [df.iloc[cycle[0]]["Cleaned_Soil_Moisture"] for cycle in cycles]
    end_vals = [df.iloc[cycle[-1]]["Cleaned_Soil_Moisture"] for cycle in cycles]

    #heatmap shows start vs. end values in each cycle
    plt.figure(figsize=(8, 6))
    sns.histplot(x=start_vals, y=end_vals, bins=30, pthresh=0.1, cmap="YlGnBu", cbar=True)
    plt.xlabel("Start Moisture")
    plt.ylabel("End Moisture")
    plt.title("Start vs. End Moisture Heatmap (Drying Cycles)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #histogram shows most frequent values observed during drying
    all_vals = [df.iloc[i]["Cleaned_Soil_Moisture"] for cycle in cycles for i in cycle]
    plt.figure(figsize=(8, 4))
    sns.histplot(all_vals, bins=30, kde=True, color="teal")
    plt.xlabel("Soil Moisture Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Soil Moisture During Drying Cycles")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def prepare_rf_dataset(df, model_6_hour_steps=5):
    features = ["Cleaned_Soil_Moisture", "Ambient_Temperature", "Humidity"]
    cycles = extract_drying_cycles(df, min_cycle_length=2)

    X = []
    y = []

    for cycle in cycles:
        if len(cycle) < model_6_hour_steps + 1:
            continue

        input_idx = cycle[0]
        future_idxs = cycle[1:1 + model_6_hour_steps]

        input_features = df.iloc[input_idx][features].values
        target_values = df.iloc[future_idxs]["Cleaned_Soil_Moisture"].values

        X.append(input_features)
        y.append(target_values)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid training samples.")

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_random_forest(X, y, scaler_y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print("Random Forest Evaluation:")
    print(f" - MAE: {mae:.4f}")
    print(f" - RMSE: {rmse:.4f}")

    plt.plot(y_test_real[0], label="Actual")
    plt.plot(y_pred_real[0], label="Predicted")
    plt.title("Soil Moisture Forecast (5 × 6-hour steps - RF)")
    plt.xlabel("6-hour Step")
    plt.ylabel("Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np

    if not hasattr(model, "feature_importances_"):
        print("The provided model does not support feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def save_model_and_scalers(model, scaler_X, scaler_y, model_filename='soil_moisture_model.pkl'):
    if os.path.exists('/app'):
        models_dir = '/app/models'
    else:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)

    scaler_X_path = os.path.join(models_dir, 'scaler_X.save')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.save')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    print(f"Model and scalers saved to {models_dir}")

    timestamp_path = os.path.join(models_dir, 'last_updated.txt')
    with open(timestamp_path, 'w') as f:
        f.write(f"Model last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timestamp saved to {timestamp_path}")


def main():
    filepath = "plant_health_data.csv"
    df = load_and_preprocess_data(filepath)
    #analyze_drying_cycles(df)
    model_6_hour_steps = 5

    #prepare drying cycles
    cycles = extract_drying_cycles(df, min_cycle_length=2)
    cycle_lengths = [len(c) for c in cycles]
    top_cycles = sorted(cycle_lengths, reverse=True)[:5]

    #print(f"Detected {len(cycles)} drying cycles")
    #print(f"Top 5 drying cycle lengths: {top_cycles}")

    #data preparation and model training
    X, y, scaler_X, scaler_y = prepare_rf_dataset(df, model_6_hour_steps=model_6_hour_steps)
    model = train_random_forest(X, y, scaler_y)

    save_model_and_scalers(model, scaler_X, scaler_y)

    #feature impact
    feature_names = ["Cleaned_Soil_Moisture", "Ambient_Temperature", "Humidity"]
    plot_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()

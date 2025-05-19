import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import joblib

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print("Missing values:\n", df.isnull().sum())
    return df.dropna()


def visualize_data(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include='number').columns
    plt.figure(figsize=(15, 10))
    df[numeric_cols].boxplot()
    plt.xticks(rotation=90)
    plt.title("Boxplots of Numeric Features")
    plt.show()

    df[numeric_cols].hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    #make sure that datetime features are circular values
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_of_week'] = df['Timestamp'].dt.weekday  # 0 = Monday 1= Tuesday ... 6 = Sunday
    df['month'] = df['Timestamp'].dt.month

    # Optional: drop raw hour/month if not useful
    #df.drop(columns=['hour'], inplace=True)

    return df

def plot_feature_importances(model, feature_names):

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Print the table
    print("\nFeature Importances:")
    print(importance_df)

    # Plot the importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances from Random Forest')
    plt.tight_layout()
    plt.show()


def prepare_features(df: pd.DataFrame):
    df = add_datetime_features(df)

    X = df[['Ambient_Temperature', 'Humidity', 'hour_sin', 'hour_cos', 'day_of_week']]
    y = df['Soil_Moisture']
    return X, y


def tune_and_train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return rmse, r2


def save_model(model, model_filename='soil_moisture_model.pkl'):
    if os.path.exists('/app'):
        models_dir = '/app/models'
    else:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Construct full file paths
    model_path = os.path.join(models_dir, model_filename)

    # Save model and scaler using joblib
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    # Save the last updated timestamp
    timestamp_path = os.path.join(models_dir, 'last_updated.txt')
    with open(timestamp_path, 'w') as f:
        f.write(f"Model last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timestamp saved to {timestamp_path}")


def main():
    df = load_and_clean_data("plant_health_data.csv")
    #visualize_data(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tune_and_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importances(model, X.columns)
    save_model(model)


if __name__ == "__main__":
    main()

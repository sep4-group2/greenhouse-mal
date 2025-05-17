import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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


def prepare_features(df: pd.DataFrame):
    X = df[['Ambient_Temperature', 'Humidity']]
    y = df['Soil_Moisture']
    return X, y


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return rmse, r2


def save_model(model, scaler, model_path='soil_moisture_model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def main():
    df = load_and_clean_data("plant_health_data.csv")
    visualize_data(df)

    X, y = prepare_features(df)
    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    save_model(model, scaler)


if __name__ == "__main__":
    main()

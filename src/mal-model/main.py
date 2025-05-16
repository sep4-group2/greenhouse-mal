import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("plant_health_data.csv")

# Checked for missing values
print("Missing values:\n", df.isnull().sum())


print("\nDataFrame Info:")
print(df.info())


print("\nDescriptive Statistics:")
print(df.describe())

# Boxplots to check for outliers visually
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


X = df.drop(columns=['Soil_Moisture', 'Timestamp', 'Plant_ID', 'Plant_Health_Status'])
y = df['Soil_Moisture']


print("\nSelected features for X:")
print(X.columns)


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (Optional) Encoding Plant_Health_Status if needed later
df['Plant_Health_Status_Encoded'] = df['Plant_Health_Status'].astype('category').cat.codes

print("\nTransformations complete. Data is ready for modeling.")

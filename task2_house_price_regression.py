"""
Task 2: House price prediction with Linear Regression
Dataset: Kaggle "Housing Prices Dataset" (Housing.csv) - columns: price, area, bedrooms,
bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning,
parking, prefarea, furnishingstatus.
Usage:  python task2_house_price_regression.py [path_to_Housing.csv]
If the file isn't found, synthetic data with the same columns is generated so the script runs.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def make_sample_data(n=545, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area": rng.integers(1500, 16000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 4, n),
        "stories": rng.integers(1, 5, n),
        "mainroad": rng.choice(["yes", "no"], n, p=[0.85, 0.15]),
        "guestroom": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "basement": rng.choice(["yes", "no"], n),
        "airconditioning": rng.choice(["yes", "no"], n),
        "parking": rng.integers(0, 4, n),
        "prefarea": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        "furnishingstatus": rng.choice(["furnished", "semi-furnished", "unfurnished"], n),
    })
    df["price"] = (df.area * 450 + df.bedrooms * 300000 + df.bathrooms * 700000
                   + df.stories * 350000 + (df.airconditioning == "yes") * 800000
                   + (df.prefarea == "yes") * 650000 + df.parking * 250000
                   + rng.normal(0, 500000, n)).round(-3)
    return df


# ---------- 1. Load ----------
path = sys.argv[1] if len(sys.argv) > 1 else "Housing.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
else:
    df = make_sample_data()
    print("Housing.csv not found -> using synthetic sample data")

TARGET = "price"
print(df.head())

# ---------- 2. Preprocess ----------
df = df.drop_duplicates()
# Fill missing: numeric -> median, categorical -> mode
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Remove extreme price outliers (IQR rule)
q1, q3 = df[TARGET].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[(df[TARGET] >= q1 - 1.5 * iqr) & (df[TARGET] <= q3 + 1.5 * iqr)]

# Encode categoricals (yes/no, furnishing status, location etc.)
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=[TARGET]).astype(float)
y = df[TARGET]

# ---------- 3. Train ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

# ---------- 4. Evaluate ----------
pred = model.predict(X_test_s)
print("\n=== Test-set performance ===")
print(f"R2   : {r2_score(y_test, pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, pred):,.0f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, pred)):,.0f}")

coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop feature influences (standardized coefficients):")
print(coef.head(8).round(0))

# ---------- 5. Plots ----------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, alpha=0.6, edgecolor="k")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted House Price")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
coef.head(8)[::-1].plot(kind="barh", color="teal")
plt.title("Top Feature Coefficients")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()


# ---------- 6. Predict a new house ----------
def predict_price(**features):
    row = pd.DataFrame([features])
    row = pd.get_dummies(row).reindex(columns=X.columns, fill_value=0).astype(float)
    return model.predict(scaler.transform(row))[0]


if "area" in X.columns:
    example = {"area": 5000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "parking": 1,
               "airconditioning_yes": 1}
    print(f"\nExample prediction (5000 sqft, 3 bed, 2 bath): {predict_price(**example):,.0f}")

print("\nSaved: actual_vs_predicted.png, feature_importance.png")

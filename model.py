#!/usr/bin/python3
"""Predict Student Score Given Study Hour."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split


df = pd.read_csv("linear_hours_scores.csv")

# Visualiation of Data
plt.scatter(df["Times"], df["Scores"], color="blue", marker="*")
plt.xlabel("Time [Hour]")
plt.ylabel("Score")
plt.title("Score vs. Time")
plt.savefig("score_vs_hours.png", dpi=300, bbox_inches='tight')

# Split for training
features = ["Times"]
target = "Scores"

X = df[features]  # Features Matrix (Must be 2D)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_mean = y_train.mean()
print("Mean Score", round(y_mean, 2))

y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Baseline MAE:", round(mae_baseline, 2))

# Create pipeline and impute missing data.
pipeline = Pipeline( steps=[
    ("imputer", SimpleImputer()),
    ("model", LinearRegression())
])
pipeline.fit(X_train, y_train)  # Train Model
y_pred_training = pipeline.predict(X_train)

# MAE during traing.
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))

# MAE duringg testing.
y_pred_testing = pipeline.predict(X_test)
mae_testing = mean_absolute_error(y_test, y_pred_testing)
print("Testing MAE:", round(mae_training, 2))

model = pipeline.named_steps["model"]
intercept = model.intercept_
coefficients = model.coef_[0]

print("Model Eqn: ", f"score = ({coefficients:.3f} * time) + {intercept:.3f}")

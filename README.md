# 📊 Score Prediction Module

A simple machine learning project that predicts student scores based on study hours using **Linear Regression**. This project demonstrates dataset preparation, baseline modeling, model training, and evaluation using scikit-learn.

## 📂 Project Structure
score_prediction_model/
├── linear_hours_scores.csv # Dataset: Study hours vs. Scores
├── model.py # Main Python script for model training and evaluation
└── score_vs_hours.png # Visualization: Scatter plot of Hours vs. Scores


## 🚀 Features
- Data visualization with Matplotlib
- Linear Regression model training with scikit-learn
- Calculation of **Baseline MAE (Mean Absolute Error)** for comparison
- Visualization of regression trends
- Handles missing data with `SimpleImputer`
- Clean and modular pipeline with `Pipeline()`

## 📁 Dataset
The dataset contains two columns:
- `Hours` → Study hours per day
- `Scores` → Exam scores corresponding to study hours

## 🛠️ Requirements
- Python 3.x
- pandas
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install pandas matplotlib scikit-learn
```

## ⚙️ Usage
python3 model.py

## Output:
-- Baseline MAE (using mean predictor)

-- Model MAE (trained Linear Regression)

-- Scatter plot visualization saved as score_vs_hours.png
## 📌 Why This Project?
This project was built to:

-- Understand the basics of regression modeling

-- Learn about model evaluation with baseline comparison

-- Practice using pipelines and imputation for preprocessing

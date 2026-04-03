# 🏥 Medical Insurance Price Prediction using Machine Learning

A machine learning project that predicts medical insurance charges for individuals based on key personal and lifestyle factors such as age, BMI, smoking status, and number of children. Multiple regression models are trained, compared, and tuned to identify the best-performing approach.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Overview

Medical insurance pricing is influenced by a range of personal and demographic factors. This project builds a predictive model that estimates an individual's insurance charges using supervised machine learning regression techniques. The final model is saved as a `.pkl` file for deployment or future use.

---

## Dataset

The dataset used is `insurance.csv`, a publicly available dataset with the following features:

| Feature    | Description                                      |
|------------|--------------------------------------------------|
| `age`      | Age of the primary beneficiary                   |
| `sex`      | Gender of the beneficiary (male/female)          |
| `bmi`      | Body Mass Index                                  |
| `children` | Number of dependents covered                     |
| `smoker`   | Smoking status (yes/no)                          |
| `region`   | Residential area in the US                       |
| `charges`  | Individual medical insurance cost (target)       |

---

## Project Workflow

1. **Data Loading & Exploration** — Loading the dataset, checking data types, shape, and statistical summary.
2. **Exploratory Data Analysis (EDA)** — Visualizing distributions using pie charts, bar plots, and scatter plots to understand relationships between features and insurance charges.
3. **Data Preprocessing**
   - Removed duplicate records
   - Handled outliers in `bmi` using the IQR method and `ArbitraryOutlierCapper`
   - Label encoded categorical features (`sex`, `smoker`, `region`)
4. **Feature Selection** — Used XGBoost feature importances to identify that `smoker`, `age`, `bmi`, and `children` are the most influential features. `sex` and `region` were dropped.
5. **Model Training & Evaluation** — Trained multiple regression models and evaluated using R² score and 5-fold cross-validation.
6. **Hyperparameter Tuning** — Applied `GridSearchCV` to optimize `RandomForestRegressor`, `GradientBoostingRegressor`, and `XGBRegressor`.
7. **Model Saving** — Final model saved to `insurancemodelf.pkl` using `pickle`.
8. **Prediction** — Demonstrated prediction on a new unseen data sample.

---

## Models Used

| Model                        | Notes                                    |
|------------------------------|------------------------------------------|
| Linear Regression            | Baseline model                           |
| Support Vector Regressor     | SVR with default kernel                  |
| Random Forest Regressor      | Tuned with GridSearchCV (n_estimators=120) |
| Gradient Boosting Regressor  | Tuned (n_estimators=19, learning_rate=0.2) |
| **XGBoost Regressor**        | **Best model** (n_estimators=15, max_depth=3, gamma=0) |

---

## Results

The **XGBoost Regressor** (after hyperparameter tuning and feature selection) achieved the best performance. Feature importance analysis confirmed that `smoker`, `age`, `bmi`, and `children` are the primary drivers of insurance cost.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amir6862/medical-insurance-price-prediction.git
   cd medical-insurance-price-prediction
   ```

2. **Install required libraries**
   ```bash
   pip install numpy pandas scikit-learn xgboost seaborn matplotlib feature_engine
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Medical_Insurance_Price_Prediction_using_Machine_Learning_-_Python.ipynb
   ```

---

## Usage

To predict insurance charges for a new individual:

```python
import pandas as pd
from pickle import load

model = load(open('insurancemodelf.pkl', 'rb'))

new_data = pd.DataFrame({
    'age': [19],
    'bmi': [27.9],
    'children': [0],
    'smoker': [1]   # 1 = yes, 0 = no
})

predicted_charge = model.predict(new_data)
print(f"Predicted Insurance Charge: ${predicted_charge[0]:,.2f}")
```

---

## Project Structure

```
📦 medical-insurance-price-prediction
 ┣ 📓 Medical_Insurance_Price_Prediction_using_Machine_Learning_-_Python.ipynb
 ┣ 📄 insurance.csv
 ┣ 📦 insurancemodelf.pkl
 ┗ 📄 README.md
```

---

## Author

**Amir** — [@amir6862](https://github.com/amir6862)

---

> ⭐ If you found this project helpful, please consider giving it a star!

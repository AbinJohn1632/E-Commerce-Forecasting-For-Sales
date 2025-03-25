# E-Commerce Forecasting for Sales

## Overview
This project focuses on developing a machine learning model to predict future sales quantities of e-commerce products using historical sales data. By leveraging advanced data science techniques, the goal is to help e-commerce businesses optimize stock levels, minimize overstocking, and prevent stockouts.

## Techniques Used

### Data Pre-processing
```python
# Feature Selection
X = df[['Price', 'Past_Purchase_Trends', 'Competitor_Price', 'effective_price', 'price_sensitivity', 'comp_impact', 'year_sine']]
y = df['Sales_Quantity']

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Model Development
```python
import xgboost as xgb
import optuna
from sklearn.metrics import mean_tweedie_deviance

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1),
        "tweedie_variance_power": trial.suggest_uniform("tweedie_variance_power", 1.01, 1.09),
    }
    
    model = xgb.XGBRegressor(**params, objective="reg:tweedie", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    tweedie_loss = mean_tweedie_deviance(y_test, y_pred, power=params["tweedie_variance_power"])
    
    return tweedie_loss

# Hyperparameter Tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

# Best Parameters
best_params = study.best_params
print("Best Parameters:", best_params)
```

### Validation and Testing
```python
# Ensure Model Generalization
final_model = xgb.XGBRegressor(**best_params, objective="reg:tweedie", random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

# Evaluate Model Performance
final_tweedie_loss = mean_tweedie_deviance(y_test, y_pred, power=best_params["tweedie_variance_power"])
print("Final Tweedie Loss:", final_tweedie_loss)
```

## Getting Started
1. **Install Dependencies:** Ensure required libraries (`pandas`, `numpy`, `xgboost`, `optuna`, `sklearn`) are installed.
2. **Load Data:** Read the dataset (`train.csv` and `test.csv`).
3. **Feature Engineering & Preprocessing:** Apply scaling and transformations.
4. **Train Model:** Run hyperparameter tuning and model training.
5. **Evaluate Performance:** Validate using RMSE and Tweedie loss.
6. **Generate Predictions:** Use the trained model for final predictions.



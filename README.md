# What-Drives-the-Price-of-a-Car?-

# Used Car Price Prediction

This project investigates the key factors that influence used car prices, using a dataset of over 370,000 vehicles. The goal is to support used car dealerships in making informed, data-driven pricing decisions by building a predictive model and uncovering meaningful patterns in the data.

## Objective

To identify the primary features that affect a used car's market price and build a machine learning model that can predict price with a high degree of accuracy.

## Methodology

This project follows the CRISP-DM framework to guide each phase of analysis and modeling:

### 1. Business Understanding

We framed the problem from a dealership’s perspective: how can data be used to predict used car prices more accurately and consistently? The key objective was to support better pricing decisions that account for both vehicle condition and market behavior.

### 2. Data Understanding

We explored a dataset containing listings for over 370,000 used vehicles. Through initial analysis, we discovered that variables like `price`, `odometer`, `year`, `condition`, and `manufacturer` had meaningful variation and appeared to be relevant for predicting outcomes. Outliers were identified in both the price and mileage distributions, and many columns contained missing or inconsistent data.

### 3. Data Preparation

Several cleaning and transformation steps were applied to get the dataset ready for modeling:

- Removed irrelevant columns (e.g., `id`, `VIN`, `region`, `state`) that do not contribute to pricing.
- Dropped rows with missing `price` or `year` values (critical to prediction).
- Filled missing categorical values (e.g., `condition`, `fuel`, `paint_color`) with `"unknown"`.
- Replaced missing `odometer` values with the **median**.
- Removed price outliers (below $1,000 or above $100,000).
- Filtered out vehicles with extreme mileage (over 300,000 miles).
- Engineered a new `vehicle_age` feature from `2023 - year`.
- Encoded categorical variables using `OneHotEncoder`.
- Used `ColumnTransformer` to process numeric and categorical columns for modeling.

### 4. Modeling

We trained and evaluated two models:
- `Linear Regression` (as a simple baseline)
- `Random Forest Regressor` (non-linear model with tunable parameters)

The dataset was split 80/20 into training and test sets. We used a pipeline to ensure all preprocessing steps were applied consistently during cross-validation and inference.

### 5. Evaluation

The `Random Forest Regressor` clearly outperformed `Linear Regression`, achieving:

- **Root Mean Squared Error (RMSE):** ~5,769  
- **Mean Absolute Error (MAE):** ~3,588  
- **R² Score:** ~0.84  

The `Linear Regression` model showed a much weaker R² of ~0.59, indicating that it failed to capture the non-linear and interaction effects present in the data.

These results highlight that Random Forest’s ability to model complex relationships makes it well-suited for this type of pricing task. The model’s strong performance indicates that key predictors like `odometer`, `vehicle age`, `condition`, and `manufacturer` are effective for estimating price.

### 6. Deployment

The final model can now be deployed to assist used car dealerships in evaluating inventory and setting prices more accurately. Insights from the analysis—such as the importance of mileage and vehicle condition—can be used by dealers even without full model deployment. The model could be integrated into an internal tool, pricing dashboard, or inventory management system for real-time price recommendations.

## Technologies Used

- Python 3.x  
- Jupyter Notebook  
- Pandas, NumPy, scikit-learn, Seaborn, Matplotlib  
- Machine learning models: `LinearRegression`, `RandomForestRegressor`  
- Preprocessing tools: `Pipeline`, `OneHotEncoder`, `ColumnTransformer`

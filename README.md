# What Drives the Price of a Car?

## Used Car Price Prediction

This project focuses on identifying the key factors that influence used car prices and building a machine learning model to support pricing decisions for used car dealerships. The dataset includes over 370,000 vehicle listings, and the analysis was structured using the CRISP-DM framework to ensure a methodical approach to solving the problem.

## Objective

The goal was to determine which features most significantly impact used car prices and to develop a predictive model that can assist dealerships in setting competitive and data-driven prices.

## Methodology

The project followed the CRISP-DM process, which guided each stage of the workflow from business understanding to deployment.

### 1. Business Understanding

The business goal was to help a used car dealership price its inventory more accurately based on historical data. The task was framed as a supervised regression problem, where the target variable was the car's listed price.

### 2. Data Understanding

Initial exploration of the dataset revealed that variables such as `price`, `odometer`, `year`, `condition`, and `manufacturer` appeared to be important in determining a vehicle’s value. There were also outliers and missing values in several columns that required further attention. A closer look showed that extreme price and mileage values could distort model performance if left untreated.

### 3. Data Preparation

The dataset was cleaned and prepared with the following steps:

- Dropped irrelevant columns like `id`, `VIN`, `region`, and `state` that do not contribute to price prediction.
- Removed rows missing critical fields like `price` or `year`.
- Replaced missing categorical fields (e.g., `fuel`, `condition`, `paint_color`) with `"unknown"`.
- Filled missing `odometer` values using the median.
- Filtered out vehicles with prices below $1,000 or above $100,000.
- Limited `odometer` values to a maximum of 300,000 miles.
- Created a new `vehicle_age` feature calculated as `2023 - year`.
- Categorical features were encoded using `OneHotEncoder` within a `ColumnTransformer` to prepare the data for `scikit-learn` models.

### 4. Modeling

Two models were developed and compared:

- **Linear Regression** was used as a baseline model for simplicity and interpretability.
- **Random Forest Regressor** was implemented to capture non-linear interactions and feature importance.

The data was split into 80% training and 20% testing. Both models were wrapped in a pipeline with preprocessing steps to ensure consistent handling of features.

### 5. Evaluation

The Random Forest model significantly outperformed the Linear Regression model:

- **Random Forest**
  - RMSE: ~5,769
  - MAE: ~3,588
  - R²: ~0.84

- **Linear Regression**
  - RMSE: ~9,225
  - MAE: ~6,469
  - R²: ~0.59

These results showed that the Random Forest model could explain a much larger proportion of the variance in car prices and was better at capturing the non-linear relationships among features. The most influential predictors included `odometer`, `vehicle age`, `condition`, and `manufacturer`.

### 6. Deployment

The final model is suitable for deployment in a dealership setting to provide pricing recommendations based on incoming vehicle data. It can be integrated into an inventory management system or used to flag inconsistencies in manual pricing decisions. The insights uncovered during the modeling phase also highlight which vehicle attributes should receive the most attention during appraisal and resale.

## Technologies Used

- Python 3.x
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (for modeling and preprocessing)
- RandomForestRegressor, LinearRegression, Pipeline, OneHotEncoder, ColumnTransformer

## Key Insights

- Higher mileage and older vehicles tend to sell for significantly less.
- Vehicle `condition` and `manufacturer` are also strong indicators of price.
- Tree-based models like Random Forest are well suited for this task, given the mix of categorical and numerical features.
- Preprocessing steps such as handling missing values and encoding categories had a major impact on final model performance.

## Future Work

- Add more detailed geographic or regional information.
- Experiment with advanced models like XGBoost or LightGBM.
- Build a web-based interface or dashboard for pricing recommendations.

# -*- coding: utf-8 -*-
"""
Air Quality Modeling – DS3000 Project
-------------------------------------

This script does the following:

1. Loads the EPA annual air quality dataset `annual_conc_by_monitor_2024.csv`.
2. Aggregates the raw monitor-level data into a site-level dataset
   (site = State + County + City + Lat + Lon + Year).
3. Performs exploratory analysis:
   - Distribution of PM2.5
   - Class counts for PM2.5 quartile-based risk labels
   - Correlation heatmap between pollutants and weather variables
4. Trains regression models to predict numeric PM2.5:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Trains classification models to predict PM2.5 risk category:
   - Logistic Regression
   - SVM (RBF kernel)
   - Neural Network (MLP)
6. Uses SHAP to interpret feature importance for the Random Forest regression model.

Assumption:
-----------
The file `annual_conc_by_monitor_2024.csv` is in the same folder as this script.
"""

import warnings
warnings.filterwarnings("ignore")  # Hide non-critical warnings to keep the console output clean

import os
import sys
import subprocess

# ========== Optional: auto-install shap if missing ==========
# This lets the script run even if shap isn't already installed in the environment.
try:
    import shap
except ImportError:
    print("shap not found. Installing shap...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities for preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, classification_report
)

from scipy import sparse

# Basic plotting style configuration
plt.rcParams["figure.figsize"] = (10, 6)
sns.set(style="whitegrid")


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the raw EPA annual concentrations dataset.
    """
    # Make sure the file actually exists before trying to read it
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find file: {data_path}")

    # Read the CSV into a DataFrame
    raw = pd.read_csv(data_path)
    print("Raw shape (rows, columns):", raw.shape)
    return raw


def build_site_level_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a site-level dataset by pivoting pollutants and weather variables.
    Each row = unique site (State, County, City, Lat, Lon) + Year.
    Columns = selected pollutants and weather variables.
    """
    # Keep only the columns needed for analysis and aggregation
    cols_needed = [
        "State Name", "County Name", "City Name",
        "Latitude", "Longitude",
        "Year",
        "Parameter Name",
        "Arithmetic Mean"
    ]
    df = raw[cols_needed].copy()

    # Pollutants and weather-like variables we care about
    selected_params = [
        "Ozone",
        "PM2.5 - Local Conditions",
        "PM10 - LC",
        "PM10 Total 0-10um STP",
        "Carbon monoxide",
        "Nitrogen dioxide (NO2)",
        "Sulfur dioxide",
        "Relative Humidity ",
        "Outdoor Temperature",
        "Average Ambient Temperature",
        "Average Ambient Pressure",
        "Wind Speed - Resultant",
        "Wind Direction - Resultant"
    ]

    # Keep only rows for these selected parameters
    df_sel = df[df["Parameter Name"].isin(selected_params)].copy()

    # Pivot so that each site/year becomes one row and each parameter becomes a column
    site_df = df_sel.pivot_table(
        index=["State Name", "County Name", "City Name", "Latitude", "Longitude", "Year"],
        columns="Parameter Name",
        values="Arithmetic Mean",
        aggfunc="mean"  # if there are multiple monitors, take the mean across them
    ).reset_index()

    # Remove the column index name left over from pivot
    site_df.columns.name = None

    print("Site-level dataset shape:", site_df.shape)
    return site_df


def prepare_modeling_data(site_df: pd.DataFrame):
    """
    Prepare the dataset for modeling:
    - Define target (PM2.5 numeric)
    - Define features (pollutants + weather + coordinates)
    - Create PM2.5 classification labels with quartiles
    """
    TARGET_COL = "PM2.5 - Local Conditions"

    # Drop any rows where the target is missing, since we can't train on those
    data = site_df.dropna(subset=[TARGET_COL]).copy()

    # All feature columns except ID-like columns and the numeric target
    feature_cols = [
        c for c in data.columns
        if c not in ["State Name", "County Name", "City Name", "Year", TARGET_COL]
    ]

    # Here we treat all these selected columns as numeric features
    numeric_features = feature_cols

    # We'll one-hot encode only State Name as a simple categorical feature
    categorical_features = ["State Name"]

    # Create quartile-based risk classes for PM2.5
    data["PM25_Class"] = pd.qcut(
        data[TARGET_COL],
        q=4,
        labels=["Low", "Moderate", "High", "Very High"]
    )

    print("Data for modeling (after dropping missing PM2.5):", data.shape)
    print(data[[TARGET_COL, "PM25_Class"]].head())

    return data, TARGET_COL, feature_cols, numeric_features, categorical_features


def simple_eda(data: pd.DataFrame, TARGET_COL: str):
    """
    Simple EDA:
    - Distribution of PM2.5
    - Class counts for PM2.5 quartile-based labels
    - Correlation heatmap between pollutants and weather variables
    """
    # Plot the distribution of the PM2.5 numeric values
    sns.histplot(data[TARGET_COL], bins=30, kde=True)
    plt.title("Distribution of Annual PM2.5 (Local Conditions)")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.show()

    # Show how many sites fall into each PM2.5 class
    sns.countplot(
        x="PM25_Class",
        data=data,
        order=["Low", "Moderate", "High", "Very High"]
    )
    plt.title("PM2.5 Risk Classes (Quartiles)")
    plt.xlabel("PM2.5 Class")
    plt.ylabel("Count")
    plt.show()

    # Correlation heatmap for numeric pollutants + weather columns
    corr_cols = [
        c for c in data.columns
        if c not in ["State Name", "County Name", "City Name", "Year", "PM25_Class"]
    ]
    corr = data[corr_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap: Pollutants and Weather Variables")
    plt.show()


def build_preprocessor(numeric_features, categorical_features):
    """
    Build a ColumnTransformer that:
    - Imputes + scales numeric features
    - Imputes + one-hot encodes categorical features
    """
    # Numeric pipeline: fill missing values with the median, then standardize
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: fill missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine numeric and categorical transforms into one preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def run_regression_models(data, TARGET_COL, feature_cols, categorical_features, preprocessor):
    """
    Train and evaluate regression models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    """
    # Build feature matrix X and target vector y for regression
    X_reg = data[feature_cols + categorical_features]
    y_reg = data[TARGET_COL]

    # Split into training and test sets
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Define a small set of regression models to compare
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    reg_results = {}

    for name, model in reg_models.items():
        # Each model gets its own pipeline with the shared preprocessor
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        # Fit on training data
        pipe.fit(X_train_reg, y_train_reg)

        # Predict on test data
        preds = pipe.predict(X_test_reg)

        # Standard regression metrics
        r2 = r2_score(y_test_reg, preds)
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))

        reg_results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

    print("\n=== Regression Results ===")
    print(pd.DataFrame(reg_results).T)

    # Return the split sets for later use in SHAP
    return X_train_reg, y_train_reg, X_test_reg, y_test_reg


def run_classification_models(data, feature_cols, categorical_features, preprocessor):
    """
    Train and evaluate classification models:
    - Logistic Regression
    - SVM
    - Neural Network (MLP)
    """
    # Build X and y for classification
    X_clf = data[feature_cols + categorical_features]
    y_clf = data["PM25_Class"]

    # Stratified split to keep class proportions balanced across train/test
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Set up the models we want to compare
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=500, multi_class="multinomial"),
        "SVM": SVC(kernel="rbf", probability=True),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32),
                                        max_iter=500, random_state=42)
    }

    # LabelEncoder is used to convert the string labels to integers for AUC
    le = LabelEncoder()
    le.fit(y_clf)

    clf_results = {}

    for name, model in clf_models.items():
        # Pipeline with shared preprocessor
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        # Fit the classifier
        pipe.fit(X_train_clf, y_train_clf)

        # Predictions and probabilities on test set
        preds = pipe.predict(X_test_clf)
        probs = pipe.predict_proba(X_test_clf)

        # Basic performance metrics
        acc = accuracy_score(y_test_clf, preds)
        f1 = f1_score(y_test_clf, preds, average="weighted")
        auc = roc_auc_score(
            le.transform(y_test_clf),
            probs,
            multi_class="ovr"
        )

        clf_results[name] = {"Accuracy": acc, "F1_weighted": f1, "AUC_ovr": auc}

        print(f"\n=== {name} ===")
        print(classification_report(y_test_clf, preds))

    print("\n=== Classification Results ===")
    print(pd.DataFrame(clf_results).T)


def shap_analysis(preprocessor, numeric_features, categorical_features,
                  X_train_reg, y_train_reg):
    """
    Run SHAP analysis on a Random Forest regressor fit
    on preprocessed training data.
    """
    # Fit the preprocessor on the training data and transform X
    X_train_reg_proc = preprocessor.fit_transform(X_train_reg)

    # If the preprocessed data is sparse, convert it to a dense array
    if sparse.issparse(X_train_reg_proc):
        X_train_reg_proc = X_train_reg_proc.toarray()

    # Make sure the dtype is float for SHAP and RandomForest
    X_train_reg_proc = X_train_reg_proc.astype(float)

    # Train a Random Forest on the processed data
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train_reg_proc, y_train_reg)

    # Build feature names after preprocessing: numeric + one-hot encoded categorical
    num_names = numeric_features
    cat_ohe_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    feature_names = num_names + cat_ohe_names  # Handy if you want to inspect later

    # Set up SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_train_reg_proc)

    # Beeswarm plot shows how each feature influences the prediction distribution
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Feature Importance – Random Forest (PM2.5 Regression)")
    plt.xlabel("SHAP value (impact on model output)")
    plt.show()


def main():
    # Path to dataset (assumed to be in the same folder as this script)
    DATA_PATH = "annual_conc_by_monitor_2024.csv"

    # 1. Load raw monitor-level data
    raw = load_data(DATA_PATH)

    # 2. Aggregate monitor-level records into a site-level dataset
    site_df = build_site_level_dataset(raw)

    # 3. Prepare data for both regression and classification tasks
    data, TARGET_COL, feature_cols, numeric_features, categorical_features = prepare_modeling_data(site_df)

    # 4. Run some basic EDA plots
    simple_eda(data, TARGET_COL)

    # 5. Build a shared preprocessor for numeric + categorical features
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 6. Train and evaluate regression models on PM2.5 numeric values
    X_train_reg, y_train_reg, X_test_reg, y_test_reg = run_regression_models(
        data, TARGET_COL, feature_cols, categorical_features, preprocessor
    )

    # 7. Train and evaluate classification models on PM2.5 risk classes
    run_classification_models(data, feature_cols, categorical_features, preprocessor)

    # 8. Use SHAP to interpret the Random Forest regression model
    shap_analysis(preprocessor, numeric_features, categorical_features, X_train_reg, y_train_reg)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()

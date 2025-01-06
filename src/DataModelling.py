import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import configparser
import os
import joblib
import sys


def read_config(config_file):
    """Reads configuration from a file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def load_dataset(file_path):
    """Loads dataset from CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Reading file from {file_path}")
        return data
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please check if the file path in the config file is correct.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def select_features(data, features, target):
    """Selects specified features and the target from the dataset."""
    if not all(col in data.columns for col in features + [target]):
        missing_cols = [
            col for col in features +
            [target] if col not in data.columns]
        print(f"Missing columns in the dataset: {missing_cols}")
        sys.exit(1)
    return data[features + [target]]


def preprocess_data(data, features, target):
    """Handles missing values and splits the data into features and target."""
    data = data.dropna(subset=features + [target])
    X = data[features]
    y = data[target]
    return X, y


def split_data(X, y, test_size, val_size):
    """Splits the data into training, validation, and testing sets."""
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)
        return X_train, y_train, X_val, y_val, X_test, y_test
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        sys.exit(1)


def standardize_data(X_train, X_val, X_test):
    """Standardizes the features."""
    try:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test
    except Exception as e:
        print(f"Error during data standardization: {e}")
        sys.exit(1)


def apply_undersampling(X_train, y_train):
    """Applies undersampling to the training data."""
    try:
        undersampler = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = undersampler.fit_resample(
            X_train, y_train)
        return X_train_under, y_train_under
    except Exception as e:
        print(f"Error during undersampling: {e}")
        sys.exit(1)


def tune_model_parameters(model, param_grid, X_train, y_train):
    """Performs hyperparameter tuning for a given model using GridSearchCV."""
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters found for {model.__class__.__name__}:")
        print(grid_search.best_params_)
        return grid_search.best_params_
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        sys.exit(1)


def train_model(model, X_train, y_train):
    """Trains a model with the given training data."""
    try:
        model.fit(X_train, y_train)
        print(f"Model trained with parameters: {model}")
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)


def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluates model performance on validation and test sets."""
    try:
        val_pred = model.predict(X_val)
        val_pred_prob = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict(X_test)
        test_pred_prob = model.predict_proba(X_test)[:, 1]

        print(f"Validation Report for {model.__class__.__name__}:")
        print(classification_report(y_val, val_pred))
        print("Confusion Matrix (Validation Set):")
        print(confusion_matrix(y_val, val_pred))
        print(
            "Accuracy Score (Validation Set):",
            accuracy_score(
                y_val,
                val_pred))
        print(
            "ROC-AUC Score (Validation Set):",
            roc_auc_score(
                y_val,
                val_pred_prob))

        print(f"Test Report for {model.__class__.__name__}:")
        print(classification_report(y_test, test_pred))
        print("Confusion Matrix (Test Set):")
        print(confusion_matrix(y_test, test_pred))
        print("Accuracy Score (Test Set):", accuracy_score(y_test, test_pred))
        print(
            "ROC-AUC Score (Test Set):",
            roc_auc_score(
                y_test,
                test_pred_prob))
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        sys.exit(1)


def save_model(model, directory, filename):
    """Saves trained model to specified directory."""
    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")
        sys.exit(1)


def main(config_file):
    """Main function to execute the entire machine learning pipeline."""
    try:
        config = read_config(config_file)

        file_path = config.get("PARAMETERS", "PROCESSED_FILE")
        test_size = float(config.get("PARAMETERS", "TEST_SIZE"))
        val_size = float(config.get("PARAMETERS", "VAL_SIZE"))
        target_column = config.get("PARAMETERS", "TARGET_COLUMN")
        model_dir = config.get("PARAMETERS", "MODEL_DIR")

        data = load_dataset(file_path)
        if data is None:
            return

        features = [
            'delivery_duration',
            'payment_value',
            'product_photos_qty',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
            'on_time',
            "customer_city",
            "customer_state",
            "product_category_name_english",
            "payment_type"
        ]

        data = select_features(data, features, target_column)
        X, y = preprocess_data(data, features, target_column)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y, test_size, val_size)
        X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)
        X_train_under, y_train_under = apply_undersampling(X_train, y_train)

        # Tuning and training RandomForest model
        rf_param_grid = {
            'max_depth': [None, 10, 20],
            'n_estimators': [50, 100, 150]
        }
        rf_best_params = tune_model_parameters(
            RandomForestClassifier(
                random_state=42),
            rf_param_grid,
            X_train_under,
            y_train_under)
        rf_model = train_model(
            RandomForestClassifier(
                **rf_best_params,
                random_state=42),
            X_train_under,
            y_train_under)
        evaluate_model(rf_model, X_val, y_val, X_test, y_test)
        save_model(rf_model, model_dir, "rf_model.pkl")

        # Training and evaluating Logistic Regression model
        logreg = train_model(
            LogisticRegression(
                random_state=42,
                class_weight='balanced'),
            X_train_under,
            y_train_under)
        evaluate_model(logreg, X_val, y_val, X_test, y_test)
        save_model(logreg, model_dir, "logreg_model.pkl")

        # Tuning and training XGBoost model
        xgb_param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_best_params = tune_model_parameters(
            XGBClassifier(
                random_state=42),
            xgb_param_grid,
            X_train_under,
            y_train_under)
        xgb_model = train_model(
            XGBClassifier(
                **xgb_best_params,
                random_state=42),
            X_train_under,
            y_train_under)
        evaluate_model(xgb_model, X_val, y_val, X_test, y_test)
        save_model(xgb_model, model_dir, "xgb_model.pkl")

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main("parameter.env")

<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

# Define path to the preprocessed data
preprocessed_data_dir = "../preprocessed_data/"

# Define the datasets to train on
datasets = ["FD001", "FD002", "FD003", "FD004"]

# Define function to load preprocessed data
def load_preprocessed_data(dataset):
    # Load train and test datasets
    train_data = pd.read_csv(os.path.join(preprocessed_data_dir, f"train_{dataset}_processed.csv"))
    test_data = pd.read_csv(os.path.join(preprocessed_data_dir, f"test_{dataset}_processed.csv"))
    return train_data, test_data

# Function to train and evaluate the model
def train_evaluate_model(train_data, test_data):
    # Define feature columns and target column
    feature_columns = [col for col in train_data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL', 'true_RUL']]
    X_train = train_data[feature_columns]
    y_train = train_data['RUL']

    X_test = test_data[feature_columns]
    y_test = test_data['true_RUL'] if 'true_RUL' in test_data.columns else test_data['RUL']

    # Split the training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train_split, y_train_split)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R2 Score: {r2}")

    return model

# Loop over all datasets and train models
for dataset in datasets:
    print(f"Training and evaluating model for {dataset}")
    
    # Load preprocessed data
    train_data, test_data = load_preprocessed_data(dataset)
    
    # Train and evaluate the model
    model = train_evaluate_model(train_data, test_data)

    # Save the trained model for future use
    model_path = os.path.join("models", f"random_forest_{dataset}.pkl")
    os.makedirs("models", exist_ok=True)
    pd.to_pickle(model, model_path)
    
    print(f"Model for {dataset} saved at {model_path}\n")
=======
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

# Define path to the preprocessed data
preprocessed_data_dir = "../preprocessed_data/"

# Define the datasets to train on
datasets = ["FD001", "FD002", "FD003", "FD004"]

# Define function to load preprocessed data
def load_preprocessed_data(dataset):
    # Load train and test datasets
    train_data = pd.read_csv(os.path.join(preprocessed_data_dir, f"train_{dataset}_processed.csv"))
    test_data = pd.read_csv(os.path.join(preprocessed_data_dir, f"test_{dataset}_processed.csv"))
    return train_data, test_data

# Function to train and evaluate the model
def train_evaluate_model(train_data, test_data):
    # Define feature columns and target column
    feature_columns = [col for col in train_data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL', 'true_RUL']]
    X_train = train_data[feature_columns]
    y_train = train_data['RUL']

    X_test = test_data[feature_columns]
    y_test = test_data['true_RUL'] if 'true_RUL' in test_data.columns else test_data['RUL']

    # Split the training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train_split, y_train_split)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R2 Score: {r2}")

    return model

# Loop over all datasets and train models
for dataset in datasets:
    print(f"Training and evaluating model for {dataset}")
    
    # Load preprocessed data
    train_data, test_data = load_preprocessed_data(dataset)
    
    # Train and evaluate the model
    model = train_evaluate_model(train_data, test_data)

    # Save the trained model for future use
    model_path = os.path.join("models", f"random_forest_{dataset}.pkl")
    os.makedirs("models", exist_ok=True)
    pd.to_pickle(model, model_path)
    
    print(f"Model for {dataset} saved at {model_path}\n")
>>>>>>> 2448e220b26ab4380d18f13d76703523981946b8

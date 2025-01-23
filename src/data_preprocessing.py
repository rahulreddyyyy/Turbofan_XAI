<<<<<<< HEAD

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

data_dir = "data/"
output_dir = "preprocessed_data/"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "FD001": {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"},
    "FD002": {"train": "train_FD002.txt", "test": "test_FD002.txt", "rul": "RUL_FD002.txt"},
    "FD003": {"train": "train_FD003.txt", "test": "test_FD003.txt", "rul": "RUL_FD003.txt"},
    "FD004": {"train": "train_FD004.txt", "test": "test_FD004.txt", "rul": "RUL_FD004.txt"}
}

def load_data(file_path):
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    columns += [f'sensor_measurement_{i}' for i in range(1, 24)]
    data = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    data.drop(columns=["sensor_measurement_22", "sensor_measurement_23"], inplace=True)  # Drop extra columns
    return data

def normalize_data(data, columns_to_scale):
    scaler = MinMaxScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data, scaler

def compute_rul(data):
    rul = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    data = data.merge(rul, on=['unit_number'], how='left')
    data['RUL'] = data['max_cycle'] - data['time_in_cycles']
    data.drop(columns=['max_cycle'], inplace=True)
    return data

def preprocess_dataset(train_path, test_path, rul_path=None):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    train_data = compute_rul(train_data)
    
    columns_to_scale = [f'sensor_measurement_{i}' for i in range(1, 22)] + ['op_setting_1', 'op_setting_2', 'op_setting_3']
    train_data, scaler = normalize_data(train_data, columns_to_scale)
    test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
    
    if rul_path:
        true_rul = pd.read_csv(rul_path, header=None, names=['RUL'])
        test_data = test_data.groupby('unit_number').tail(1).reset_index(drop=True)
        test_data['true_RUL'] = true_rul['RUL']
    
    return train_data, test_data

for dataset, paths in datasets.items():
    print(f"Processing dataset {dataset}")
    train_data, test_data = preprocess_dataset(
        os.path.join(data_dir, paths['train']),
        os.path.join(data_dir, paths['test']),
        os.path.join(data_dir, paths['rul'])
    )
    
    train_data.to_csv(os.path.join(output_dir, f"train_{dataset}_processed.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, f"test_{dataset}_processed.csv"), index=False)

    print(f"Dataset {dataset} processed and saved.\n")
=======

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

data_dir = "data/"
output_dir = "preprocessed_data/"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "FD001": {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"},
    "FD002": {"train": "train_FD002.txt", "test": "test_FD002.txt", "rul": "RUL_FD002.txt"},
    "FD003": {"train": "train_FD003.txt", "test": "test_FD003.txt", "rul": "RUL_FD003.txt"},
    "FD004": {"train": "train_FD004.txt", "test": "test_FD004.txt", "rul": "RUL_FD004.txt"}
}

def load_data(file_path):
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    columns += [f'sensor_measurement_{i}' for i in range(1, 24)]
    data = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    data.drop(columns=["sensor_measurement_22", "sensor_measurement_23"], inplace=True)  # Drop extra columns
    return data

def normalize_data(data, columns_to_scale):
    scaler = MinMaxScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data, scaler

def compute_rul(data):
    rul = data.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    data = data.merge(rul, on=['unit_number'], how='left')
    data['RUL'] = data['max_cycle'] - data['time_in_cycles']
    data.drop(columns=['max_cycle'], inplace=True)
    return data

def preprocess_dataset(train_path, test_path, rul_path=None):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    train_data = compute_rul(train_data)
    
    columns_to_scale = [f'sensor_measurement_{i}' for i in range(1, 22)] + ['op_setting_1', 'op_setting_2', 'op_setting_3']
    train_data, scaler = normalize_data(train_data, columns_to_scale)
    test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
    
    if rul_path:
        true_rul = pd.read_csv(rul_path, header=None, names=['RUL'])
        test_data = test_data.groupby('unit_number').tail(1).reset_index(drop=True)
        test_data['true_RUL'] = true_rul['RUL']
    
    return train_data, test_data

for dataset, paths in datasets.items():
    print(f"Processing dataset {dataset}")
    train_data, test_data = preprocess_dataset(
        os.path.join(data_dir, paths['train']),
        os.path.join(data_dir, paths['test']),
        os.path.join(data_dir, paths['rul'])
    )
    
    train_data.to_csv(os.path.join(output_dir, f"train_{dataset}_processed.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, f"test_{dataset}_processed.csv"), index=False)

    print(f"Dataset {dataset} processed and saved.\n")
>>>>>>> 2448e220b26ab4380d18f13d76703523981946b8

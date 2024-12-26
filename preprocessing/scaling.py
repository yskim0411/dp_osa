import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def scaling(data, cat_cols, num_cols, task, case, save_data_path):
    total_cols = cat_cols + num_cols
    data = data[total_cols]
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
    
    data_scaled.to_csv(f'{save_data_path}/{task}/{case}/scaled_data.csv', index = False)
    
    return data_scaled
import pandas as pd
import numpy as np
import os

def labeling(data, task, save_data_path):
    if task == "None":
        print("Try Again")
    
    elif task == "posa":
        print(f"data shape: {data.shape}")
        
        value = 0.5
        data = data[data['AHI_total'] >= 5]
        
        condition = data['RDI_idx'] >= 5
        
        subcondition_1 = (data['NREM_lat_min'] == 0) & (data['REM_lat_min'] == 0)
        data = data[~(condition & subcondition_1)]
        
        subcondition_2 = data['AHI_lat'] / data['AHI_sup'] < value
        data.loc[condition & subcondition_2, 'pOSA'] = 1
        
        subcondition_3 = data['AHI_lat'] / data['AHI_sup'] >= value
        data.loc[condition & subcondition_3, 'pOSA'] = 0
        
        data['pOSA'] = data['pOSA'].astype(int)
        data = data[data['pOSA'].isin([0, 1])]
    
    elif task == "remosa":
        print(f"data shape: {data.shape}")
        
        value = 2
        data = data[data['AHI_total'] >= 5]
        
        condition_1 = (data['AHI_REM'] / data['AHI_NREM'] < value) & (data['AHI_REM'] != 0)
        data.loc[condition_1, 'REM_OSA'] = 0
        
        condition_2 = (data['AHI_REM'] / data['AHI_NREM'] >= value) & (data['AHI_REM'] != 0)
        data.loc[condition_2, 'REM_OSA'] = 1
        
        data['REM_OSA'] = data['REM_OSA'].astype(int)
        data = data[data['REM_OSA'].isin([0, 1])]
        
    label_save(data, task, save_data_path)
    
    return data


def label_save(data, task, save_data_path):
    if task == "None":
        print("Try Again")
    
    elif task == "posa":
        label = data['pOSA']
        print("pOSA classification")
        
    elif task == "remosa":
        label = data['REM_OSA']
        print("REM OSA classification")
        
    if not os.path.exists(f'{save_data_path}/{task}'):
        os.makedirs(f'{save_data_path}/{task}')
        
    label.to_csv(f'{save_data_path}/{task}/label.csv', index=False)
    
    label_counts = label.value_counts()
    
    print("all_label shape", label.shape)
    print(label_counts)
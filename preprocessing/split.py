import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_split(data, scaled_data, query, test, task, case, save_data_path):
    test_index_path = f"{save_data_path}/{task}/test_index.csv"
    query_index_path = f"{save_data_path}/{task}/query_index.csv"
    
    if os.path.exists(test_index_path) and os.path.exists(query_index_path):
        test_index = pd.read_csv(test_index_path).iloc[:, 0].tolist()
        query_index = pd.read_csv(query_index_path).iloc[:, 0].tolist()

        test_data = data.loc[test_index]
        test_data.to_csv(f"{save_data_path}/{task}/{case}/test.csv", index=False)

        train_data = data.drop(test_index)
        train_data.to_csv(f"{save_data_path}/{task}/{case}/train.csv", index=False)
        
        query_data = train_data.loc[query_index]
        query_data.to_csv(f"{save_data_path}/{task}/{case}/query.csv", index=False)

        source_data = train_data.drop(query_index)
        source_data.to_csv(f"{save_data_path}/{task}/{case}/source.csv", index=False)
        
        if task == "posa":
            test_label = test_data['pOSA']
            test_label.to_csv(f"{save_data_path}/{task}/{case}/test_label.csv", index=False)
            
            train_label = train_data['pOSA']
            train_label.to_csv(f"{save_data_path}/{task}/{case}/train_label.csv", index=False)
            
            query_label = query_data['pOSA']
            query_label.to_csv(f"{save_data_path}/{task}/{case}/query_label.csv", index=False)
            
        elif task == "remosa":
            test_label = test_data['REM_OSA']
            test_label.to_csv(f"{save_data_path}/{task}/{case}/test_label.csv", index=False)
            
            train_label = train_data['REM_OSA']
            train_label.to_csv(f"{save_data_path}/{task}/{case}/train_label.csv", index=False)
            
            query_label = query_data['REM_OSA']
            query_label.to_csv(f"{save_data_path}/{task}/{case}/query_label.csv", index=False)

        query_data = train_data.loc[query_index]
        query_data.to_csv(f"{save_data_path}/{task}/{case}/query.csv", index=False)

        source_data = train_data.drop(query_index)
        source_data.to_csv(f"{save_data_path}/{task}/{case}/source.csv", index=False)
    else:
        test_data = data.sample(n=test, random_state=42)
        test_index = test_data.index.tolist()
        test_data.to_csv(f"{save_data_path}/{task}/{case}/test.csv", index=False)
        pd.DataFrame(test_index).to_csv(test_index_path, index=False)

        train_data = data.drop(test_index)
        train_data.to_csv(f"{save_data_path}/{task}/{case}/train.csv", index=False)
        
        query_data = train_data.sample(n=query, random_state=42)
        query_index = query_data.index.tolist()
        query_data.to_csv(f"{save_data_path}/{task}/{case}/query.csv", index=False)
        pd.DataFrame(query_index).to_csv(query_index_path, index=False)

        source_data = train_data.drop(query_index)
        source_data.to_csv(f"{save_data_path}/{task}/{case}/source.csv", index=False)
        
        if task == "posa":
            test_label = test_data['pOSA']
            test_label.to_csv(f"{save_data_path}/{task}/{case}/test_label.csv", index=False)
            
            test_feature = test_data.drop(columns=['pOSA'], inplace=True)
            test_feature.to_csv(f"{save_data_path}/{task}/{case}/test_feature.csv", index=False)

            train_label = train_data['pOSA']
            train_label.to_csv(f"{save_data_path}/{task}/{case}/train_label.csv", index=False)
            
            query_label = query_data['pOSA']
            query_label.to_csv(f"{save_data_path}/{task}/{case}/query_label.csv", index=False)
            
            query_data.drop(columns=['pOSA'], inplace=True)
            query_data.to_csv(f"{save_data_path}/{task}/{case}/query_feature.csv", index=False)
            
        elif task == "remosa":
            test_label = test_data['REM_OSA']
            test_label.to_csv(f"{save_data_path}/{task}/{case}/test_label.csv", index=False)
            
            test_feature = test_data.drop(columns=['REM_OSA'], inplace=True)
            test_feature.to_csv(f"{save_data_path}/{task}/{case}/test_feature.csv", index=False)
            
            train_label = train_data['REM_OSA']
            train_label.to_csv(f"{save_data_path}/{task}/{case}/train_label.csv", index=False)
            
            query_label = query_data['REM_OSA']
            query_label.to_csv(f"{save_data_path}/{task}/{case}/query_label.csv", index=False)
            
            query_feature = query_data.drop(columns=['REM_OSA'], inplace=True)
            query_feature.to_csv(f"{save_data_path}/{task}/{case}/query_feature.csv", index=False)

        
    return train_data, test_data


def split(data, task, case, save_data_path):
    test_index_path = f"{save_data_path}/{task}/test_index.csv"
    
    test_index = pd.read_csv(test_index_path).iloc[:, 0].tolist()

    scaled_test_data = data.loc[test_index]
    scaled_test_data.to_csv(f"{save_data_path}/{task}/{case}/scaled_test.csv", index=False)

    scaled_train_data = data.drop(test_index)
    scaled_train_data.to_csv(f"{save_data_path}/{task}/{case}/scaled_train.csv", index=False)
    
    

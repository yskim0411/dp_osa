import pandas as pd
import pickle
import os

def save_pickle(save_data_path, task, case):
    src_dir = f'{save_data_path}/{task}/{case}'
    dest_dir = f'{save_data_path}/{task}/{case}/pickle'
    
    csv_file_list = ['scaled_test.csv',
                     'scaled_train.csv',
                     'test_label.csv',
                     'train_label.csv']
    
    # 디렉토리가 존재하지 않는다면 생성
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for csv_file in csv_file_list:
        # create path for csv file
        csv_path = os.path.join(src_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        
        # read csv file
        data = pd.read_csv(csv_path)
        
        # create path for pickle
        pickle_file = os.path.splitext(csv_file)[0] + '.pkl'
        pickle_path = os.path.join(dest_dir, pickle_file)
        
        # save pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
            
    label_save_pickle(save_data_path, task, case)
    
    print(f'Conversion completed. Pickle files saved in {dest_dir}.')
    

def label_save_pickle(save_data_path, task, case):
    src_dir = f'{save_data_path}/{task}'
    dest_dir = f'{save_data_path}/{task}/{case}/pickle'
    
    csv_file_list = ['label.csv']
    
    # 디렉토리가 존재하지 않는다면 생성
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for csv_file in csv_file_list:
        # create path for csv file
        csv_path = os.path.join(src_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        
        # read csv file
        data = pd.read_csv(csv_path)
        
        # create path for pickle
        pickle_file = os.path.splitext(csv_file)[0] + '.pkl'
        pickle_path = os.path.join(dest_dir, pickle_file)
        
        # save pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
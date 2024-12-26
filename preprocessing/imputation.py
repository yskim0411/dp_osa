import sys
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers
from hyperimpute import logger
import torch
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(42)

device = torch.device("cpu")
np.set_printoptions(threshold=sys.maxsize)

def hyperimpute(data, cat_cols, num_cols, task, case, save_data_path):
    logger.add(sink=sys.stderr, level='DEBUG')
    logger.add(sink=sys.stderr, level='INFO')
    plugin = Imputers().get('hyperimpute')
    
    total_cols = cat_cols + num_cols
    total_data = data[total_cols]
    
    total_data = plugin.fit_transform(total_data.copy())
    total_data = pd.DataFrame(total_data, columns=total_cols)
    
    if not os.path.exists(f'{save_data_path}/{task}/{case}'):
        os.makedirs(f'{save_data_path}/{task}/{case}')
        
    total_data.to_csv(f'{save_data_path}/{task}/{case}/im_total_data.csv', index=False)
    
    all_label_path = f'{save_data_path}/{task}/all_label.csv'
    
    if os.path.exists(all_label_path):
        all_label = pd.read_csv(all_label_path)
        combined_data = pd.concat([total_data, all_label], axis=1)
        
        # 새로운 CSV로 저장
        combined_data.to_csv(f'{save_data_path}/{task}/{case}/posa.csv', index=False)
    else:
        logger.error(f"all_label.csv not found at {all_label_path}")
    
    return total_data, combined_data
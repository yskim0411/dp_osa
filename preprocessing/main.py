import pandas as pd 
import labeling
import seperate
import imputation
import scaling
import split
import save_pickle
import argparse
import os

parser = argparse.ArgumentParser(description = "Preprocessing Data.")
parser.add_argument("-t", "--task", type = str, default = "None", help = "posa, remosa")
parser.add_argument("-q", "--query", type = int, default = 100, help = "number of query data")
parser.add_argument("-tst", "--test", type = int, default = 50, help = "number of test data")

args = parser.parse_args()

# data_path = "OOO.xlsx"
# data = pd.read_excel(data_path)
# save_data_path = ""

# 카테고리를 적절한 시트 이름으로 변환
if args.task.lower() == 'posa':
    sheet_name = 'pOSA'
elif args.task.lower() == 'remosa':
    sheet_name = 'REMOSA' 
else:
    raise ValueError("Invalid category. Choose from 'posa', 'remosa'.")

# 데이터 불러오기
data = "/home/aix21202/dp_osa/241014_EUMC_final.xlsx"
data = pd.read_excel(data, sheet_name = sheet_name)
save_data_path = "/home/aix21202/dp_osa/data"

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

labeled_data = labeling.labeling(data, args.task, save_data_path)

for case in range(1, 5):
    if case == 1: 
        cat_cols, num_cols = seperate.sc1_cols()
    elif case == 2:
        cat_cols, num_cols = seperate.sc2_cols()
    elif case == 3:
        cat_cols, num_cols = seperate.sc3_cols()
    elif case == 4:
        cat_cols, num_cols = seperate.sc4_cols()

    imputation_feature, imputation_data = imputation.hyperimpute(labeled_data, cat_cols, num_cols, args.task, case, save_data_path)
    
    scaled_data = scaling.scaling(imputation_data, cat_cols, num_cols, args.task, case, save_data_path)
    
    train_data, test_data = split.data_split(imputation_data, scaled_data, args.query, args.test, args.task, case, save_data_path)
    split.split(scaled_data, args.task, case, save_data_path)
    
    save_pickle.save_pickle(save_data_path, args.task, case)
    
    print(f"Case {case} preprocessing completed.")
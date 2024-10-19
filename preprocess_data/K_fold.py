import pandas as pd
import numpy as np
import os
from tqdm import tqdm
def normalize_01(data):
    return (data - data.min()) / (data.max() - data.min())

# make data folder
data_dir = 'k_fold_data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

## Import training data (raw spectra and pre-processed spectra)
x_2d_snv_train = pd.read_csv("challenge_data/x_2d_snv_train.csv", index_col=0)
y_brix_train = pd.read_csv("challenge_data/y_brix_train.csv", index_col=0)
y_firm_train = pd.read_csv("challenge_data/y_firm_train.csv", index_col=0)

x_2d_snv_train = np.array(x_2d_snv_train)
y_brix_train = np.array(y_brix_train)
y_firm_train = np.array(y_firm_train)

x_2d_snv_train = normalize_01(x_2d_snv_train)
# split data into 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf.get_n_splits(x_2d_snv_train)
for fold, (train_index, test_index) in tqdm(enumerate(kf.split(x_2d_snv_train), start=1)):
    x_train, x_test = x_2d_snv_train[train_index], x_2d_snv_train[test_index]
    y_brix_train_fold, y_brix_test_fold = y_brix_train[train_index], y_brix_train[test_index]
    y_firm_train_fold, y_firm_test_fold= y_firm_train[train_index], y_firm_train[test_index]
    # save to csv 5 folds name as x_train_fold1.csv, x_test_fold1.csv, y_train_fold1.csv, y_test_fold1.csv
    pd.DataFrame(x_train).to_csv(f"{data_dir}x_train_fold{fold}.csv", index=False)
    pd.DataFrame(x_test).to_csv(f"{data_dir}x_test_fold{fold}.csv", index=False)
    pd.DataFrame(y_brix_train_fold).to_csv(f"{data_dir}y_brix_train_fold{fold}.csv", index=False)
    pd.DataFrame(y_brix_test_fold).to_csv(f"{data_dir}y_brix_test_fold{fold}.csv", index=False)
    pd.DataFrame(y_firm_train_fold).to_csv(f"{data_dir}y_firm_train_fold{fold}.csv", index=False)
    pd.DataFrame(y_firm_test_fold).to_csv(f"{data_dir}y_firm_test_fold{fold}.csv", index=False)

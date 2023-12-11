import numpy as np
import pandas as pd

sparse_feats = ['C' + str(i) for i in range(1, 27)]
dense_feats = ['I' + str(i) for i in range(1, 14)]
target_columns = ['label']
columns = target_columns + dense_feats + sparse_feats

def filter_na():
    chunksize = 2000000
    # data = pd.read_csv("./test.txt", sep='\t', names = columns)
    new_data = pd.DataFrame(columns=columns)
    for chunk in pd.read_csv("./train.txt", sep='\t', names = columns, chunksize=chunksize, iterator=True):
        loaded_data = chunk.dropna()
        new_data = pd.concat([new_data, loaded_data])
        print(loaded_data.shape, new_data.shape)
    print("loaded data")
    print("new_data shape", new_data.shape)
    new_data.to_csv('filtered_data.csv', index=False)


threshold = 5000
# threshold = 50
data = pd.read_csv("./filtered_data.csv")
for col in sparse_feats:
    vf = data[col].value_counts()
    print(col, vf.shape)
    rare_values = vf[vf < threshold].index.to_list()
    if len(rare_values) > 0:
        replace_v = rare_values[0]
        rare_values = {r: replace_v for r in rare_values}
        print("replace rare with:", replace_v, "; replace count:", np.sum(vf[vf < threshold].to_list()))
        data[col].replace(to_replace=rare_values, inplace=True)
        print("after", data[col].value_counts().shape)
data.to_csv('preprocessed_data.csv', index=False)
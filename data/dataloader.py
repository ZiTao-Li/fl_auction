import numpy as np
import pickle
import os
# import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

class CriteoLoader:
    def __init__(self, config, seed=100):
        np.random.seed(seed)
        dataset_name = config['dataset']
        assert dataset_name in ['criteo']
        data = pd.read_csv('./ctr_data/criteo/preprocessed_data.csv')
        self.sparse_feats = ['C' + str(i) for i in range(1, 27)]
        self.dense_feats = ['I' + str(i) for i in range(1, 14)]
        self.train_feats = self.dense_feats + self.sparse_feats

        # convert categorical features to one-hot
        one_hot_mapping = {}
        cur = len(self.dense_feats)
        for col in self.sparse_feats:
            vf = data[col].value_counts()
            one_hot_mapping[col] = (cur, vf.shape[0] + cur)
            cur += vf.shape[0]
        print("one hot mapping:", one_hot_mapping)
        X = pd.get_dummies(data[self.train_feats], columns=self.sparse_feats)

        # normalize dense features
        for col in self.dense_feats:
            x_mean = X[col].mean()
            x_max = X[col].max()
            x_min = X[col].min()
            print(f'{x_mean}:{x_max}:{x_min}')
            X[col] = (X[col] - x_mean) / (x_max - x_min)
        y = data['label'].to_numpy()
        X = X.to_numpy()

        # split data
        self.train_data_x, self.validation_data_x, self.train_data_y, self.validation_data_y \
            = train_test_split(X, y, test_size=0.5)
        self.test_data_x, self.validation_data_x, self.test_data_y, self.validation_data_y \
            = train_test_split(self.validation_data_x, self.validation_data_y, test_size=0.4)

        print(self.train_data_x.shape, self.validation_data_x.shape, self.test_data_x.shape)
        print(self.train_data_y.shape, self.test_data_y.shape)

        assert 'attributes' in config
        if os.path.isfile(config['attributes']):
            self.attr_split = pickle.load(open(config['attributes'], 'rb'))
        else:
            N = config['N']
            dense_d = len(self.dense_feats)
            client_attr_num = config['task_party_attr_num']
            data_party_attrs = list(np.arange(int(dense_d/2)))
            for col in self.sparse_feats[:int(len(self.sparse_feats)/2)]:
                data_party_attrs += list(np.arange(one_hot_mapping[col][0], one_hot_mapping[col][1]))
            attrs = {
                'data_party': data_party_attrs
            }
            client_possible_feats = list(np.arange(int(dense_d/2))) + self.sparse_feats[int(len(self.sparse_feats)/2):]
            for i in range(N):
                feats = np.random.choice(client_possible_feats, client_attr_num, replace=False)
                attrs[i] = []
                for f in feats:
                    if f.startswith('C'):
                        attrs[i] += list(np.arange(one_hot_mapping[f][0], one_hot_mapping[f][1]))
                    else:
                        attrs[i].append(int(f))
            pickle.dump(attrs, open(config['attributes'], 'wb'))
            self.attr_split = attrs
        print(self.attr_split)
        # exit()
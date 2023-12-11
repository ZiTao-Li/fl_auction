import os.path
import numpy as np
import json
from sklearn.model_selection import train_test_split

from .base import base_train, base_test_f1, base_test_auc

EVAL_METRIC_HANDLER = {
    'f1': base_test_f1,
    'auc': base_test_auc,
}

def eval_improvement(dataloader, client_id, config, tag):
    data_party_attrs = dataloader.attr_split['data_party']
    client_attrs = dataloader.attr_split[client_id]
    print("="*20)
    print(f"Data party attrs: {data_party_attrs}")
    print(f"Client {client_id} local attrs: {client_attrs}")
    client_local_model = base_train(dataloader.train_data_x[:, client_attrs], dataloader.train_data_y)
    client_local_performance = EVAL_METRIC_HANDLER[config['eval_metric']](
        client_local_model,
        dataloader.test_data_x[:, client_attrs],
        dataloader.test_data_y
    )
    print(f"Client {client_id} local trained model performance")
    print(f"Metric: {config['eval_metric']}, result: {client_local_performance}")

    # vfl_performance = 0
    vfl_model = base_train(
        dataloader.train_data_x[:, np.append(data_party_attrs, client_attrs)],
        dataloader.train_data_y
    )
    vfl_validation_performance = []
    for _ in range(20):
        _, validation_data_x, _, validation_data_y \
            = train_test_split(dataloader.validation_data_x, dataloader.validation_data_y, test_size=0.5)
        vfl_validation_performance.append(EVAL_METRIC_HANDLER[config['eval_metric']](
            vfl_model,
            validation_data_x[:, np.append(data_party_attrs, client_attrs)],
            validation_data_y
        ))
    vfl_test_performance = EVAL_METRIC_HANDLER[config['eval_metric']](
        vfl_model,
        dataloader.test_data_x[:, np.append(data_party_attrs, client_attrs)],
        dataloader.test_data_y
    )

    print(f"Client {client_id} VFL trained model performance")
    print(f"Metric: {config['eval_metric']}, result: {vfl_validation_performance}, {vfl_test_performance}")

    # store results
    result_file_path = config['vfl_improvement_store']['dir'] + \
                       tag + \
                       config['vfl_improvement_store']['file_name']
    print(result_file_path)
    # exit()
    if os.path.exists(result_file_path) and client_id > 0:
        results = json.load(
            open(result_file_path, 'r'))
        results[client_id] = {
            'local': client_local_performance,
            'vfl_valid': vfl_validation_performance,
            'vfl_test': vfl_test_performance,
            # 'validation_improve': vfl_validation_performance - client_local_performance,
            'test_improve': vfl_test_performance - client_local_performance,
        }
    else:
        results = {
            client_id: {
                'local': client_local_performance,
                'vfl_valid': vfl_validation_performance,
                'vfl_test': vfl_test_performance,
                # 'validation_improve': vfl_validation_performance - client_local_performance,
                'test_improve': vfl_test_performance - client_local_performance
            }
        }
        if not os.path.exists(config['vfl_improvement_store']['dir']):
            os.makedirs(config['vfl_improvement_store']['dir'])
    json.dump(results, open(result_file_path, 'w'), indent=2)
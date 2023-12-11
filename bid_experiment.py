import os.path
import random
from argparse import ArgumentParser
import json, yaml, copy
import numpy as np
import time

from bidding.DGA import eval_DGA_profit

VALUE_RANGE = (0, 1)

COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:gray']
VAL_TIMES = 20


def load_performance_results(config):
    result_path = config['performance_result']
    performance = json.load(open(result_path, 'r'))
    N = len(performance)
    performance_improvement = {'val': np.zeros((VAL_TIMES, N)),
                               'test': np.zeros(N)}
    # performance_improvement = np.zeros(N)
    for i in range(N):
        performance_improvement['test'][i] = performance[str(i)][
            "test_improve"]
        local = performance[str(i)]["local"]
        for j, v in enumerate(performance[str(i)]['vfl_valid']):
            performance_improvement['val'][j, i] = v - local
    return performance_improvement


def generate_values(N, random_type='uniform'):
    if random_type == 'uniform':
        return np.round(
            np.random.random(N) * (VALUE_RANGE[1] - VALUE_RANGE[0]), 3)
    else:
        raise NotImplementedError


def emulate_untruthful(client_values, untruthful_client, granularity=100):
    '''
    client_value: real client values, float in (0, 1)
    untruthful_client: assuming the
    :return: list of bids in which only varying untruthful_client bids
    '''
    untruthful_cases = []
    manipulates = []
    for i in range(granularity):
        case = copy.deepcopy(client_values)
        untruthful_bid = np.round(
            (VALUE_RANGE[1] - VALUE_RANGE[0]) / granularity * (i + 1), 3)
        case[untruthful_client] = untruthful_bid
        untruthful_cases.append(case)
        manipulates.append(untruthful_bid)
    return untruthful_cases, manipulates


def eval_fixed_k_profit(client_values, client_bids_or_ranking, performances,
                        config):
    K = config['K']
    profit = np.zeros(len(client_values))
    if 'changing_ranking' in config and config[
        'changing_ranking'] and 'cur_untruthful' in config:
        current_untruthful = config['cur_untruthful']
        sorted_bidders = get_sorted(performances, client_values)
        sorted_bidders = [i for i in sorted_bidders if i != current_untruthful]
        ranking = client_bids_or_ranking
        if ranking <= K:
            # force this client one of the winner
            threshold_bidder = sorted_bidders[-K]
            payment = client_values[threshold_bidder] * performances[
                threshold_bidder]
            winners = np.append(sorted_bidders[-(K - 1):], current_untruthful)
        else:
            # force this client not to be winner
            threshold_bidder = sorted_bidders[-(K + 1)]
            payment = client_values[threshold_bidder] * performances[
                threshold_bidder]
            winners = sorted_bidders[-K:]
    elif 'changing_score' in config and config[
        'changing_score'] and 'cur_untruthful' in config:
        current_untruthful = config['cur_untruthful']
        hacked_perfs = copy.deepcopy(performances)
        hacked_perfs[current_untruthful] = client_bids_or_ranking[
                                               current_untruthful] * config[
                                               'sensitivity']
        hacked_bids = copy.deepcopy(client_bids_or_ranking)
        hacked_bids[current_untruthful] = 1.0
        sorted_bidders = get_sorted(hacked_perfs, hacked_bids)
        threshold_bidder = sorted_bidders[-(K + 1)]
        payment = hacked_bids[threshold_bidder] * hacked_perfs[
            threshold_bidder]
        winners = sorted_bidders[-K:]
        print(current_untruthful, hacked_perfs[current_untruthful],
              hacked_bids[current_untruthful])
    elif 'changing_performance' in config and config[
        'changing_performance'] and 'cur_untruthful' in config:
        current_untruthful = config['cur_untruthful']
        hacked_perfs = copy.deepcopy(performances)
        hacked_perfs[current_untruthful] = client_bids_or_ranking[
                                               current_untruthful] * config[
                                               'sensitivity']
        hacked_bids = copy.deepcopy(client_bids_or_ranking)
        hacked_bids[current_untruthful] = client_values[current_untruthful]
        sorted_bidders = get_sorted(hacked_perfs, hacked_bids)
        threshold_bidder = sorted_bidders[-(K + 1)]
        payment = hacked_bids[threshold_bidder] * hacked_perfs[
            threshold_bidder]
        winners = sorted_bidders[-K:]
        print(current_untruthful, hacked_perfs[current_untruthful],
              hacked_bids[current_untruthful])
    else:
        client_bids = client_bids_or_ranking
        sorted_bidders = get_sorted(performances, client_bids)
        threshold_bidder = sorted_bidders[-(K + 1)]
        payment = client_bids[threshold_bidder] * performances[
            threshold_bidder]
        winners = sorted_bidders[-K:]
    print('== winners', winners)
    gains = client_values[winners] * performances[winners]
    profit[winners] = gains - payment
    seller_profit = payment * len(winners)
    return profit, seller_profit


def get_sorted(performances, bids):
    performances = np.array(performances)
    bids = np.array(bids)
    gain = performances * bids
    sorted_clients = np.argsort(gain)
    return sorted_clients


def eval_with_val(eval_profit, client_values, bids, client_performance,
                  config):
    eval_val_client_results = []
    eval_val_seller_results = []
    for i in range(VAL_TIMES):
        client_profit, seller_profit = eval_profit(client_values, bids,
                                                   client_performance['val'][
                                                       i], config)
        eval_val_client_results.append(client_profit)
        eval_val_seller_results.append(seller_profit)

    if isinstance(eval_val_client_results[0][0], float):
        eval_val_client_avg = np.mean(eval_val_client_results, axis=0)
        eval_val_client_std = np.std(eval_val_client_results, axis=0)
        eval_val_client_min = np.min(eval_val_client_results, axis=0)
        eval_val_client_max = np.max(eval_val_client_results, axis=0)
    else:
        # for EM
        eval_val_client_avg = np.mean(eval_val_client_results, axis=(0, 1))
        eval_val_client_std = np.std(eval_val_client_results, axis=(0, 1))
        eval_val_client_min = np.quantile(eval_val_client_results, q=0.25,
                                          axis=(0, 1))
        eval_val_client_max = np.quantile(eval_val_client_results, q=0.75,
                                          axis=(0, 1))
    eval_val_seller_avg = np.mean(eval_val_seller_results)
    eval_val_seller_std = np.std(eval_val_seller_results)
    return eval_val_client_avg, \
        eval_val_client_std, \
        eval_val_client_min, \
        eval_val_client_max, \
        eval_val_seller_avg, \
        eval_val_seller_std


def eval_with_test(eval_profit, client_values, bids, client_performance,
                   config):
    if config['auction_type'] == 'DGA' and config["strategy"] != "naive":
        client_profit, seller_profit = eval_profit(client_values, bids,
                                                   client_performance['test'],
                                                   config)
        client_profit = np.mean(client_profit, axis=0)
        seller_profit = np.mean(seller_profit)
    else:
        client_profit, seller_profit = eval_profit(client_values, bids,
                                                   client_performance['test'],
                                                   config)
    return client_profit, seller_profit


def save_to_json(results, untruthful_client, true_rank, config):
    file_name = "./results/bidding/" + str(config['fig_save_name']).replace(
        '.png', '')
    file_name += '.json'
    if os.path.isfile(file_name):
        with open(file_name, 'r') as jsonf:
            old_results = json.load(jsonf)
    else:
        if not os.path.isdir("./results/bidding/"):
            os.makedirs("./results/bidding/")
        old_results = {}
    old_results[untruthful_client] = [results, true_rank]
    with open(file_name, 'w') as jsonf:
        json.dump(old_results, jsonf, indent=2)
    print("save to " + file_name)


def clean_json(config):
    file_name = "./results/bidding/" + str(config['fig_save_name']).replace(
        '.png', '')
    file_name += '.json'
    with open(file_name, 'w') as jsonf:
        json.dump({}, jsonf, indent=2)


def simulate_profit(args):
    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    if 'seed' in config:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    client_performance = load_performance_results(config)
    print(np.max(client_performance['test']),
          np.max(client_performance['val']))
    print("max perf:", max(client_performance['test']),
          np.max(client_performance['val']))
    client_test_performance = client_performance['test']
    client_val_performance = client_performance['val']
    client_values = generate_values(config['N'])
    print(client_values)
    print("best:", np.argsort(client_performance['test'] * client_values))
    client_bids = copy.deepcopy(client_values)
    N = len(client_values)
    # different auctions
    if config['auction_type'] == 'DGA':
        eval_profit = eval_DGA_profit
    else:
        eval_profit = eval_fixed_k_profit

    clean_json(config)
    for untruthful_client in config['untruthful_client']:
        print(
            f"untruthful client performance improved: {client_performance['test'][untruthful_client]}"
            f" {client_performance['val'][:, untruthful_client]}")
        untruthful_bidding_cases, bids = emulate_untruthful(client_bids,
                                                            untruthful_client)
        various_profit = {}
        truthful_eval_config = copy.deepcopy(config)
        if 'strategy' in truthful_eval_config:
            truthful_eval_config['strategy'] = 'naive'
        truthful_profits, seller_profit = \
            eval_with_test(eval_profit, client_values, client_values,
                           client_performance, truthful_eval_config)
        if 'changing_performance' in config:
            various_profit['truthful'] = (
            client_performance['test'][untruthful_client],
            truthful_profits[untruthful_client])
        elif 'changing_score' in config:
            various_profit['truthful'] = (
            client_performance['test'][untruthful_client] * client_values[
                untruthful_client],
            truthful_profits[untruthful_client])
        else:
            various_profit['truthful'] = (client_values[untruthful_client],
                                          truthful_profits[untruthful_client])
        print(untruthful_client, various_profit, various_profit['truthful'])
        # exit()
        various_profit['untruthful'] = {}
        # for manipulate untruthful client ranking
        config['cur_untruthful'] = untruthful_client
        if 'changing_ranking' in config and config['changing_ranking']:
            untruthful_bidding_cases = np.arange(1, len(client_values) + 1)
            bids = list(range(1, len(client_values) + 1))
        if 'changing_performance' in config and config['changing_performance']:
            various_profit['truthful'] = (
            client_performance['test'][untruthful_client],
            truthful_profits[untruthful_client])
            granularity = 100
            bids = np.arange(0, config['sensitivity'] + 1e-5,
                             config['sensitivity'] * (
                                         VALUE_RANGE[1] - VALUE_RANGE[
                                     0]) / granularity)
        if 'changing_score' in config and config['changing_score']:
            various_profit['truthful'] = (
            client_performance['test'][untruthful_client] * client_values[
                untruthful_client],
            truthful_profits[untruthful_client])
            granularity = 100
            bids = np.arange(0, config['sensitivity'] + 1e-5,
                             config['sensitivity'] * (
                                         VALUE_RANGE[1] - VALUE_RANGE[
                                     0]) / granularity)
        for i, ut_bids in enumerate(untruthful_bidding_cases):
            # print(ut_bids[untruthful_client])
            eval_val_client_avg, eval_val_client_std, eval_val_client_min, eval_val_client_max, eval_val_seller_avg, eval_val_seller_std = \
                eval_with_val(eval_profit, client_values, ut_bids,
                              client_performance, config)
            client_ut_test_profit, seller_ut_test_profit = \
                eval_with_test(eval_profit, client_values, ut_bids,
                               client_performance, config)
            various_profit['untruthful'][bids[i]] = [
                eval_val_client_avg[untruthful_client],
                eval_val_client_std[untruthful_client],
                client_ut_test_profit[untruthful_client],
                eval_val_client_min[untruthful_client],
                eval_val_client_max[untruthful_client],
            ]
        del config['cur_untruthful']
        true_rank = -1
        sorted_true = get_sorted(client_test_performance, client_values)
        print("sorted test (no manipulation):", sorted_true)
        if 'K' in config:
            print(sorted_true[-int(config['K']):],
                  sorted_true[-int(config['K']) - 5:-int(config['K'])])
        for i, c in enumerate(sorted_true):
            if c == untruthful_client:
                true_rank = i
        save_to_json(various_profit, untruthful_client, N - true_rank, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-c", "--config", type=str,
                        help='bidding config file path')
    args = parser.parse_args()
    start_time = time.time()
    simulate_profit(args)
    end_time = time.time()
    print(f"auction mechanism takes time: {end_time - start_time}")

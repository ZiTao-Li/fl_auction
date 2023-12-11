import os.path
import random
from argparse import ArgumentParser
import json, yaml, copy
import numpy as np
import matplotlib.pyplot as plt
import time

from bidding.DGA import eval_DGA_profit

VALUE_RANGE = (0, 1)

COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:gray']
VAL_TIMES = 20

def load_performance_results(config):
    result_path = config['performance_result']
    performance = json.load(open(result_path, 'r'))
    N = len(performance)
    performance_improvement = {'val': np.zeros((VAL_TIMES, N)), 'test': np.zeros(N)}
    # performance_improvement = np.zeros(N)
    for i in range(N):
        performance_improvement['test'][i] = performance[str(i)]["test_improve"]
        local = performance[str(i)]["local"]
        for j, v in  enumerate(performance[str(i)]['vfl_valid']):
            performance_improvement['val'][j, i] = v - local
    return performance_improvement

def generate_values(N, random_type='uniform'):
    if random_type == 'uniform':
        return np.round(np.random.random(N) * (VALUE_RANGE[1] - VALUE_RANGE[0]), 3)
    else:
        raise NotImplementedError

# def emulate_untruthful(client_values, untruthful_client, granularity=100):
#     '''
#     client_value: real client values, float in (0, 1)
#     untruthful_client: assuming the
#     :return: list of bids in which only varying untruthful_client bids
#     '''
#     untruthful_cases = []
#     bids = []
#     for i in range(granularity):
#         case = copy.deepcopy(client_values)
#         untruthful_bid = np.round((VALUE_RANGE[1] - VALUE_RANGE[0]) / granularity * (i + 1), 3)
#         case[untruthful_client] =untruthful_bid
#         untruthful_cases.append(case)
#         bids.append(untruthful_bid)
#     return untruthful_cases, bids


def eval_fixed_k_social_welfare(client_values, client_bids, val_performances, test_performances, config):
    K = config['K']
    profit = np.zeros(len(client_values))
    val_sorted_bidders = get_sorted(val_performances, client_bids)
    # K is fixed
    val_winners = val_sorted_bidders[-K:]
    social_welfare = np.sum(client_values[val_winners] * val_performances[val_winners])
    test_sorted_bidders = get_sorted(test_performances, client_bids)
    test_winners = test_sorted_bidders[-K:]
    test_social_welfare = np.sum(client_values[test_winners] * test_performances[test_winners])
    val_K = val_sorted_bidders[K]
    product_social_welfare = np.sum(client_values[val_winners] * test_performances[val_winners])
    print("val:", val_winners, social_welfare, val_performances[val_K] * client_bids[val_K] )
    print("test", test_winners, test_social_welfare)
    # return profit, seller_profit
    # return profit, social_welfare
    return profit, product_social_welfare

def get_sorted(performances, bids):
    performances = np.array(performances)
    bids = np.array(bids)
    gain = performances * bids
    sorted_clients = np.argsort(gain)
    return sorted_clients


def plot_client_profit(results, untruthful_client, true_rank):
    color = COLOR_LIST.pop(0)
    plt.scatter(results['truthful'][0], results['truthful'][1], color=color, s=150)
    # if isinstance(results['untruthful'][0.01], float):
    untruthful_bids = sorted(results['untruthful'].keys())
    untruthful_test_profits = [results['untruthful'][i][-1] for i in untruthful_bids]
    untruthful_val_profits = np.array([results['untruthful'][i][0] for i in untruthful_bids])
    untruthful_val_profits_std = np.array([results['untruthful'][i][1] for i in untruthful_bids])
    plt.plot(untruthful_bids, untruthful_test_profits,
             label="client "+str(untruthful_client) + f" ({true_rank})",
             color=color, linewidth=4,
             )
    plt.plot(untruthful_bids, untruthful_val_profits,
             label="client " + str(untruthful_client) + f" ({true_rank})",
             color=color, linewidth=4, linestyle='dotted',
             )
    plt.fill_between(
        untruthful_bids,
        untruthful_val_profits - untruthful_val_profits_std,
        untruthful_val_profits + untruthful_val_profits_std,
        alpha=0.2, color=color,
        )
    # else:
    #     untruthful_bids = sorted(results['untruthful'].keys())
    #     untruthful_profits = [np.mean(results['untruthful'][i], axis=0) for i in untruthful_bids]
    #     untruthful_profits_std = np.array([np.std(results['untruthful'][i], axis=0) for i in untruthful_bids])
    #     plt.plot(untruthful_bids,
    #              untruthful_profits,
    #              label="client " + str(untruthful_client) + f" ({true_rank})",
    #              color=color, linewidth=4,
    #              )
    #     plt.fill_between(untruthful_bids,
    #                      untruthful_profits - untruthful_profits_std,
    #                      untruthful_profits + untruthful_profits_std,
    #                      alpha=0.2, color=color,
    #                      )


def eval_with_val(eval_profit, client_values, bids, client_performance, config):
    eval_val_client_results = []
    eval_val_desired_metric_results = []
    for i in range(VAL_TIMES):
        if config['auction_type'] == 'DGA' and config['strategy'] == 'EM':
            client_profit, desired_metric = eval_profit(client_values, bids, client_performance['val'][i], config)
        else:
            client_profit, desired_metric = eval_profit(client_values, bids, client_performance['val'][i],
                                                        client_performance['test'], config)
        print(">>", len(client_profit))
        eval_val_client_results.append(client_profit)
        eval_val_desired_metric_results.append(desired_metric)
    print(">>> ", len(eval_val_client_results), len(eval_val_client_results[0]))
    if isinstance(eval_val_client_results[0][0], float):
        eval_val_client_avg = np.mean(eval_val_client_results, axis=0)
        eval_val_client_std = np.std(eval_val_client_results, axis=0)
    else:
        eval_val_client_avg = np.mean(eval_val_client_results, axis=(0,1))
        eval_val_client_std = np.std(eval_val_client_results, axis=(0,1))
    eval_val_seller_avg = np.mean(eval_val_desired_metric_results)
    eval_val_seller_std = np.std(eval_val_desired_metric_results)
    eval_val_seller_min = np.min(eval_val_desired_metric_results)
    eval_val_seller_max = np.max(eval_val_desired_metric_results)
    print(eval_val_client_avg.shape)
    return eval_val_client_avg, eval_val_client_std, \
        eval_val_seller_avg, eval_val_seller_std, eval_val_seller_min, eval_val_seller_max


def eval_with_test(eval_profit, client_values, bids, client_performance, config):
    if config['auction_type'] == 'DGA' and config["strategy"] != "naive" :
        client_profit, seller_profit = eval_profit(client_values, bids, client_performance['test'], config)
        client_profit = np.mean(client_profit, axis=0)
        seller_profit = np.mean(seller_profit)
    else:
        client_profit, seller_profit = eval_profit(client_values, bids, client_performance['test'], client_performance['test'], config)
    return client_profit, seller_profit

def save_to_json(results, config):
    file_name = "./results/sensitivity/"+str(config['fig_save_name']).replace('.png', '')
    file_name += '.json'
    if not os.path.isdir("./results/sensitivity/"):
        os.makedirs("./results/sensitivity/")
    with open(file_name, 'w') as jsonf:
        json.dump(results, jsonf, indent=2)


def simulate_profit(args):
    plt.figure(figsize=(8, 6))
    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    print(config)
    if 'seed' in config:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    client_performance = load_performance_results(config)
    client_test_performance = client_performance['test']
    client_val_performance = client_performance['val']
    # if os.path.isfile(config['values']):
    #     client_values = np.load(config['values'])
    # else:
    client_values = generate_values(config['N'])
        # np.save(config['values'], client_values)
    client_bids = copy.deepcopy(client_values)
    N = len(client_values)
    # different auctions
    if config['auction_type'] == 'DGA':
        eval_profit = eval_DGA_profit
    else:
        eval_profit = eval_fixed_k_social_welfare

    new_config = copy.deepcopy(config)
    Ns = copy.deepcopy(config['N'])

    various_results = {}
    print(config)
    if config['auction_type'] == "FixedK":
        Ks = copy.deepcopy(config['K'])
        for K in Ks:
            print("-" * 10, N, K)
            new_config['N'] = N
            new_config['K'] = K
            eval_val_client_avg, eval_val_client_std, eval_val_seller_avg, eval_val_seller_std, \
                eval_val_seller_min, eval_val_seller_max = \
                eval_with_val(eval_profit, client_values, client_bids, client_performance, new_config)
            client_test_profit, seller_test_profit = \
                eval_with_test(eval_profit, client_values, client_bids, client_performance, new_config)
            various_results[K] = [
                eval_val_seller_avg,
                eval_val_seller_std,
                seller_test_profit,
                eval_val_seller_min,
                eval_val_seller_max,
            ]
            print("-" * 20)
    elif config['auction_type'] == "DGA" and config['strategy'] == "EM":
        epss = copy.deepcopy(config['eps'])
        for eps in epss:
            print("-" * 10, N, eps)
            new_config['N'] = N
            new_config['eps'] = eps
            eval_val_client_avg, eval_val_client_std, \
                eval_val_seller_avg, eval_val_seller_std, eval_val_seller_min, eval_val_seller_max = \
                eval_with_val(eval_profit, client_values, client_bids, client_performance, new_config)
            client_test_profit, seller_test_profit = \
                eval_with_test(eval_profit, client_values, client_bids, client_performance, new_config)
            various_results[str(eps)] = [
                eval_val_seller_avg,
                eval_val_seller_std,
                seller_test_profit,
                eval_val_seller_min,
                eval_val_seller_max
            ]
            print("-" * 20)

    save_to_json(various_results, new_config)
    # plt.legend(fontsize=18)
    # plt.xlabel("bids", fontsize=18)
    # plt.ylabel("profit", fontsize=18)
    # # plt.yscale('symlog')
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    #
    # # plt.show()
    # plt.savefig("./figure/"+config['fig_save_name'], bbox_inches='tight')




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help='bidding config file path')
    start_time = time.time()
    args = parser.parse_args()
    simulate_profit(args)
    end_time = time.time()
    print(f"total time: {end_time - start_time}" )

import copy

import numpy as np

def gen_price_set(config):
    lower, upper = config['price_set']['lower'], config['price_set']['upper']
    assert config['price_set']['separate'] in ['uniform', 'multi']
    if config['price_set']['separate'] == 'uniform':
        step = config['price_set']['factor']
        price_set = np.arange(start=lower, stop=upper, step=step)
    else:
        factor = config['price_set']['factor']
        price_set = [lower]
        while price_set[-1] < upper:
            price_set.append(price_set * (1 + factor))
    return price_set


def eval_DGA_profit(client_values, client_bids, performances, config):
    strategy = config['strategy']
    price_set = gen_price_set(config)
    if 'changing_performance' in config and config['changing_performance'] and 'cur_untruthful' in config:
        true_performances = copy.deepcopy(performances)
        client_values, bids, eval_performances, config = manipulate_performance(client_values,
                                                                                  client_bids,
                                                                                  performances,
                                                                                  config)
        print("**", true_performances[config['untruthful_client']], eval_performances[config['untruthful_client']])
    elif 'changing_score' in config and config['changing_score'] and 'cur_untruthful' in config:
        true_performances = copy.deepcopy(performances)
        client_values, bids, eval_performances, config = manipulate_score(client_values,
                                                                                client_bids,
                                                                                performances,
                                                                                config)
        print("**", bids[config['untruthful_client']], eval_performances[config['untruthful_client']])
    else:
        eval_performances = performances
        bids = client_bids

    if strategy is None or strategy == 'naive':
        return naive(client_values, bids, eval_performances, price_set)
    elif strategy == 'EM' and 'changing_performance' in config and config['changing_performance']:
        return exponential_mechanism(client_values, bids, eval_performances, price_set, config, true_performances)
    elif strategy == 'EM' and 'changing_score' in config and config['changing_score']:
        return exponential_mechanism(client_values, bids, eval_performances, price_set, config, true_performances)
    elif strategy == 'EM':
        return exponential_mechanism(client_values, bids, eval_performances, price_set, config)
    elif strategy == 'random_sampling' and 'changing_performance' in config and config['changing_performance']:
        return random_sampling(client_values, bids, eval_performances, price_set, true_performances)
    elif strategy == 'random_sampling' and 'changing_score' in config and config['changing_score']:
        return random_sampling(client_values, bids, eval_performances, price_set, true_performances)
    elif strategy == 'random_sampling':
        return random_sampling(client_values, bids, eval_performances, price_set)
    else:
        raise NotImplementedError

def manipulate_performance(client_values, client_bids, performances, config):
    assert 'untruthful_client' in config and 'changing_performance' in config and config['changing_performance']
    untruthful_client = config['cur_untruthful']
    change = client_bids[untruthful_client]
    manipulate_perf = copy.deepcopy(performances)
    manipulate_perf[untruthful_client] = change * config['sensitivity']
    print(change * config['sensitivity'] )
    client_bids = copy.deepcopy(client_values)
    return client_values, client_bids, manipulate_perf, config

def manipulate_score(client_values, client_bids, performances, config):
    assert 'untruthful_client' in config and 'changing_score' in config and config['changing_score']
    untruthful_client = config['cur_untruthful']
    change = client_bids[untruthful_client]
    manipulate_perf = copy.deepcopy(performances)
    manipulate_perf[untruthful_client] = change * config['sensitivity']
    client_bids = copy.deepcopy(client_values)
    client_bids[untruthful_client] = 1.0
    print(manipulate_perf[untruthful_client] * client_values[untruthful_client])
    return client_values, client_bids, manipulate_perf, config


def naive(client_values, client_bids, performances, price_set):
    buyer_profit = np.zeros(len(client_values))
    gain = performances * client_bids
    utilities = np.zeros(len(price_set))
    for i, p in enumerate(price_set):
        utilities[i] = p * np.sum(gain > p)
    best_p = price_set[np.argmax(utilities)]
    winners = np.argwhere(gain > best_p).flatten()
    print(gain[winners])
    # print(price_set)
    print(f"best price {best_p}, winners: {winners}")
    # exit()
    seller_profit = best_p * len(winners)
    buyer_profit[winners] = client_values[winners] * performances[winners] - best_p
    return buyer_profit, seller_profit


def exponential_mechanism(client_values, client_bids, performances, price_set, config, true_performances=None):
    gain = performances * client_bids
    utilities = np.zeros(len(price_set))
    for i, p in enumerate(price_set):
        utilities[i] = p * np.sum(gain > p)

    if true_performances is None:
        true_performances = copy.deepcopy(performances)

    eps, sensitivity = config['eps'], config['sensitivity']
    prob_scale = np.exp(eps * utilities / sensitivity)
    probs = prob_scale / np.sum(prob_scale)
    repeat_times = 100
    buyer_profit = [np.zeros(len(client_values)) for _ in range(repeat_times)]
    seller_profit = [0 for _ in range(repeat_times)]
    # repeat experiments for 20 times
    for r in range(repeat_times):
        best_p = np.random.choice(price_set, size=1, p=probs)
        winners = np.argwhere(gain > best_p).flatten()
        seller_profit[r] = best_p * len(winners)
        buyer_profit[r][winners] += client_values[winners] * true_performances[winners] - best_p
        print(f"best price {best_p}, winners: {winners}")
    return buyer_profit, seller_profit


def random_sampling(client_values, client_bids, performances, price_set, true_performances=None):
    N = len(client_values)
    buyer_profit = np.zeros(N)
    gain = performances * client_bids
    clients = np.arange(N)
    repeat_times = 10

    if true_performances is None:
        true_performances = copy.deepcopy(performances)

    buyer_profit = [np.zeros(len(client_values)) for _ in range(repeat_times)]
    seller_profit = [0 for _ in range(repeat_times)]
    for r in range(repeat_times):
        if r % 10 == 0:
            print(f"round {r}")
        np.random.shuffle(clients)
        # set_1 = np.random.choice(N, int(N/2), replace=False)
        set_1_clients = clients[:int(N/2)]
        set_2_clients = clients[int(N/2):]
        set_1_gains = gain[set_1_clients]
        set_2_gains = gain[set_2_clients]
        set_1_utilities, set_2_utilities = np.zeros(len(price_set)),  np.zeros(len(price_set))
        # for i, p in enumerate(price_set):
        #     set_1_utilities[i] = p * np.sum(set_1_gains > p)
        #     set_2_utilities[i] = p * np.sum(set_2_gains > p)
        reshaped_p = price_set.reshape((-1, 1))
        set_1_utilities = (reshaped_p * ((set_1_gains - reshaped_p ) > 0)).sum(axis=1)
        set_2_utilities = (reshaped_p * ((set_2_gains - reshaped_p) > 0)).sum(axis=1)
        best_set_1_p = price_set[np.argmax(set_1_utilities)]
        best_set_2_p = price_set[np.argmax(set_2_utilities)]
        # apply best set 2 price to set 1 clients
        set_1_winners = set_1_clients[np.argwhere(set_1_gains > best_set_2_p).flatten()]
        # apply best set 1 price to set 2 clients
        set_2_winners = set_2_clients[np.argwhere(set_2_gains > best_set_1_p).flatten()]
        # print("best prices:", best_set_1_p, best_set_2_p, len(set_1_winners), len(set_2_winners))
        seller_profit[r] = best_set_2_p * len(set_1_winners) + best_set_1_p * len(set_2_winners)
        buyer_profit[r][set_1_winners] = client_values[set_1_winners] * true_performances[set_1_winners] - best_set_2_p
        buyer_profit[r][set_2_winners] = client_values[set_2_winners] * true_performances[set_2_winners] - best_set_1_p
    return buyer_profit, seller_profit

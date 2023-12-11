import os.path
import random
from argparse import ArgumentParser
import json, yaml, copy
import numpy as np
import matplotlib.pyplot as plt

from bidding.DGA import eval_DGA_profit

VALUE_RANGE = (0, 1)

COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:gray', 'tab:olive']
MARKER_LIST = ['o', '^', 'p', 'P', 'X', "*"]
VAL_TIMES = 20
plt.style.use('seaborn')

def plot_client_profit(args):
    plt.figure(figsize=(9, 4))

    config = yaml.safe_load(open(args.config, 'r'))
    json_file_name = "./results/bidding/" + str(config['fig_save_name']).replace('.png', '')
    json_file_name += '.json'
    png_file_name = "./results/bidding/sensitivity-" + str(config['fig_save_name'])

    with open(json_file_name, 'r') as jsonf:
        all_results = json.load(jsonf)
    # print(all_results)

    for untruthful_client, [results,true_rank]  in all_results.items():
        print(COLOR_LIST)
        color = COLOR_LIST.pop(0)
        marker = MARKER_LIST.pop(0)
        print(results)
        # sactter
        if 'changing_ranking' in config and config['changing_ranking']:
            plt.scatter(true_rank, results['truthful'][1], color=color, s=150, marker=marker)
        else:
            plt.scatter(results['truthful'][0], results['truthful'][1], color=color, s=150, marker=marker)

        # if isinstance(results['untruthful'][0.01], float):
        untruthful_bids = sorted([float(b) for b in results['untruthful'].keys()])
        if 'changing_ranking' in config and config['changing_ranking']:
            untruthful_bids = sorted([int(b) for b in results['untruthful'].keys()])
        untruthful_test_profits = [results['untruthful'][str(i)][2] for i in untruthful_bids]
        untruthful_val_profits = np.array([results['untruthful'][str(i)][0] for i in untruthful_bids])
        untruthful_val_profits_std = np.array([results['untruthful'][str(i)][1] for i in untruthful_bids])
        untruthful_val_profits_min = np.array([results['untruthful'][str(i)][3] for i in untruthful_bids])
        untruthful_val_profits_max = np.array([results['untruthful'][str(i)][4] for i in untruthful_bids])
        plt.plot(untruthful_bids, untruthful_test_profits,
                 # label="client "+str(untruthful_client) + f" ({true_rank})",
                 color=color, linewidth=2,linestyle='dotted',
                 )
        # plt.vlines(results['truthful'][0], ymin=0, ymax=untruthful_val_profits[-1],
        #            color=color,
        #            linestyle='dashed',
        #            # label="client "+str(untruthful_client)+' value'
        #            )
        plt.plot(untruthful_bids, untruthful_val_profits,
                 # label="client " + str(untruthful_client) ,
                 color=color, linewidth=2,
                 )
        plt.fill_between(
            untruthful_bids,
            # untruthful_val_profits - untruthful_val_profits_std,
            # untruthful_val_profits + untruthful_val_profits_std,
            untruthful_val_profits_min, untruthful_val_profits_max,
            alpha=0.2, color=color,
            )
        plt.plot([], [],  color=color, marker=marker, markersize=15,
                 label="client " + str(untruthful_client) + f" ({true_rank})")
    if len(config['untruthful_client']) == 2:
        cols = 1
    else:
        cols = 3
    plt.legend(
        bbox_to_anchor=(-0.2, 1.04),
        # bbox_to_anchor=(-0.55, 0.6),
        loc="lower left", columnspacing=0.8,
        fontsize=24,
        ncol=cols,
        frameon=True
    )
    if 'changing_ranking' in config and config['changing_ranking']:
        plt.xlabel("rank", fontsize=24)
    elif 'changing_performance' in config and config['changing_performance']:
        plt.xlabel("improvement", fontsize=24)
    elif 'changing_score' in config and config['changing_score']:
        plt.xlabel("score", fontsize=24)
    else:
        plt.xlabel("bid value", fontsize=24)
    plt.ylabel("utility", fontsize=24)
    # plt.yscale('symlog')
    plt.xticks(fontsize=24)
    if 'y_ticks' in config:
        plt.yticks(config['y_ticks'], fontsize=24)
    else:
        plt.yticks(fontsize=24)

    # plt.show()
    plt.savefig("./figure/" + config['fig_save_name'], bbox_inches='tight')

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-c", "--config", type=str,
                        help='bidding config file path')
    args = parser.parse_args()
    plot_client_profit(args)

import os.path
import random
from argparse import ArgumentParser
import json, yaml, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

from bidding.DGA import eval_DGA_profit

VALUE_RANGE = (0, 1)
plt.style.use('seaborn')

COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:gray']
VAL_TIMES = 20


def export_legend(legend, filename):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def plot_client_profit(args):
    plt.figure(figsize=(8, 4))
    config = yaml.safe_load(open(args.config, 'r'))
    png_file_name = "./figure/sensitivity-" + config['figure_save_name']
    assert isinstance(config['result_files'], dict)
    color_idx = 0
    for name, result_file in config['result_files'].items():
        json_file_name = "./results/sensitivity/" + result_file
        with open(json_file_name, 'r') as jsonf:
            results = json.load(jsonf)
        print(results)
        if config['auction_type'] == "FixedK":
            Xs = sorted(int(k) for k in results.keys())
        elif config['auction_type'] == "DGA":
            Xs = sorted(float(k) for k in results.keys())
        vals = np.array([results[str(k)][0] for k in Xs])
        stds = np.array([results[str(k)][1] for k in Xs])
        test = np.array([results[str(k)][2] for k in Xs])
        if config['auction_type'] == "FixedK":
            mins = np.array([results[str(k)][3] for k in Xs])
            maxs = np.array([results[str(k)][4] for k in Xs])
        else:
            mins = vals - stds
            maxs = vals + stds
        print(maxs - mins, stds)
        tag = 'social welfare' if config['auction_type'] == "FixedK" else 'profit'
        plt.plot(Xs, vals,
                 label="test "+ " "+ name,
                 color=COLOR_LIST[color_idx], linewidth=2,
                 )
        plt.plot(Xs, test,
                 label='production ' + " " + name,
                 color=COLOR_LIST[color_idx], linewidth=4, linestyle='dotted',
                 )
        plt.fill_between(
            Xs,
            mins,
            maxs,
            alpha=0.2, color=COLOR_LIST[color_idx],
        )
        color_idx += 1
        # plt.legend(fontsize=18)
    if config['auction_type'] == "FixedK":
        plt.xlabel("number of winners (K)", fontsize=24)
        plt.ylabel("social welfare", fontsize=24)
    elif config['auction_type'] == "DGA" and config["strategy"] == 'EM':
        plt.xlabel(r"$\epsilon$", fontsize=24)
        plt.ylabel("Seller's profit", fontsize=24)
    # plt.yscale('symlog')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    only_export_legend = False
    if only_export_legend:
        legend = plt.legend(bbox_to_anchor=(-0.2, 1.04), loc="lower left",
                            fontsize=35, ncol=4, frameon=True, columnspacing=0.3, handletextpad=0.1)
        fig_name = "./figure/sensitivity-legend-" + str(config['figure_save_name'])
        export_legend(legend, fig_name)
        exit()
    # plt.show()
    plt.savefig(png_file_name, bbox_inches='tight')



if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset can be provided via command line
    parser.add_argument("-c", "--config", type=str,
                        help='config file path')
    args = parser.parse_args()
    plot_client_profit(args)

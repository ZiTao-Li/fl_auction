import random
from datetime import datetime
import yaml
import numpy as np

from utils.arguments import get_arguments
from data.dataloader import CriteoLoader
from train.vfl_improve import eval_improvement

def main(args):
    print(args)
    config = yaml.safe_load(open(args.config, 'r'))
    if 'seed' in config:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    dataloader = CriteoLoader(config)
    tag = datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '-'
    for i in range(config['N']):
        print("tag:", tag)
        eval_improvement(dataloader, i, config, tag)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
### Download Criteo dataset:
The public Criteo dataset can be download from kaggle: 

https://www.kaggle.com/datasets/mrkmakr/criteo-dataset/data 

Move the dataset to `./ctr_data` and preprocess it with `./ctr_data/preprocess.py`   

### Simulation commands
VFL simulation:

`python ./ctr_experiments.py -c ./config/criteo_N_100.yaml`

Auction command:

`python ./bid_experiment.py -c ./config/criteo_N_100_fix_20.yaml`

`python ./bid_experiment.py -c ./config/criteo_change_perf_EM_1.yaml`



import os

from pathlib import Path
from train import train
from train import Trainer

from config import get_config

config = get_config()


data_path = Path(r"/home/etardieu/Documents/my_data/these/V1/Dataset/Dataset")

# [branche, lr mas one cycle, ratio lr groupe0/groupe1]
irc                = [[['IRC']], 1e-3  , 3.33]
mnh                = [[['MNH']], 1e-3  , 3.33]
biomasse           = [[['biomasse']], 1e-3  , 3.33]

for branches in [irc, mnh, biomasse]:
    train(data_path, branches)





# ################# LR RANGE TEST ##################
# branches = irc_irc
# trainer = Trainer(data_path, branches)
# trainer.run_lr_range_test(start_lr=1e-6, end_lr=1e-2, num_iters=200)
# ##################################################

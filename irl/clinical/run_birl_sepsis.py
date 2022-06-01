import os, sys
import pandas as pd

import pandas as pd
from irl.clinical.sepsis_env import SepsisEnv
from irl.original_birl.birl.mdp import MDP
import numpy as np
from irl.original_birl.birl.birl import birl
from irl.original_birl.birl.main import initialize_gridworld, initialize_rewards, initialize_true_rewards, convert_state_coord_to_state_idx, convert_state_idx_to_state_coord
from irl.original_birl.birl.prior import *

## Create SepsisEnv
iterations = 1000
reward_vec = [0.8, 0.6, 0.4]
env = SepsisEnv(reward_vec)


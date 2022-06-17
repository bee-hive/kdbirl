import os, sys
import numpy as np
from irl.gp_birl.gp_cde import GPCDE

# Generate the data
behavior_opt = None
training_points = None
training_rewards = None
gpcde = GPCDE(layer_sizes=4, latent_size=2)

# Posterior calculation (need to run the full likelihood) over all elements of the dataset (double for loop)
for pt in behavior_opt:
	likelihood = gpcde.likelihood(training_rewards, training_points, pt) # p(a_i, s_i|R)p(R)
# Put this all in stan.



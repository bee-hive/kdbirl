import pystan
from hashlib import md5
from os.path import join as pjoin
import os
import numpy as np
import pandas as pd
import pickle
import tqdm
from irl.inventory.irl_inventory import runFQI

# TODO: more data, visualize the reward cube, new weights
# Reward function: should penalize high levels of SOFA and increases in SOFA score and lactate levels

def sample_kdbirl(behavior_points, observations_points, observations_rewards, n_iter, n_posterior_samples, n, m, h, h_prime):
    idx = [i for i in range(len(behavior_points))]
    size_obs = min(n, len(behavior_points))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    behavior_points = np.asarray(behavior_points)[new_idx]
    behavior_points = behavior_points.reshape((n, 2))
    print("Len behavior: ", behavior_points.shape[0])

    idx = [i for i in range(len(observations_points))]
    size_obs = min(m, len(observations_points))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations_points = np.asarray(observations_points)[new_idx]
    observations_points = observations_points.reshape((m, 2))
    observations_rewards = np.asarray(observations_rewards)[new_idx]
    print("len observations: ", observations_points.shape[0])

    #h, h_prime = find_bandwidth(observations_points, observations_rewards)
    # h = 0.9
    # h_prime = 0.05
    # Fit and check if chains mixed
    rhat_failed = True
    kdbirl_data = {"J": 5, "n": n, "m": m, "training_points":
        observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}

    while rhat_failed:
        fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1, init='random', seed=1000)
        rhat_vals = fit.summary()["summary"][:, -1]
        print("RHAT: ", rhat_vals)
        rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)

    # Get samples, it's an 800 by 4 numpy array
    sample_reward = np.squeeze(fit.extract()["sample_reward"])
    cols = [str(i) for i in range(3)]
    sample_df = pd.DataFrame(np.vstack(sample_reward), columns=cols)
    print(str(fit))
    fname = "reward_samples/h=" + str(h) + "_hprime=" + str(h_prime) + "_iter=" + str(n_iter) + ".csv"
    sample_df.to_csv(fname)

train=True
evd =False
rollouts = False
if train:
    # Train a Q-value function
    # Generate demonstrations according to the reward parameters
    print("Expert Demonstrations")
    reward_params = [100, 5, 2, 3, 2]
    # behavior_points, _, _ = runFQI(reward_params)
	#
    # # Generate training data
    # observations_points = []
    # observations_rewards = []
	#
    # print("Training Demonstrations")
    # for kk, i in enumerate(tqdm.tqdm(range(0, 10, 3))):
    #     for j in range(0, 10, 3):
    #         for k in range(0, 10, 3):
    #             vec = [100, 5, i, j, k]
    #             dems_points, _, dems_rewards = runFQI(vec)
    #             observations_points.extend(dems_points)
    #             observations_rewards.extend(dems_rewards)
    # pickle.dump(behavior_points, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/behavior_points.pkl", 'wb'))
    # pickle.dump(observations_points, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/observations_points.pkl", 'wb'))
    # pickle.dump(observations_rewards, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/observations_rewards.pkl", 'wb'))

    behavior_points = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/behavior_points.pkl", 'rb'))
    observations_rewards = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/observations_rewards.pkl", 'rb'))
    observations_points = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/inventory/data/observations_points.pkl", 'rb'))
    with open("kdbirl_inventory.stan", "r") as file:
        model_code = file.read()
    code_hash = md5(model_code.encode("ascii")).hexdigest()
    cache_fn = pjoin("cached_models", "cached-model-{}.pkl".format(code_hash))

    if os.path.isfile(cache_fn):
        print("Loading cached model...")
        sm = pickle.load(open(cache_fn, "rb"))
    else:
        print("Saving model to cache...")
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, "wb") as f:
            pickle.dump(sm, f)

    # sample_kdbirl(behavior_points, observations_points, observations_rewards, n_iter=10000, n_posterior_samples=500, n=200,
    #               m=1000, h=0.01,  h_prime=0.01)

    sample_kdbirl(behavior_points, observations_points, observations_rewards, n_iter=5000, n_posterior_samples=1000, n=300,
                  m=1000, h=0.2, h_prime=0.2)
from gym_sepsis.envs.sepsis_env import SepsisEnv
from irl.clinical.sepsis_env import SepsisEnv as ModifiedSepsisEnv
import tqdm
import pystan
from hashlib import md5
from os.path import join as pjoin
import os
import pickle
import numpy as np
import tqdm
import pandas as pd
import pickle
import scipy
from irl.clinical.irl_sepsis import generate_trajectories, find_bandwidth, runFQI, value_function

# TODO: more data, visualize the reward cube, new weights
# Reward function: should penalize high levels of SOFA and increases in SOFA score and lactate levels

def sample_kdbirl(behavior_points, observations_points, observations_rewards, n_iter, n_posterior_samples, n, m, h, h_prime):
    idx = [i for i in range(len(behavior_points))]
    size_obs = min(n, len(behavior_points))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    behavior_points = np.asarray(behavior_points)[new_idx]
    print("Len behavior: ", behavior_points.shape[0])

    idx = [i for i in range(len(observations_points))]
    size_obs = min(m, len(observations_points))
    new_idx = np.random.choice(idx, size=size_obs, replace=False)
    observations_points = np.asarray(observations_points)[new_idx]
    observations_rewards = np.asarray(observations_rewards)[new_idx]
    print("len observations: ", observations_points.shape[0])

    #h, h_prime = find_bandwidth(observations_points, observations_rewards)
    # h = 0.9
    # h_prime = 0.05
    # Fit and check if chains mixed
    rhat_failed = True
    kdbirl_data = {"J": 3, "n": n, "m": m, "training_points":
        observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}

    while rhat_failed:
        fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1, control={"adapt_delta": 0.85}
                          )
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

train=False
evd =True
rollouts = False
if train:
    # Train a Q-value function
    # Generate demonstrations according to the reward parameters
    print("Expert Demonstrations")
    reward_params = [0.8, 0.6, 0.4]
    # behavior_points, _ = runFQI(reward_params, it=30, dim=len(reward_params))
    #
    # # Generate training data
    # observations_points = []
    # observations_rewards = []
    #
    # print("Training Demonstrations")
    # for kk, i in enumerate(tqdm.tqdm(range(0, 101, 30))):
    #     for j in range(0, 101, 30):
    #         for k in range(0, 101, 49):
    #             vec = [i / 100, j / 100, k/100]
    #             dems_points, dems_rewards = runFQI(vec, it=20, dim=len(reward_params))
    #             observations_points.extend(dems_points)
    #             observations_rewards.extend(dems_rewards)
    # pickle.dump(behavior_points, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/behavior_points.pkl", 'wb'))
    # pickle.dump(observations_points, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/observations_points.pkl", 'wb'))
    # pickle.dump(observations_rewards, open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/observations_rewards.pkl", 'wb'))

    behavior_points = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/behavior_points.pkl", 'rb'))
    observations_rewards = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/observations_rewards.pkl", 'rb'))
    observations_points = pickle.load(open("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/data/observations_rewards.pkl", 'rb'))
    with open("kdbirl_sepsis.stan", "r") as file:
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
                  m=1000, h=0.01, h_prime=0.01)

elif evd:
    true_reward = [0.8, 0.6, 0.4]
    file = "h=0.01_hprime=0.01_iter=5000.csv"
    df = pd.read_csv("/home/amandyam/contrastive-rl/contrastive-rl/irl/clinical/reward_samples/" + file)

    kdbirl_evds = []
    # KDBIRL trajectory
    for ii, i in enumerate(tqdm.tqdm(range(0, df.shape[0], 10))):
        sample = df.iloc[i][1:]
        evd = abs(abs(value_function(true_reward, true_reward, dim=3)) - abs(value_function(true_reward, sample, dim=3)))
        kdbirl_evds.append(evd)
    np.save("evds/" + file[:-4] + ".npy", np.asarray(kdbirl_evds))
elif rollouts:
    reward_params = [0.8, 0.6, 0.4]
    behavior_points, behavior_actions, behavior_rewards = runFQI(reward_params, it=10, full=True)
    with  open("behavior_states.pkl", 'wb') as f:
        pickle.dump(behavior_points, f)
    with open("behavior_actions.pkl", 'wb') as f:
        pickle.dump(behavior_actions, f)

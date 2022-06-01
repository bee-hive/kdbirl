import numpy as np
import pandas as pd
import pickle, os, csv, math, time, joblib
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import tqdm
import random

BEHAVIOR_PATH = "./behavior.pkl"

shape, scale = 2, 10
transition_foreground = np.random.gamma(shape, scale, (12, 10))
mu, sigma = 0, 4 # mean and standard deviation
transition_background = np.random.normal(mu, sigma, (12, 10))

mu, sigma = 0, 5
reward_function = np.random.normal(mu, sigma, (12, 1))

exploit = 0.8
explore = 1-exploit
num_samples = 100
num_patients = 100
actions = [[0, 0], [0, 1], [1, 0], [1, 1]]
mu, sigma = 0, 4

def a2c(action):
    actions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #actions = [[0, 0], [0, 2], [3, 1], [4, 4]]
    classes = []
    for a in action:
        a = list(a)
        for c in range(len(actions)):
            if actions[c] == a:
                classes.append(c)
    return classes

def p2c(pred):
    if pred <= 0.25:
        action = [0, 0]
    elif pred <= 0.5:
        action = [0, 1]
    elif pred <= 0.75:
        action = [1, 0]
    else:
        action = [1, 1]

# Mapping states to actions?
def c2a(c):
    d = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
    return np.array([d[k] for k in c])

def random_weights(size=5):
    # w = 2*np.random.uniform(size=size) - 1
    w = norm(np.random.uniform(size=size))
    # w / np.sum(np.abs(w))

    return w

def norm(vec):
    return vec / np.sum(np.abs(vec))

def learnBehaviour(training_set, test_set, state_dim):
    floc = "behavior.pkl"
    # if os.path.exists(floc):
    #    behaviour_pi = pickle.load(open(floc, 'rb'))
    # else:
    ## Use a linear regression to predict behavior
    behaviour_pi = LinearRegression()
    X = np.vstack((training_set['s'], test_set['s']))
    X = np.reshape(X, (-1, state_dim))
    if state_dim ==3 or state_dim ==46:
        y = np.hstack((training_set['a'], test_set['a']))
    else:
        y = a2c(np.vstack((training_set['a'], test_set['a'])))

    behaviour_pi.fit(X, y)
    pickle.dump(behaviour_pi, open(floc, 'wb'))

    return behaviour_pi

def construct_dicts(train_tuples, test_tuples):
    train = {}
    test = {}
    elts = ['s', 'a', 'ns', 'r', 'ds', 'vnum']
    for elt in elts:
        train[elt] = []
        test[elt] = []

    for tup in train_tuples:
        train['s'].append(tup[0])
        a = tup[1]
        try:
            a = np.concatenate(a).ravel()
            a = list(a)
            train['a'].append(a)
        except:
            train['a'].append(a)
        train['ns'].append(tup[2])
        train['r'].append(tup[3])
        train['ds'].append(tup[4])
        train['vnum'].append(tup[5])

    for tup in test_tuples:
        test['s'].append(tup[0])
        try:
            a = tup[1]
            a = np.concatenate(a).ravel()
            a = list(a)
            test['a'].append(a)
        except:
            test['a'].append(tup[1])
        test['ns'].append(tup[2])
        test['r'].append(tup[3])
        test['ds'].append(tup[4])
        test['vnum'].append(tup[5])
    return train, test

def cumulative_reward(rewards):
    c_reward = [rewards[0]]
    for i in range(1, len(rewards)):
        c_reward.append(rewards[i] + c_reward[i-1])
    return c_reward

def generate_tuples():
    transition_tuples = []
    for k, pat in enumerate(tqdm.tqdm(range(num_patients))):

        flip = np.random.choice(2)
        if flip == 0:
            ds = 'foreground'
        else:
            ds = 'background'
        # Generate a random initial state
        s = np.random.normal(mu, sigma, (10, 1))

        # Generate all of the tuples for this patient
        for i in range(num_samples):
            flip = random.uniform(0, 1)
            # Exploit
            if flip < exploit:            
                all_rewards = []
                for j, a in enumerate(actions):
                    a = np.asarray(a)
                    a = np.reshape(a, (2, 1))
                    s_a = np.concatenate((s, a))
                    reward = np.dot(reward_function.T, s_a)
                    all_rewards.append(reward)

                noise = np.random.normal(0, 0.05, 1)
                all_rewards = np.asarray(all_rewards)
                a = actions[np.argmax(all_rewards)]
                reward = np.max(all_rewards) + noise

                if ds == 'foreground':
                    t_m = transition_foreground
                else:
                    t_m = transition_background
                ns = np.matmul(s_a.T, t_m) / np.linalg.norm(np.matmul(s_a.T, t_m), ord=2)
                ns = np.add(ns, np.random.normal(0, 0.5, (1, 10))) # Add noise


            # Explore
            else:
                a = np.asarray(actions[np.random.choice(3)])
                a = np.reshape(a, (2, 1))
                s_a = np.concatenate((s, a)) # concatenate the state and action

                if ds == 'foreground':
                    t_m = transition_foreground
                else:
                    t_m = transition_background
                ns = np.matmul(s_a.T, t_m) / np.linalg.norm(np.matmul(s_a.T, t_m), ord=2)
                ns = np.add(ns, np.random.normal(0, 0.5, (1, 10))) # Add noise

                reward = np.dot(reward_function.T, s_a) + np.random.normal(0, 0.5, 1)

            # Transition tuple includes state, action, next state, reward, ds
            transition_tuples.append((list(s.flatten()), list(a), list(ns.flatten()), reward.flatten(), ds, i))
            s = ns.T
    split = int(0.8*len(transition_tuples))
    train_tuples = transition_tuples[:split]
    test_tuples = transition_tuples[split:]
    return train_tuples, test_tuples

def construct_mixedlm_ds(tuples, fname):
    tup_dict = {}
    elts = ['a0', 'a1', 'r', 'ds']
    for i in range(10):
        elts.append('s' +str(i))
    for elt in elts:
        tup_dict[elt] = []

    for j, tup in enumerate(tqdm.tqdm(tuples)):
        state = tup[0]
        for i in range(len(state)):
            tup_dict['s' + str(i)].append(state[i])

        try:
            a = tup[1]
            a = np.concatenate(a).ravel()
            a = list(a)
            tup_dict['a0'].append(int(a[0]))
            tup_dict['a1'].append(int(a[1]))
        except:
            a = tup[1]
            tup_dict['a0'].append(int(a[0]))
            tup_dict['a1'].append(int(a[1]))

        tup_dict['r'].append(float(tup[3]))
        tup_dict['ds'].append(int(tup[4] == 'foreground'))
    
    with open(fname, 'w') as fp:
        json.dump(tup_dict, fp)
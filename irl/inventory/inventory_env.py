import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

# Copied from online source
class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment
    TO BE EDITED
    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, n, k, c, h, p, lam=8):
        self.n = n
        self.action_space = spaces.Discrete(n)
        # self.observation_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(low=0, high=n, shape=(2, 1, 1),
											dtype=np.float32)
        self.max = n # Inventory size
        self.state = np.reshape(np.asarray([n, 0]), (2, 1, 1))
        self.k = k # Down payment
        self.c = c # Cost of item
        self.h = h # Ratio
        self.p = p # Price
        self.lam = lam # Demand

        # Set seed
        self._seed()

        # Start the first round
        self.reset()

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        x = x[0][0][0]
        m = self.max
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0)
        return r

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        obs = self.state
        demand = self.demand()
        obs2 = self.transition(obs, action, demand)
        self.state = np.reshape(np.asarray([[obs2, action]]), (2, 1, 1))
        obs2 = self.state
        reward = self.reward(obs[0][0][0], action, obs2[0][0][0])
        done = 0
        return obs2, reward, done, {}

    def reset(self):
        return self.state
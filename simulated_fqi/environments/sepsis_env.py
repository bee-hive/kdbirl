import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np
import os
from collections import deque
import pandas as pd
from gym import spaces
from random import randrange

STATE_MODEL = "sepsis_states.model"
TERMINATION_MODEL = "sepsis_termination.model"
OUTCOME_MODEL = "sepsis_outcome.model"
STARTING_STATES_VALUES = "sepsis_starting_states.npz"

NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 24

EPISODE_MEMORY = 10

am_dir = "/Users/aishwaryamandyam/Documents/Research/BEE/contrastive-rl/contrastive-rl/simulated_fqi/environments/sepsis_model/"
aj_dir = "./sepsis_model"


features = [
    "ALBUMIN",
    "ANION GAP",
    "BANDS",
    "BICARBONATE",
    "BILIRUBIN",
    "BUN",
    "CHLORIDE",
    "CREATININE",
    "DiasBP",
    "Glucose",
    "GLUCOSE",
    "HeartRate",
    "HEMATOCRIT",
    "HEMOGLOBIN",
    "INR",
    "LACTATE",
    "MeanBP",
    "PaCO2",
    "PLATELET",
    "POTASSIUM",
    "PT",
    "PTT",
    "RespRate",
    "SODIUM",
    "SpO2",
    "SysBP",
    "TempC",
    "WBC",
    "age",
    "is_male",
    "race_white",
    "race_black",
    "race_hispanic",
    "race_other",
    "height",
    "weight",
    "vent",
    "sofa",
    "lods",
    "sirs",
    "qsofa",
    "qsofa_sysbp_score",
    "qsofa_gcs_score",
    "qsofa_resprate_score",
    "elixhauser_hospital",
    "blood_culture_positive",
    "action",
    "state_idx",
]


class SepsisEnv(gym.Env):
    """
    Built from trained models on top of the MIMIC dataset, this
    Environment simulates the behavior of the Sepsis patient
    in response to medical interventions.
    For details see: https://github.com/akiani/gym-sepsis
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, starting_state=None, verbose=False):
        # module_path = os.path.dirname(__file__)
        module_path = am_dir
        self.verbose = verbose
        self.state_model = keras.models.load_model(
            os.path.join(module_path, STATE_MODEL)
        )
        self.termination_model = keras.models.load_model(
            os.path.join(module_path, TERMINATION_MODEL)
        )
        self.outcome_model = keras.models.load_model(
            os.path.join(module_path, OUTCOME_MODEL)
        )
        self.starting_states = np.load(
            os.path.join(module_path, STARTING_STATES_VALUES)
        )["sepsis_starting_states"]
        self.seed()
        self.action_space = spaces.Discrete(24)

        # use a pixel to represent next state
        self.observation_space = spaces.Box(
            low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES - 2, 1, 1), dtype=np.float32
        )
        self.reset(starting_state=starting_state)
        self.features = features

        self.n_actions = 25
        self.action_list = np.arange(self.n_actions)
        return

    def step(self, action):
        # create memory of present
        self.memory.append(
            np.append(
                np.append(self.s.reshape((1, NUM_FEATURES - 2)), action), self.state_idx
            )
        )
        if self.verbose:
            print("running on memory: ", self.memory)

        memory_array = np.expand_dims(self.memory, 0)
        next_state = self.state_model.predict(memory_array[:, :, :-1])

        # overwrite constant variables (these should't change during episode)
        constants = [
            "age",
            "race_white",
            "race_black",
            "race_hispanic",
            "race_other",
            "height",
            "weight",
        ]
        for constant in constants:
            idx = features.index(constant)
            val = self.state_0[idx]
            next_state[0, idx] = val

        termination = self.termination_model.predict(memory_array)
        outcome = self.outcome_model.predict(memory_array)

        termination_categories = ["continue", "done"]
        outcome_categories = ["death", "release"]

        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        reward = 1 + randrange(4)
        done = False

        if termination_state == "done":
            done = True
            if outcome_state == "death":
                reward = -15
            else:
                reward = 15

        # keep next state in memory
        self.s = next_state.reshape(46, 1, 1)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return self.s, reward, done, {"prob": 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[
                np.random.randint(0, len(self.starting_states))
            ][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(NUM_FEATURES - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="ansi", verbose=False):
        df = pd.DataFrame(self.memory, columns=features, index=range(0, 10))
        if verbose:
            print(df)
        return df

    def generate_tuples(self, group, n_trajectories=101):

        if group not in ["background", "foreground"]:
            raise Exception("group must be a string: 'background' or 'foreground'")
        # group is a string: "background" or "foreground"

        tuples = []

        for traj_idx in range(n_trajectories):

            # Reset agent
            self.reset()
            STATE_DIM = len(self.features) - 2
            states = []
            actions = []
            rewards = []

            # Save initial action
            s_init = np.squeeze(self.s)
            states.append(s_init)

            # Step until "done" flag becomes True (patient is released or dies)
            done = False
            while not done:

                # Choose random action
                a = np.random.choice(self.action_list)

                # Take action
                s, r, done, prob = self.step(a)
                s = np.squeeze(s)

                # Get current state
                df = self.render()

                # Save step
                states.append(s)
                actions.append(a)
                rewards.append(r)

            rewards = np.array(rewards)
            actions = np.array(actions)
            states = np.vstack(states)
            n_iter = len(rewards)

            ## Form tuples
            for ii in range(1, n_iter):

                s = states[ii, :]
                a = actions[ii]
                ns = states[ii + 1, :]
                r = rewards[ii]

                # Tuples are (state, action, next state, reward, group, index)
                curr_tuple = (s, a, ns, np.asarray([r]), group, ii)
                tuples.append(curr_tuple)

        return tuples

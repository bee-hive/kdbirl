#!/usr/bin/env python
import numpy as np
import pygame as pg
from math import sqrt
import math
import decimal

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GOALCOLOR = (18, 236, 229)
STARTCOLOR = (240, 134, 28)
AGENTCOLOR = (255, 255, 102)

"""
- make a gridworld an environment
- make it have states
- make it have transition function
- make it have a reward function
- make grid states have features
- make a demonstration class that stores a trajectory
- keep a list of demonstrations
- have gridworld have a record function that takes in number of demos to record
- MDP have states, transitions, rewards, record function
- Make instances of MDP like gridworld have record function
"""


class GridWorld:
    def __init__(self, mdp):
        self.mdp = mdp
        self.pixelx = 700
        self.pixely = 700
        self.num_states = np.shape(self.mdp.transitions)[0]
        self.height = int(math.ceil(sqrt(self.num_states)))
        self.width = int(sqrt(np.shape(self.mdp.transitions)[0]))
        self.rows = self.height
        self.cols = self.width
        self.start = 0
        self.goal = self.num_states - 1
        self.screen = None
        # self.policy = policy
        # self.is_human = policy is None

    def draw_box(self, row, col, color):
        # Leave 2 pixels between boxes
        block_size_y = (self.pixely - self.height * 2) / self.height
        block_size_x = (self.pixelx - self.width * 2) / self.width
        ul = (block_size_x * col + col * 2, block_size_y * row + row * 2)
        ur = (block_size_x * (col + 1) + (col - 1) * 2, block_size_y * row + row * 2)
        bl = (block_size_x * col + col * 2, block_size_y * (row + 1) + (row - 1) * 2)
        br = (
            block_size_x * (col + 1) + (col - 1) * 2,
            block_size_y * (row + 1) + (row - 1) * 2,
        )
        pg.draw.line(self.screen, color, ul, ur, 1)
        pg.draw.line(self.screen, color, ul, bl, 1)
        pg.draw.line(self.screen, color, bl, br, 1)
        pg.draw.line(self.screen, color, ur, br, 1)

    def draw_boxes(self):
        for y in range(self.height):
            for x in range(self.width):
                self.draw_box(y, x, BLACK)

    def display(self):
        pg.init()
        self.screen = pg.display.set_mode((self.pixely, self.pixelx))
        pg.display.set_caption("Gridworld")
        # fill with white
        done = False
        block_size = 400 / self.height
        clock = pg.time.Clock()
        # agent = self.agent
        # If you want a background image, replace this clear with blit'ing the
        # background image.
        self.screen.fill(WHITE)

        # --- Drawing code should go here
        # self.draw_agent(agent)
        self.draw_boxes()
        # self.draw_rewards()
        pg.display.flip()
        # self.recording.append(self.start)
        reward = 0.0
        while not done:
            # action_val = GridWorldAction.right
            # --- Main event loop
            if self.is_human:
                cont = True
                while cont:
                    for event in pg.event.get():
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_LEFT:
                                action_val = GridWorldAction.left
                                cont = False
                                break
                            elif event.key == pg.K_RIGHT:
                                action_val = GridWorldAction.right
                                cont = False
                                break
                            elif event.key == pg.K_UP:
                                action_val = GridWorldAction.up
                                cont = False
                                break
                            elif event.key == pg.K_DOWN:
                                action_val = GridWorldAction.down
                                cont = False
                                break
                        if event.type == pg.QUIT:
                            done = True
                            cont = False
                            break
            else:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        done = True

            # --- Game logic should go here
            # if isinstance(agent, HumanAgent):
            # agent.take_action(action_val)
            # self.recording.append(action_val.value)
            # else:
            # 	agent.take_action(self.policy)
            # --- Screen-clearing code goes here

            # Here, we clear the screen to white. Don't put other drawing commands
            # above this, or they will be erased with this command.

            # If you want a background image, replace this clear with blit'ing the
            # background image.
            self.screen.fill(WHITE)

            # --- Drawing code should go here
            # self.draw_agent(agent)
            self.draw_boxes()
            # self.draw_rewards()
            # --- Go ahead and update the screen with what we've drawn.
            # reward += self.reward_mat[agent.get_loc()]
            # self.recording.append(agent.get_loc())
            # delay = 100
            # if agent.get_loc() == self.goal:
            # 	delay = 1000
            # 	done = True
            # save rewards
            # save recordings
            pg.display.flip()
            # --- Limit to 60 frames per second
            clock.tick(60)
            pg.time.delay(delay)
            # wait
        pg.quit()
        # return (reward, self.recording)

    def draw_agent(self, loc):
        block_size_y = (self.pixely - self.height * 2) / self.height
        block_size_x = (self.pixelx - self.width * 2) / self.width
        row = loc[0]
        col = loc[1]
        centerx = col * block_size_x + block_size_x / 2 + col * 2
        centery = row * block_size_y + block_size_y / 2 + row * 2
        pg.draw.circle(
            self.screen,
            AGENTCOLOR,
            (centerx, centery),
            (min(block_size_y, block_size_x) / 2) * 2 / 3,
            0,
        )

    def get_state_tuple(self, state):
        col = state % self.cols
        row = state / self.rows
        return (row, col)

    def get_state_value(self, state_tuple):
        return state_tuple[0] * self.rows + state_tuple[1]

    def get_next_state(self, state, action):
        for s in range(np.shape(self.mdp.transitions)[0]):
            if self.mdp.transitions[state][action][s] == 1:
                return s
                break

    def draw_rewards(self):
        block_size_y = (self.pixely - self.height * 2) / self.height
        block_size_x = (self.pixelx - self.width * 2) / self.width
        font_size = (block_size_x * block_size_y) / 800
        for row in range(self.height):
            for col in range(self.width):
                state = self.get_state_value((row, col))
                FONT = pg.font.SysFont(
                    "monospace", int(font_size)
                )  # SysFont creates a font object from available pygame fonts
                color = RED
                if self.mdp.rewards[state] == 0:
                    color = BLACK
                elif self.mdp.rewards[state] > -1:
                    color = BLUE
                str_reward = str(
                    decimal.Decimal(str(self.mdp.rewards[state])).quantize(
                        decimal.Decimal(10) ** -2
                    )
                )
                SURFACEFONT = FONT.render(
                    "" + str_reward, True, color, WHITE
                )  # True is for anti-aliasing, looks better when true
                if self.goal == (row, col):
                    SURFACEFONT = FONT.render(
                        "" + str_reward, True, color, GOALCOLOR
                    )  # True is for anti-aliasing, looks better when true
                elif self.start == (row, col):
                    SURFACEFONT = FONT.render("" + str_reward, True, color, STARTCOLOR)
                SURFACER = (
                    SURFACEFONT.get_rect()
                )  # meaning SURFACER will gain rectangular values
                centerx = col * block_size_x + block_size_x / 2 + col * 2
                centery = row * block_size_y + block_size_y / 2 + row * 2
                SURFACER.center = (centerx, centery)
                self.screen.blit(SURFACEFONT, SURFACER)

    def record_demo(self, confidence):
        # list of recordings
        demo = list()
        pg.init()
        self.screen = pg.display.set_mode((self.pixely, self.pixelx))
        pg.display.set_caption("Gridworld")
        # fill with white
        done = False
        block_size = 400 / self.height
        clock = pg.time.Clock()
        # agent = self.agent
        # If you want a background image, replace this clear with blit'ing the
        # background image.
        self.screen.fill(WHITE)
        # starting state
        state = self.start
        # --- Drawing code should go here
        self.draw_agent(self.get_state_tuple(state))
        self.draw_boxes()
        self.draw_rewards()
        pg.display.flip()
        # add starting state to the demonstration
        reward = 0.0
        while not done:
            # --- Main event loop
            cont = True
            while cont:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_LEFT:
                            action_val = 2
                            cont = False
                            break
                        elif event.key == pg.K_RIGHT:
                            action_val = 3
                            cont = False
                            break
                        elif event.key == pg.K_UP:
                            action_val = 0
                            cont = False
                            break
                        elif event.key == pg.K_DOWN:
                            action_val = 1
                            cont = False
                            break
                    if event.type == pg.QUIT:
                        done = True
                        cont = False
                        break

            demo.append((state, action_val))

            # --- Game logic should go here
            state = self.get_next_state(state, action_val)

            # add action to demonstration

            # --- Screen-clearing code goes here

            # Here, we clear the screen to white. Don't put other drawing commands
            # above this, or they will be erased with this command.

            # If you want a background image, replace this clear with blit'ing the
            # background image.
            self.screen.fill(WHITE)

            # --- Drawing code should go here
            self.draw_agent(self.get_state_tuple(state))
            self.draw_boxes()
            self.draw_rewards()
            # --- Go ahead and update the screen with what we've drawn.
            reward += self.mdp.rewards[state]
            # add new state to demonstration
            delay = 100
            if state == self.goal:
                delay = 1000
                done = True

            pg.display.flip()
            # --- Limit to 60 frames per second
            clock.tick(60)
            pg.time.delay(delay)
            # wait
        pg.quit()
        return (reward, demo, confidence)

    def play(self, policy):
        # list of recordings
        demo = list()
        pg.init()
        self.screen = pg.display.set_mode((self.pixely, self.pixelx))
        pg.display.set_caption("Gridworld")
        # fill with white
        done = False
        block_size = 400 / self.height
        clock = pg.time.Clock()
        # agent = self.agent
        # If you want a background image, replace this clear with blit'ing the
        # background image.
        self.screen.fill(WHITE)
        # starting state
        state = self.start
        # --- Drawing code should go here
        self.draw_agent(self.get_state_tuple(state))
        self.draw_boxes()
        self.draw_rewards()
        pg.display.flip()
        # add starting state to the demonstration
        reward = 0.0
        while not done:
            # --- Main event loop
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                    cont = False
                    break

            action_val = policy[state]

            demo.append((state, action_val))

            # --- Game logic should go here
            state = self.get_next_state(state, action_val)

            # add action to demonstration

            # --- Screen-clearing code goes here

            # Here, we clear the screen to white. Don't put other drawing commands
            # above this, or they will be erased with this command.

            # If you want a background image, replace this clear with blit'ing the
            # background image.
            self.screen.fill(WHITE)

            # --- Drawing code should go here
            self.draw_agent(self.get_state_tuple(state))
            self.draw_boxes()
            self.draw_rewards()
            # --- Go ahead and update the screen with what we've drawn.
            reward += self.mdp.rewards[state]
            # add new state to demonstration
            delay = 100
            if state == self.goal:
                delay = 1000
                done = True

            pg.display.flip()
            # --- Limit to 60 frames per second
            clock.tick(60)
            pg.time.delay(delay)
            # wait
        pg.quit()
        return (reward, demo)

    def record(self, num_episodes):
        demos = list()
        confidence = 400.0
        for i in range(num_episodes):
            demo = self.record_demo(confidence)
            demos.append(demo)
        return demos

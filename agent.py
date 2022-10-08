import torch
import random
import numpy as np
import cv2
from collections import deque

import snake_game
from model import Linear_QNet, Value, Actor, QTrainer, PGTrainer
from helper import plot

from sys import getsizeof


class Agent:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        # Buffer
        self.MAX_MEMORY = 100_000
        self.memory = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.n_games = 0

        # Models
        # self.model = Linear_QNet()
        # self.model.load_state_dict(torch.load("./model/lin_q.pth", map_location=torch.device('cpu')))
        # self.model = self.model.to(self.device)

        self.policy = Actor()
        # self.policy.load_state_dict(torch.load("./model/actor.pth", map_location=torch.device('cpu')))
        self.policy = self.policy.to(self.device)

        self.v = Value()
        # self.V.load_state_dict(torch.load("./model/value.pth", map_location=torch.device('cpu')))
        self.v = self.v.to(self.device)

        # Params
        self.LR_policy = 3e-4
        self.LR_v = 3e-04
        self.gamma = 0.95  # discount rate
        self.BATCH_SIZE = 256

        self.epsilon = 15  # randomness
        self.epsilon_decay = 1e-05
        # self.trainer = QTrainer(self.model, lr=self.LR_policy, gamma=self.gamma)
        self.trainer_pg = PGTrainer(self.policy, self.v, lr_policy=self.LR_policy, lr_v=self.LR_v, gamma=self.gamma)

        print(count_parameters(self.policy))
        print(count_parameters(self.v))

    @staticmethod
    def get_state(game):
        head = game.snake[0]

        point_l = [head[0] - 20, head[1]]
        point_r = [head[0] + 20, head[1]]
        point_u = [head[0], head[1] - 20]
        point_d = [head[0], head[1] + 20]

        dir_l = game.direction == 'left'
        dir_r = game.direction == 'right'
        dir_u = game.direction == 'up'
        dir_d = game.direction == 'down'

        danger_straight = (dir_r and game.is_collision(point_r)) or \
                          (dir_l and game.is_collision(point_l)) or \
                          (dir_u and game.is_collision(point_u)) or \
                          (dir_d and game.is_collision(point_d))

        danger_right = (dir_u and game.is_collision(point_r)) or \
                       (dir_d and game.is_collision(point_l)) or \
                       (dir_l and game.is_collision(point_u)) or \
                       (dir_r and game.is_collision(point_d))

        danger_left = (dir_d and game.is_collision(point_r)) or \
                      (dir_u and game.is_collision(point_l)) or \
                      (dir_r and game.is_collision(point_u)) or \
                      (dir_l and game.is_collision(point_d))

        state = [
            # Dangers
            danger_straight,
            danger_right,
            danger_left,

            # Move direction
            dir_r,
            dir_u,
            dir_l,
            dir_d,

            # Food location
            game.food[0][0] < game.head[0],  # food left
            game.food[0][0] > game.head[0],  # food right
            game.food[0][1] < game.head[1],  # food up
            game.food[0][1] > game.head[1]  # food down
        ]
        # print(state)
        return np.array(state, dtype=int)

    @staticmethod
    def get_state_pixels(game):
        frame = game.frame
        block_size = 20
        frame_small = cv2.resize(frame, (18, 18))
        frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        frame_small = frame_small.astype(np.float64) / 255.
        # print(frame_small)

        frame_small = np.expand_dims(frame_small, axis=-1)
        frame_small = np.transpose(frame_small, (2, 0, 1))  # conv

        return frame_small

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def forget(self):
        self.memory.clear()

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
        else:
            mini_sample = random.sample(self.memory, len(self.memory))

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_pg(self):
        mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer_pg.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state, game):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.epsilon * (1 - self.epsilon_decay)
        if self.epsilon < 3:
            self.epsilon = 3

        if random.randint(0, 100) < self.epsilon:
            direction = game.direction

            if direction == 'right':
                new_dir = np.random.choice(["up", "down"], 1)[0]
            if direction == 'up':
                new_dir = np.random.choice(["right", "left"], 1)[0]
            if direction == 'left':
                new_dir = np.random.choice(["up", "down"], 1)[0]
            if direction == 'down':
                new_dir = np.random.choice(["right", "left"], 1)[0]

            final_move = self.move_str2list(new_dir)

        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # conv

            prediction = self.model(state_tensor)
            final_move = prediction

        return final_move

    def get_action_pg(self, state, game):
        self.epsilon = self.epsilon * (1 - self.epsilon_decay)
        if self.epsilon < 3:
            self.epsilon = 3

        # if random.randint(0, 100) < self.epsilon:
        #     direction = game.direction
        #
        #     if direction == 'right':
        #         new_dir = np.random.choice(["up", "down"], 1)[0]
        #     if direction == 'up':
        #         new_dir = np.random.choice(["right", "left"], 1)[0]
        #     if direction == 'left':
        #         new_dir = np.random.choice(["up", "down"], 1)[0]
        #     if direction == 'down':
        #         new_dir = np.random.choice(["right", "left"], 1)[0]
        #
        #     final_move = self.move_str2list(new_dir)
        #
        # else:

        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)  # conv
        prediction = self.policy(state_tensor)

        move = prediction.sample().detach()
        log_prob = prediction.log_prob(move)

        return move, log_prob

    def move_prediction2str(self, prediction=None, move=None):

        if prediction is not None:
            idx = torch.argmax(prediction)
        if move is not None:
            idx = move

        move_list = [0, 0, 0, 0]
        move_list[idx] = 1

        # [Right, Up, Left, Down]
        if move_list == [1, 0, 0, 0]:
            move_str = 'right'
        elif move_list == [0, 1, 0, 0]:
            move_str = 'up'
        elif move_list == [0, 0, 1, 0]:
            move_str = 'left'
        elif move_list == [0, 0, 0, 1]:
            move_str = 'down'
        else:
            raise ValueError('list to str error')
        return move_str

    def move_str2list(self, str):
        # [Right, Up, Left, Down]
        if str == 'right':
            move_list = torch.tensor([1, 0, 0, 0])
        elif str == 'up':
            move_list = torch.tensor([0, 1, 0, 0])
        elif str == 'left':
            move_list = torch.tensor([0, 0, 1, 0])
        elif str == 'down':
            move_list = torch.tensor([0, 0, 0, 1])
        else:
            raise ValueError('list to str error')
        return move_list


def train_ql():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game.SnakeGameAI(food_number=1)

    while True:
        # get old state
        state_old = agent.get_state(game)
        # state_old = agent.get_state_pixels(game)

        # get move
        final_move = agent.get_action(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(agent.move_prediction2str(final_move),
                                             visuals=True,
                                             food_number=1)

        # print(f"move: {agent.move_prediction2str(final_move)}")
        # cv2.waitKey(0)

        state_new = agent.get_state(game)
        # state_new = agent.get_state_pixels(game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.model.save()
            if agent.n_games % 50 == 0:
                agent.model.save()
            if agent.n_games % 20 == 0:
                for _ in range(20):
                    agent.train_long_memory()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

            if agent.n_games % 1 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'AVG:', mean_score,
                      'Memory:', len(agent.memory), 'Exploration:', agent.epsilon)


def train_pg():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game.SnakeGameAI(food_number=1)

    while True:
        # get old state
        # state_old = agent.get_state_pixels(game)
        state_old = agent.get_state(game)

        # get move
        # final_move, log_prob = agent.get_action_pg(state_old, game)
        final_move, log_prob = agent.get_action_pg(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(agent.move_prediction2str(move=final_move),
                                             visuals=True,
                                             food_number=1)

        state_new = agent.get_state(game)

        # remember
        # agent.remember(state_old, final_move, reward, log_prob, done)
        agent.remember(state_old, log_prob, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            if agent.n_games % 50 == 0:
                agent.policy.save()
                agent.v.save()

            if len(agent.memory) > agent.BATCH_SIZE:
                for _ in range(1):
                    agent.train_pg()
                    agent.forget()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

            if agent.n_games % 1 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'AVG:', mean_score, 'Memory:', len(agent.memory))
                print('=======================================================')

            # agent.forget()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # train_ql()
    train_pg()

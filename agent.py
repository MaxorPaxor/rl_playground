import torch
import random
import numpy as np
from collections import deque
import snake_game
from model import Linear_QNet, QTrainer, Conv_QNet
from helper import plot

from sys import getsizeof

MAX_MEMORY = 20_000
BATCH_SIZE = 500
LR = 0.001


class Agent:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=int(MAX_MEMORY))  # popleft()
        # self.memory_2 = deque(maxlen=int(MAX_MEMORY/4))
        # self.memory_3 = deque(maxlen=int(MAX_MEMORY/4))
        # self.memory_4 = deque(maxlen=int(MAX_MEMORY/4))
        # self.model = Linear_QNet(11, 256, 4)
        self.model = Conv_QNet()
        # self.model.load_state_dict(torch.load("./model/model_conv.pth", map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        print("Device: {}".format(self.device))

    def get_state(self, game):
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
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food[0] < game.head[0],  # food left
            game.food[0] > game.head[0],  # food right
            game.food[1] < game.head[1],  # food up
            game.food[1] > game.head[1]  # food down
        ]

        return np.array(state, dtype=int)

    def get_state_pixels(self, game):
        frame = game.frame
        BLOCK_SIZE = 20
        frame_small = frame[::BLOCK_SIZE, ::BLOCK_SIZE, :]
        return frame_small

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.n_games < 3000:
            self.epsilon = 10
        elif self.n_games < 10000:
            self.epsilon = 5
        else:
            self.epsilon = 0
        final_move = [0, 0, 0, 0]  # [Right, Up, Left, Down]
        if random.randint(0, 100) < self.epsilon:  # 30% random moves
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state = np.transpose(state, (2, 0, 1))  # conv
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # conv

            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def move_list2str(self, move_list):
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


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game.SnakeGameAI(food_number=10)

    while True:
        # get old state
        # state_old = agent.get_state(game)
        state_old = agent.get_state_pixels(game)
        # state_old = np.expand_dims(state_old, axis=0)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(agent.move_list2str(final_move),
                                             visuals=True,
                                             food_number=10)

        state_new = agent.get_state_pixels(game)
        # state_new = np.expand_dims(state_new, axis=0)

        state_old = np.transpose(state_old, (2, 0, 1))  # conv
        state_new = np.transpose(state_new, (2, 0, 1))  # conv

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            #plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'AVG:', mean_score)
            print("memory len: {}".format(len(agent.memory)))


if __name__ == '__main__':
    train()

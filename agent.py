import torch
import random
import numpy as np
from collections import deque
import snake_game
from model import Linear_QNet, ConvNet1, ConvNet4, QTrainer, PGTrainer
from helper import plot

from sys import getsizeof


class Agent:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        # Buffer
        self.MAX_MEMORY = 1000
        self.memory = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.n_games = 0

        # Models
        # self.model = Linear_QNet(11, 256, 4)
        # self.model = ConvNet4()
        # self.model.load_state_dict(torch.load("./model/model_conv.pth", map_location=torch.device('cpu')))
        # self.model = self.model.to(self.device)

        self.policy = ConvNet4()
        self.policy.load_state_dict(torch.load("./model/Policy_ConvNet4.pth", map_location=torch.device('cpu')))
        self.policy = self.policy.to(self.device)

        self.V = ConvNet1()
        self.V.load_state_dict(torch.load("./model/V_ConvNet1.pth", map_location=torch.device('cpu')))
        self.V = self.V.to(self.device)

        # Params
        self.LR_policy = 5e-08
        self.LR_V = 1e-04
        self.gamma = 0.9  # discount rate
        self.BATCH_SIZE = 1024
        self.epsilon = 20  # randomness
        # self.trainer = QTrainer(self.model, lr=self.LR_policy, gamma=self.gamma)
        self.trainer_pg = PGTrainer(self.policy, self.V, lr_policy=self.LR_policy, lr_V=self.LR_V, gamma=self.gamma)

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
        block_size = 20
        frame_small = frame[::block_size, ::block_size, :]
        return frame_small

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def forget(self):
        self.memory.clear()

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_pg(self):
        traj = self.memory
        states, actions, rewards, next_states, dones = zip(*traj)
        self.trainer_pg.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state, game):
        # random moves: tradeoff exploration / exploitation
        if len(game.snake) > 10:
            self.epsilon = 20

        elif len(game.snake) > 5:
            self.epsilon = 15

        else:
            self.epsilon = 10

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

    def get_action_pg(self, state, game):

        final_move = [0, 0, 0, 0]  # [Right, Up, Left, Down]

        state = np.transpose(state, (2, 0, 1))  # conv
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)  # conv
        prediction = self.policy(state_tensor)

        if random.randint(0, 100) < self.epsilon:  # 30% random moves
            move = random.randint(0, 3)
            log_prob = torch.log(prediction.squeeze(0)[move])
            final_move[move] = 1

        else:
            move = np.random.choice(4, p=np.squeeze(prediction.detach().numpy()))
            log_prob = torch.log(prediction.squeeze(0)[move])
            final_move[move] = 1

        print("Preds: {}".format(prediction))

        return final_move, log_prob

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


def train_ql():
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
        final_move = agent.get_action(state_old, game)

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
            if agent.n_games % 50 == 0:
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

            if agent.n_games % 1 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'AVG:', mean_score, 'Memory:', len(agent.memory))


def train_pg():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game.SnakeGameAI(food_number=10)

    while True:
        # get old state
        state_old = agent.get_state_pixels(game)

        # get move
        final_move, log_prob = agent.get_action_pg(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(agent.move_list2str(final_move),
                                             visuals=True,
                                             food_number=10)

        # remember
        state_old = np.transpose(state_old, (2, 0, 1))  # conv
        agent.remember(state_old, final_move, reward, log_prob, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_pg()

            if score > record:
                record = score

            if agent.n_games % 50 == 0:
                agent.policy.save()
                agent.V.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

            if agent.n_games % 1 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'AVG:', mean_score, 'Memory:', len(agent.memory))

            agent.forget()


if __name__ == "__main__":
    # train_ql()
    train_pg()

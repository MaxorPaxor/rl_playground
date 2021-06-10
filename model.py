import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Conv_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # image size 640x640x3 reduced to 32x32x3
        # image size 320x320x3 reduced to 16x16x3 or x1 for grey

        self.conv1 = nn.Conv2d(3, 16, 5)  # 12*12*10
        self.conv2 = nn.Conv2d(16, 32, 5)  # 8*8*20
        self.conv3 = nn.Conv2d(32, 64, 5)  # 4*4*30 -> 2*2*30
        self.fc1 = nn.Linear(2 * 2 * 64, 128)
        self.fc2 = nn.Linear(128, 4)
        # self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model_conv.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


def compare_models(model_1, model_2):
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            print('Models mismatch')
            return
    print('Models match perfectly! :)')


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.batch_num = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)  # [29, 3, 32, 32]
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)  # [29, 3, 32, 32]
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = self.target_model(state)
        # (n, 4) where the second dimension is the Q value for that action
        # pred = Q(si, ai) = [Q_right, Q_up, Q_left, Q_down]_i

        if self.batch_num % 50 == 0:
            # compare_models(self.target_model, self.model)
            print("Updated model")
            self.target_model = copy.deepcopy(self.model)
            # compare_models(self.target_model, self.model)

        # go through all steps
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx].unsqueeze(0)))
                # yi = r(s,a) + gamma * max_ai' ( Q(si', ai') )

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # action[idx] = [0 0 1 0]
            # torch.argmax(action[idx]).item() = 2
            # target[idx] = [Q_right, Q_up, Q_left, Q_down]_idx
            # target[idx][2] = Q_new = [Q_right, Q_up, _Q_new_, Q_down]_idx

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        self.batch_num += 1


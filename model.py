import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import distributions
import os
import copy
import numpy as np

torch.autograd.set_detect_anomaly(True)


class Linear_QNet(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.fc1 = nn.Linear(18 * 18, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 4)

        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)
        # x = self.dropout(x)
        return x

    def save(self, file_name='lin_q.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # image size 640x640x3 reduced to 32x32x3
        # image size 360x360x3 reduced to 18x18x3 or x1 for grey

        # self.conv1 = nn.Conv2d(3, 8, 1)  # 18*18*8
        # self.bn1 = nn.BatchNorm2d(16)

        # self.conv2 = nn.Conv2d(8, 12, 5)  # 8*8*12
        # self.bn2 = nn.BatchNorm2d(32)

        # self.conv3 = nn.Conv2d(12, 16, 3)  # 6*6*16
        # self.bn2 = nn.BatchNorm2d(32)

        # self.conv4 = nn.Conv2d(16, 32, 3)  # 6*6*96
        # self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv4(x))
        # x = self.bn4(x)
        #x = self.dropout(x)

        # x = F.max_pool2d(x, 2)

        # x = self.fc3(x)
        # x = distributions.Categorical(logits=x)

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)
        x = distributions.Categorical(logits=x.clone())
        # x = nn.LogSoftmax()(x)

        return x

    def save(self, file_name='actor.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        # image size 640x640x3 reduced to 32x32x3
        # image size 360x360x3 reduced to 18x18x3 or x1 for grey

        # self.conv1 = nn.Conv2d(3, 16, 5)  # 14*14*16
        # self.bn1 = nn.BatchNorm2d(16)

        # self.conv2 = nn.Conv2d(16, 32, 5)  # 10*10*32
        # self.bn2 = nn.BatchNorm2d(32)

        # self.conv3 = nn.Conv2d(32, 64, 3)  # 8*8*64
        # self.bn2 = nn.BatchNorm2d(32)

        # self.conv4 = nn.Conv2d(64, 96, 3)  # 6*6*96
        # self.bn2 = nn.BatchNorm2d(32)

        # self.fc1 = nn.Linear(4 * 4 * 64, 64)
        # self.fc2 = nn.Linear(64, 1)
        # self.fc3 = nn.Linear(16, 1)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        #x = self.dropout(x)

        # x = F.relu(self.conv4(x))
        # x = self.bn4(x)
        #x = self.dropout(x)

        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        # x = self.fc1(x)
        # x = F.relu(x)
        #x = self.dropout(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        #x = self.dropout(x)

        # x = self.fc3(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)

        return x

    def save(self, file_name='value.pth'):
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
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.criterion = nn.MSELoss()
        self.batch_num = 0

    def update_network_parameters(self, tau=0.001):  # tau=0.03 works best
        # Network params
        model_params = self.model.named_parameters()
        target_model_params = self.target_model.named_parameters()

        model_params_dict = dict(model_params)
        target_model_params_dict = dict(target_model_params)

        # Network buffers
        model_buffers = self.model.named_buffers()
        target_model_buffers = self.target_model.named_buffers()

        model_buffers_dict = dict(model_buffers)
        target_model_buffers_dict = dict(target_model_buffers)

        # Update params
        for name in model_params:
            model_params_dict[name] = tau * model_params_dict[name].clone() + \
                                      (1 - tau) * target_model_params_dict[name].clone()

        # Update buffers
        for name in model_buffers_dict:
            model_buffers_dict[name] = tau * model_buffers_dict[name].clone() + \
                                       (1 - tau) * target_model_buffers_dict[name].clone()

        self.target_model.load_state_dict(model_params_dict)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)  # [N, 3, 16+2, 16+2]
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)  # [N, 3, 16+2, 16+2]
        # action = torch.tensor(action, dtype=torch.long).to(self.device)  # [1, 4]
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Normalize rewards
        if len(reward) > 1:
            reward = (reward - reward.mean()) / (reward.std() + 1e-9)  # normalize discounted rewards

        pred = self.model(state)
        target = self.target_model(state)

        # go through all steps
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx].unsqueeze(0)))
                # yi = r(s,a) + gamma * max_ai' ( Q(si', ai') )

            target = target.clone()
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        print(f"Loss:{loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        self.batch_num += 1

        self.update_network_parameters()


class PGTrainer:
    def __init__(self, policy, v, lr_policy, lr_v, gamma):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.policy = policy
        self.v = v
        self.target_v = copy.deepcopy(self.v)

        self.optimizer_v = optim.Adam(self.v.parameters(), lr=lr_v)
        self.criterion_v = nn.MSELoss()

        self.optimizer_PG = optim.Adam(self.policy.parameters(), lr=lr_policy)

        self.batch_num = 0

    def train_step(self, state, log_prob_, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)  # [N, 3, 16+2, 16+2]
        log_prob = torch.stack(log_prob_)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)  # [N, 3, 16+2, 16+2]

        # Normalize rewards
        if len(reward) > 1:
            reward = (reward - reward.mean()) / (reward.std() + 1e-9)  # normalize discounted rewards

        # 1: Fit V(s)
        if self.batch_num % 20 == 0:
            self.target_v = copy.deepcopy(self.v)

        self.policy.eval()
        self.v.eval()
        self.target_v.eval()

        pred_v = self.v.forward(state)
        target_v = self.target_v.forward(state).detach()
        for idx in range(len(done)):
            v_new = reward[idx]
            if not done[idx]:
                v_new = reward[idx] + self.gamma * self.target_v.forward(next_state[idx].unsqueeze(0))

            target_v = target_v.clone()
            target_v[idx] = v_new

        self.v.train()
        self.optimizer_v.zero_grad()
        loss_V = self.criterion_v(target_v, pred_v)
        loss_V.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), .25)
        self.optimizer_v.step()
        self.v.eval()

        # 2: Evaluate A(s, a)
        A = target_v - pred_v
        A = A.squeeze(1).detach()

        # 3: Calc Policy Gradient
        self.policy.train()
        self.optimizer_PG.zero_grad()
        loss_PG = -1 * torch.mean(log_prob * A)
        print("loss_PG: {}".format(loss_PG))
        loss_PG.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .25)
        self.optimizer_PG.step()
        self.policy.eval()

        # for i in self.policy.parameters():
        #     print(i)

        self.batch_num += 1

from model import ActorCritic
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Source of Model Structure: https://github.com/yenchenlin/DeepLearningFlappyBird

class CNN(ActorCritic):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, (8, 8), 4, 2)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 2)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, 256)
        self.action_layer = nn.Linear(256, 2)
        self.value_layer = nn.Linear(256, 1)

        torch.nn.init.zeros_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv3.weight)

        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.action_layer.weight)
        torch.nn.init.zeros_(self.value_layer.weight)

    def model(self, state):
        state = torch.from_numpy(state).float().permute(2, 0, 1)
        state = self.pool(F.relu(self.conv1(state)))
        state = self.pool(F.relu(self.conv2(state)))
        state = self.pool(F.relu(self.conv3(state)))

        state = torch.flatten(state)
        
        state = F.relu(self.fc1(state))
        
        return state
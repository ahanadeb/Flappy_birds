from model import ActorCritic
import torch.nn as nn
import torch
import torch.nn.functional as F

class NN(ActorCritic):
    def __init__(self):
        super(NN, self).__init__()
        # self.affine = nn.Linear(8, 128)
        self.affine = nn.Linear(2, 64)
        
        # self.action_layer = nn.Linear(128, 4)
        # self.value_layer = nn.Linear(128, 1)
        self.action_layer = nn.Linear(64, 2)
        self.value_layer = nn.Linear(64, 1)

    def model(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        return state
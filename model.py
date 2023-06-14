import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# adapted from https://github.com/Talendar/flappy-bird-gym

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = self.model(state)
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        # print("probs",action_probs)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        # print("action", action)

        # print(action_distribution)
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

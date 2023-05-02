from test import test
from model import ActorCritic
import torch
import torch.optim as optim
import gym
import time
import flappy_bird_gym

def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = True
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543
    
    torch.manual_seed(random_seed)
    env = flappy_bird_gym.make("FlappyBird-v0")

    print(env.action_space)
    #env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    
    policy = ActorCritic()
    print(policy)
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr,betas)
    
    running_reward = 0
    for i_episode in range(0, 10000):
        state = env.reset()
        for t in range(10000):
            # print(state)
            action = policy(state)
            # print("action", env.step(action))
            state, reward, done, info = env.step(action)
            reward = 0.01 * t + 10 * info["score"]
            if done:
                reward = -1
            policy.rewards.append(reward)
            if t>95:
                print(t, reward, done)
            running_reward += reward
            if render and i_episode % 1000 < 10 and False:
                env.render()
                time.sleep(1/30)
            if done:
                break
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()
        
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        
        if running_reward > 400000:
            torch.save(policy.state_dict(), './preTrained/Flappy_bird_save.pth'.format(lr, betas[0], betas[1]))
            print("########## Solved! ##########")
            test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            break
        
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
            
if __name__ == '__main__':
    train()

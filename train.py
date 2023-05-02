from test import test
from model import ActorCritic
import torch
import torch.optim as optim
import gym
import time
import flappy_bird_gym
from datetime import datetime
import pickle

def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = False
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

    rewards_over_episodes = []
    
    running_reward = 0
    for i_episode in range(0, 10000):
        state = env.reset()
        for t in range(10000):
            reward_per_episode = 0
            # print(state)
            action = policy(state)
            # print("action", env.step(action))
            state, reward, done, info = env.step(action)
            reward = 0.01 * t + 10 * info["score"] - 1 * state[1]
            # reward = 10 * info["score"]
            if done:
                reward = -10 * state[1]
                print("y", info["playery"])
                if info["playery"] > 380:
                    reward = -1000000000000
            policy.rewards.append(reward)
            running_reward += reward
            reward_per_episode += reward
            if render and i_episode % 100 < 10:
                env.render()
                time.sleep(1/30)
            if done:
                break

        rewards_over_episodes.append(reward_per_episode)
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()
        
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))

        if i_episode % 100 == 0:
            running_reward = running_reward/100
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
    
    with open('./preTrained/FlappyBird_rewards_{}.txt'.format(str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")), 'wb') as file:
        pickle.dump(rewards_over_episodes, file)

    torch.save(policy.state_dict(), './preTrained/FlappyBird_{}.pth'.format(str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")))
    print("########## Solved! ##########")
    # test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            
if __name__ == '__main__':
    train()

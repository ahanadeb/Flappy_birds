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

    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543
    
    torch.manual_seed(random_seed)
    env = flappy_bird_gym.make("FlappyBird-v0")

    print(env.action_space)
    env.seed(random_seed)
    
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

    rewards_over_episodes = []
    
    running_reward = 0
    for i_episode in range(0, 10000):
        state = env.reset()
        reward_per_episode = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, info = env.step(action)

            # give low reward for coming far and high reward for each passed tube
            # punishing being far away from the next tubes gap
            reward = 0.01 * t + 10 * info["score"] - 1 * state[1]
            if done:
                # when failed punishing tries with higher distance to the gap
                reward = -10 * state[1]
                if info["playery"] > 380:
                    # punish hitting the ground with a high negative reward
                    reward = -100

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
        
        # count the good runs in a row
        if reward_per_episode > 1000000:
            counter_good_runs_in_a_row += 1
            print("good run number", counter_good_runs_in_a_row)
        else:
            # reset counter
            counter_good_runs_in_a_row = 0

        # save the model when there are 3 good runs in a row
        if counter_good_runs_in_a_row >= 3:
            with open('./preTrained/FlappyBird_rewards_{}.txt'.format(str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")), 'wb') as file:
                pickle.dump(rewards_over_episodes, file)

            torch.save(policy.state_dict(), './preTrained/FlappyBird_{}.pth'.format(str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")))
            print("########## Solved! ##########")
            break

        if i_episode % 100 == 0:
            running_reward = running_reward/100
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
            
if __name__ == '__main__':
    train()

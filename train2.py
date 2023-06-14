from test import test
from model import ActorCritic
from model2 import Sarsa
import torch
import torch.optim as optim
import gym
import time
import flappy_bird_gym
from datetime import datetime
import pickle

def train2():

    render = True
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543
    
    torch.manual_seed(random_seed)
    env = flappy_bird_gym.make("FlappyBird-v0")

    print(env.action_space)
    env.seed(random_seed)
    
    policy = Sarsa()
    epsilon =0.9
   
   # optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

    rewards_over_episodes = []
    lens=[]
    running_reward = 0
    for i_episode in range(0, 5000):
        state = env.reset()
        reward_per_episode = 0
        qs=policy.net(torch.Tensor(state))
        action=policy.choose_action(qs)
       
        for t in range(1000000):
            policy.epsilon =epsilon 
            state_, reward, done, info = env.step(action)
            reward = 0.01 * t + 10 * info["score"] - 1 * state[1]
            if done:
                # when failed punishing tries with higher distance to the gap
                reward = -10* state[1]
                if info["playery"] > 380:
                    # punish hitting the ground with a high negative reward
                    reward = -100

            #policy.rewards.append(reward)
            running_reward += reward
            reward_per_episode += reward


         #   if render and i_episode % 100 == 0:
          #      env.render()
          #      time.sleep(1/30)
            
            action=policy.learn(state,action,state_,reward,done)
            state=state_
            if done:
                break


        if i_episode*0 == 0:
            policy.epsilon = 1
            st = env.reset()
            reward_per_episode1=0
            qs1=policy.net(torch.Tensor(st))
            a=policy.choose_action(qs)
            for t1 in range(1000000):
                
                st_, reward1, done1, info = env.step(a)
                reward = 0.01 * t + 10 * info["score"] - 1 * st[1]
                if done1:
                    # when failed punishing tries with higher distance to the gap
                    reward1= -10* st[1]
                    if info["playery"] > 380:
                        # punish hitting the ground with a high negative reward
                        reward1 = -100
                reward_per_episode1=reward_per_episode1+reward1

                qs1=policy.net(torch.Tensor(st_))
                a=policy.greedy_action(qs1)
                st=st_
                if done1:
                    break
            #policy.rewards.append(reward)
            running_reward += reward
            reward_per_episode += reward
            rewards_over_episodes.append(reward_per_episode1)
            lens.append(t1)
           
            # env.render()
            # time.sleep(1/30)
                    
        # Updating the policy :
     #   optimizer.zero_grad()
     #   loss = policy.calculateLoss(gamma)
     #   loss.backward()
     #   optimizer.step()        
     #   policy.clearMemory()
        
        # count the good runs in a row
        if reward_per_episode > 1000000:
            counter_good_runs_in_a_row += 1
            print("good run number", counter_good_runs_in_a_row)
        else:
            # reset counter
            counter_good_runs_in_a_row = 0

        # save the model when there are 3 good runs in a row
        #if counter_good_runs_in_a_row >= 1:
        if i_episode == 5000-2:
            with open('./preTrained/sarsalens_{}.txt'.format(str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")), 'wb') as file:
                pickle.dump(lens, file)

    
            break

        if i_episode % 100 == 0:
            running_reward = running_reward/100
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
            
            
if __name__ == '__main__':
    train2()

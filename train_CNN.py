from test_NN import test
from CNN import CNN
import torch
import torch.optim as optim
import gym
import time
import flappy_bird_gym
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

def preprocess_img(img):
    img[img == 200] = 0
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j, 0] == 200 and img[i, j, 1] == 200 and img[i, j, 2] == 200:
    #             img[i, j, 0] = 0
    #             img[i, j, 1] = 0
    #             img[i, j, 2] = 0
    x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    return x_t

def train():

    render = True
    use_initialized = True
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543
    
    torch.manual_seed(random_seed)
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")

    print(env.action_space)
    env.seed(random_seed)
    
    policy = CNN()

    if use_initialized:
        policy.load_state_dict(torch.load('./preTrained/FlappyBird_autosave_20000_2023-05-08_22_31_28_706821.pth'))

    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

    rewards_over_episodes = []
    
    running_reward = 0
    last_runs_t = []
    for i_episode in range(0, 100000):
        state = env.reset()

        x_t = preprocess_img(state)
        state = np.stack((x_t, x_t, x_t, x_t), axis=2)

        reward_per_episode = 0
        last_actions = []
        for t in range(10000):
            action = policy(state)
            last_actions.append(action)
            if len(last_actions) > 120:
                last_actions.pop(0)
            img, reward, done, info = env.step(action)

            # plt.imshow(img)
            # plt.show()

            x_t = preprocess_img(img)

            # plt.imshow(x_t)
            # plt.show()

            x_t1 = np.reshape(x_t, (80, 80, 1))
            state = np.append(x_t1, state[:, :, :3], axis=2)

            # if t % 4 == 0:
            #     f, axarr = plt.subplots(2,2)
            #     axarr[0,0].imshow(state[:,:,0], cmap='gray')
            #     axarr[0,1].imshow(state[:,:,1], cmap='gray')
            #     axarr[1,0].imshow(state[:,:,2], cmap='gray')
            #     axarr[1,1].imshow(state[:,:,3], cmap='gray')
                # plt.show()


            # give low reward for coming far and high reward for each passed tube
            # punishing being far away from the next tubes gap
            reward = 0.005 * t + 100 * info["score"]
            if info["playery"] < 0:
                reward -= 10
            if done:
                reward = -1
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
                last_runs_t.append(t)
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

        if i_episode % 10000 == 0 and i_episode > 0:
            with open('./preTrained/FlappyBird_autosave_{}_rewards_{}.txt'.format(i_episode ,str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")), 'wb') as file:
                pickle.dump(rewards_over_episodes, file)

            torch.save(policy.state_dict(), './preTrained/FlappyBird_autosave_{}_{}.pth'.format(i_episode, str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")))
            print("########## Autosaved! ##########")

        if i_episode % 100 == 0:
            running_reward = running_reward/100
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, sum(last_runs_t)/len(last_runs_t), running_reward))
            running_reward = 0
            last_runs_t = []
            
if __name__ == '__main__':
    train()

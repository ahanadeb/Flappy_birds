from NN import NN
import torch
import gym
import time
import flappy_bird_gym

from PIL import Image

def test(n_episodes=5, name='FlappyBird_2023-05-03_11_42_53_258562.pth'):
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    policy = NN()
    
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 time.sleep(1/30)
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
            
if __name__ == '__main__':
    test()

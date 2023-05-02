import matplotlib.pyplot as plt
import pickle

FILENAME = "./preTrained/FlappyBird_rewards_2023-05-02_18_20_03_457597.txt"

with open(FILENAME, 'rb') as file:
    rewards_over_episodes = pickle.load(file)   

plt.plot(rewards_over_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

plt.semilogx(rewards_over_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
import matplotlib.pyplot as plt
import pickle

FILENAME = "./preTrained/FlappyBird_rewards_2023-05-03_00_13_41_212170.txt"

with open(FILENAME, 'rb') as file:
    rewards_over_episodes = pickle.load(file)  

print(len(rewards_over_episodes))
print(rewards_over_episodes) 

plt.plot(rewards_over_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

plt.semilogx(rewards_over_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
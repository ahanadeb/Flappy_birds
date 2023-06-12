import matplotlib.pyplot as plt
import pickle

FILENAME = "./preTrained/FlappyBird_rewards_2023-06-12_11_23_09_839505.txt"

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
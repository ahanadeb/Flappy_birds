import matplotlib.pyplot as plt
import pickle
import numpy as np
FILENAME = "./preTrained/FlappyBird_rewards_2023-06-12_16_16_47_245936.txt"
FILENAME ="/Users/ahanadeb/Downloads/ac50003_2023-06-13_01_55_19_016277.txt"
FILENAME2="/Users/ahanadeb/Documents/books/RL/FB/preTrained/sarsa_2023-06-13_04_13_40_560785.txt"
FILENAME2="/Users/ahanadeb/Documents/books/RL/FB/preTrained/sarsa500_2023-06-13_11_50_34_704260.txt"
with open(FILENAME, 'rb') as file:
    rewards_over_episodes = pickle.load(file)  


with open(FILENAME2, 'rb') as file:
    rewards_over_episodes2 = pickle.load(file)   

print(len(rewards_over_episodes))

print(len(rewards_over_episodes2))
n = np.zeros((5000,1))
j=0
for i in range(5000):
    if i%100 == 0:
        n[i]=rewards_over_episodes2[j]
        j=j+1
n=rewards_over_episodes2

#print(rewards_over_episodes) 
#rewards_over_episodes2 = rewards_over_episodes2[:len(rewards_over_episodes)]

plt.semilogy(rewards_over_episodes, color='b', zorder=2)
plt.semilogy(n, color='r',zorder=1)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

plt.semilogx(rewards_over_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
import matplotlib.pyplot as plt
import pickle
import numpy as np
#FILENAME = "./preTrained/FlappyBird_rewards_2023-06-12_16_16_47_245936.txt"
FILENAME ="/Users/ahanadeb/Downloads/ac50003_2023-06-13_01_55_19_016277.txt"
FILENAME2="/Users/ahanadeb/Documents/books/RL/FB/preTrained/sarsa_2023-06-13_04_13_40_560785.txt"
FILENAME2="/Users/ahanadeb/Documents/books/RL/FB/preTrained/sarsa500_2023-06-13_11_50_34_704260.txt"
FILENAME2="/Users/ahanadeb/Downloads/2222ac50003_2023-06-14_09_35_40_286979.txt"
with open(FILENAME, 'rb') as file:
    rewards_over_episodes = pickle.load(file)  


with open(FILENAME2, 'rb') as file:
    rewards_over_episodes2 = pickle.load(file)   

print(len(rewards_over_episodes))

print(len(rewards_over_episodes2))
n = np.zeros((250,1))
j=0
su=0
for i in range(5000):
    su=su+rewards_over_episodes[i]
    if i%20 == 0:
        n[j]=su/20
        j=j+1
        su=0




#print(rewards_over_episodes) 
#rewards_over_episodes2 = rewards_over_episodes2[:len(rewards_over_episodes)]
x=np.arange(250)
x=x*20
plt.semilogy(x,n, color='b', zorder=2, label="64")
plt.semilogy(x,n2, color='r',zorder=1, label="2")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Average reward per episode")
plt.show()


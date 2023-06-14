import numpy as np
import matplotlib.pyplot as plt
import pickle


FILENAME ="/Users/ahanadeb/Documents/books/RL/FB/preTrained/sarsalens_2023-06-13_12_04_28_155163.txt"
FILENAME2="/Users/ahanadeb/Downloads/ac50003_2023-06-13_11_33_52_550987.txt"
with open(FILENAME, 'rb') as file:
    len_sarsa = pickle.load(file)  


with open(FILENAME2, 'rb') as file:
    len_ac = pickle.load(file)   


#print(rewards_over_episodes) 
#rewards_over_episodes2 = rewards_over_episodes2[:len(rewards_over_episodes)]
print(len_sarsa)
plt.plot(len_sarsa, color='b', zorder=2)
plt.plot(len_ac, color='r',zorder=1)
plt.xlabel("Episodes")
plt.ylabel("Episode lengths")
plt.show()
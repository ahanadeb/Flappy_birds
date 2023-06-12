import torch as th
import numpy as np


gamma = 0.99
lr = 0.02
episilon=0.9


class net(th.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=th.nn.Linear(2,10)
        self.out=th.nn.Linear(10,2)
    

    def forward(self,x):
        x=self.fc1(x)
        x=th.nn.functional.relu(x)
        out=self.out(x)
        return out

class Sarsa():
    def __init__(self):
        self.net,self.target_net=net(),net()
        self.iter_num=0
        self.optimizer=th.optim.Adam(self.net.parameters(),lr=lr)

    def learn(self,s,a,s_,r,done):
        eval_q=self.net(th.Tensor(s))[a]
        target_q=self.target_net(th.FloatTensor(s_))
        target_a=self.choose_action(target_q)
        target_q=target_q[target_a]
        if not done:
            y=gamma*target_q+r
        else:
            y=r
        loss=(y-eval_q)**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_num+=1
        if self.iter_num%10==0:
            self.target_net.load_state_dict(self.net.state_dict())
        return target_a

    def greedy_action(self,qs):
        return th.argmax(qs)

    def random_action(self):
        return np.random.randint(0,2)

    def choose_action(self,qs):
        if np.random.rand()>episilon:
            return self.random_action()
        else:
            return self.greedy_action(qs).tolist()

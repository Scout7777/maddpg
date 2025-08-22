
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self,shape_s_self,shape_a_self,alpha,fc1,fc2):
        super(ActorNetwork,self).__init__()

        self.fc1 = nn.Linear(shape_s_self, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, shape_a_self)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        a = torch.sigmoid(self.fc3(x))

        return a

class CriticNetwork(nn.Module):
    def __init__(self,sum_sa,beta,fc1,fc2):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Linear(sum_sa, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,sum_s,sum_a):
        x = torch.relu(self.fc1(torch.cat([sum_s, sum_a], dim=1)))
        x = torch.relu(self.fc2(x))
        Q = self.fc3(x)

        return Q

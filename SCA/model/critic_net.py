import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, in_size, mean=0, std =1, conf = None):
        super(Critic, self).__init__()
        if conf is not None:
            mean, std = conf["q_weight_init"]["mean"], conf["q_weight_init"]["std"]
            
        self.fc0 = nn.Linear(in_size, 256)
        self.fc0.weight.data.normal_(mean=mean, std =std)
        self.fc1 = nn.Linear(256, 1)
        self.fc1.weight.data.normal_(mean=mean, std =std)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        return -torch.sum(x) # (batch,1)
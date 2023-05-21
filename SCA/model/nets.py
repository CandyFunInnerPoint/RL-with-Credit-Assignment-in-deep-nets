import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_size = 784):
        super(Net, self).__init__()
        self.in_size = in_size
        self.layer0 = nn.Linear(in_size, 128)
        self.layer0_act = torch.relu

        self.layer1 = nn.Linear(128, 64)
        self.layer1_act = torch.relu

        self.final_layer = nn.Linear(64, 1)
        self.final_layer_act = torch.sigmoid

        self.layerlist = [self.layer0,self.layer1,self.final_layer]
        self.layer_act_list = [self.layer0_act,self.layer1_act,self.final_layer_act]

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.layer0(x)
        x = self.layer0_act(x)
        x = self.layer1(x)
        x = self.layer1_act(x)
        x = self.final_layer(x)
        return self.final_layer_act(x) # (batch, 1)


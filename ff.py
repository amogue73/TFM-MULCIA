import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FF(nn.Module):
    def __init__(self,num_sensors,layer1,layer2):
        super(FF,self).__init__()
        self.dense1 = nn.Linear(144*num_sensors,layer1)        
        self.dense2 = nn.Linear(layer1,layer2)
        self.dense3 = nn.Linear(layer2,num_sensors)
        self.name = "FF"

    def forward(self, x):
        x = rearrange(x, 'b h w -> b (h w)')
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        x = self.dense3(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CNN(nn.Module):
    """Modelo de pytorch de aprendizaje profundo.

    Se trata de una red neuronal convolucional, que recibe como input
    las tres series temporales de cada sensor en forma de matriz
    3x144. Cada fila es una serie temporal de un sensor. Las columnas
    son cada una de las mediciones, siendo en total 144.

    Se aplica un padding circular, copiando la última fila encima de la 
    primera y la primera debajo de la última, lo que le confiere al 
    input una topología cilíndrica. De esta manera, los tres sensores
    son tratados por igual.
    """
    def __init__(self,num_sensors,batch_size,channels,kernels,dense):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,channels[0],(num_sensors,kernels[0]))
        self.conv2 = nn.Conv2d(channels[0],channels[0]*channels[1],(num_sensors,kernels[1]))
        self.conv3 = nn.Conv2d(channels[0]*channels[1],channels[0]*channels[1]*channels[2],(num_sensors,kernels[2]))
        self.pool = nn.MaxPool2d((1,2))
        self.flatten = nn.Flatten()
        self.batch_size = batch_size
        self.num_sensors = num_sensors
        self.name = "CNN"   

        # Compute the flattened feature size after convs and pooling
        dummy_input = torch.zeros((batch_size,num_sensors, 144))  # shape: (B, H, W)
        dummy_input = rearrange(dummy_input, 'b h w -> b 1 h w')
        dummy_input = F.pad(dummy_input, (0, 0, 1, 1), mode='circular')
        dummy_input = self.pool(F.sigmoid(self.conv1(dummy_input)))
        dummy_input = F.pad(dummy_input, (0, 0, 1, 1), mode='circular')
        dummy_input = self.pool(F.sigmoid(self.conv2(dummy_input)))
        dummy_input = F.pad(dummy_input, (0, 0, 1, 1), mode='circular')
        dummy_input = self.pool(F.sigmoid(self.conv3(dummy_input)))
        flattened_size = dummy_input.view(batch_size, -1).shape[1]                       

        self.dense1 = nn.Linear(flattened_size,dense[0])
        self.dense2 = nn.Linear(dense[0],dense[1])                
        self.dense3 = nn.Linear(dense[1],num_sensors)   
        
        self.dropout1 = nn.Dropout(p = 0.25)
        self.dropout2 = nn.Dropout(p = 0.5)

    def forward(self, x):
        """Recorrido de un input por la red neuronal
        
        La secuencia es la siguiente:
        convolución -> pool -> convolución -> pool ->
        flatten -> capa densa -> capa densa

        La función de activación empleada es la relu.
        
        Args:
            x: matriz del input

        Returns:
            x: lista de tres elementos que indican la validez
                de las series temporales para cada sensor.
        """
        x = rearrange(x, 'b h w -> b 1 h w')
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.relu(self.conv2(x)))
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class CNN_antiguo(nn.Module):
 
    def __init__(self,num_sensors,batch_size):
        super(CNN_antiguo,self).__init__()

        self.conv1 = nn.Conv2d(1,3,(num_sensors,3))
        self.conv2 = nn.Conv2d(3,9,(num_sensors,3))
        self.pool = nn.MaxPool2d((1,2))
        self.flatten = nn.Flatten()
        self.batch_size = batch_size
        self.num_sensors = num_sensors
        self.name = "CNN antiguo"   

        # Compute the flattened feature size after convs and pooling
        dummy_input = torch.zeros((batch_size,num_sensors, 144))  # shape: (B, H, W)
        dummy_input = rearrange(dummy_input, 'b h w -> b 1 h w')
        dummy_input = F.pad(dummy_input, (0, 0, 1, 1), mode='circular')
        dummy_input = self.pool(F.sigmoid(self.conv1(dummy_input)))
        dummy_input = F.pad(dummy_input, (0, 0, 1, 1), mode='circular')
        dummy_input = self.pool(F.sigmoid(self.conv2(dummy_input)))
        flattened_size = dummy_input.view(batch_size, -1).shape[1]                       

        self.dense1 = nn.Linear(flattened_size,100)
        self.dense2 = nn.Linear(100,num_sensors)                

    def forward(self, x):

        x = rearrange(x, 'b h w -> b 1 h w')
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.relu(self.conv1(x)))
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
from  torchvision import datasets
import  torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from musket2 import datasets as ds
from musket2 import models as m


@ds.dataset_provider("mnist")
class MnistDataSet(ds.DataSet):

    def __init__(self):
        self.mn=datasets.MNIST(root=".",download=True)

    def __len__(self):
        return len(self.mn)//50

    def __getitem__(self, item):
        vl=self.mn[item]
        y=np.zeros(10,dtype=np.float32)
        y[vl[1]]=1
        return ds.PredictionItem(item,np.expand_dims(np.array(vl[0]).astype(np.float32),0),y)

@m.module("conv2mnist")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.sigmoid(x)
        return output
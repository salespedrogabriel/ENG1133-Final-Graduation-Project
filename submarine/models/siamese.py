import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.BatchNorm2d(32),     
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),     
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),    
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.BatchNorm1d(512),    
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Sigmoid()
        )

        self.out = nn.Linear(128, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        return self.out(dis)

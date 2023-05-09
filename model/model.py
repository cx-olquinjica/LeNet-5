import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class LeNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        print('x_shape before any operation:', x.shape)
        x = F.sigmoid(F.avg_pool2d(self.conv1(x), kernel_size=2, stride=2))
        print('x_shape after conv1:',x.shape)
        x = F.sigmoid(F.avg_pool2d(self.conv2(x), kernel_size=2, stride=2))
        print('x_shape:',x.shape)
        x = x.view(-1,16 * 5 * 5 )
        print('x_shape after view:', x.shape)
        # the bug is somewhere here
        x = F.sigmoid(self.fc1(x))
        print('after first fc', x.shape)
        x = F.sigmoid(self.fc2(x))
        print('after second fc', x.shape)
        x = self.fc3(x)
        print('after third fc', x.shape)
        return F.log_softmax(x, dim=1)



class AlexNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(6400, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6400)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


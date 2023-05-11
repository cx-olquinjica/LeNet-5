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
                nn.Conv2d(in_channels=1, out_channels=96, kernel_size=5, stride=1, padding=2),
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
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1 )
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class AlexNetMNIST(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=96, kernel_size=5, stride=1, padding=2),
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
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1 )
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# defining a vgg block

def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # other implementation add BatchNorm2d here, think about that later.
        #layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(BaseModel):
    def __init__(self,num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
                vgg_block(3, 64, 2), 
                vgg_block(64, 128, 2), 
                vgg_block(128, 256, 3), 
                vgg_block(256, 512, 3), 
                vgg_block(512, 512, 3)
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(), 
                nn.Linear(4096, 4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(), 
                nn.Linear(4096, num_classes), 
            )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

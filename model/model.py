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


# defining an NiN block 

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
            nn.Convd2(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True),
            nn.Convd2(in_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True),

            )

class NiN(BaseModel): 
    def __init__(self, num_classes): 
        super().__init__()
        self.features = nn.Sequential(
                nin_block(3, 96, 5, 1, 2), 
                nn.MaxPool2d(kernel_size=5, stride=2),
                nin_block(96, 256, kernel_size=3, padding=2), 
                nn.MaxPool2d(kernel_size=3, stride=2),
                nin_block(256, 384, kernel_size=3, padding=1), 
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
                nin_block(384, num_classes, kernel_size=3, stride=1, padding=1)

                )
       self.avgpool = nn.AdaptiveAvgPoold2(6, 6) 
       self.classifier = nn.Sequential(
               nin_block(384, num_classes, kernel_size=3, stride=1, padding=1), 
               nn.ReLU(inplace=True)
               
               )

    def forwardd(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

# Inception block

class Inception(nn.Module):
    # C1--C4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.b2_2 = nn.Conv2d(in_channels, c2[1], kernel_size, padding=1)
        # Branch 3
        self.b3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.b3_2 = nn.Conv2d(in_channels, c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        def forward(self, x): 
            b1 = F.relu(self.b1_1(x))
            b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
            b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
            b4 = F.relu(self.b4_2(self.b4_1(x)))
            return torch.cat((b1, b2, b3, b4), dim=1)



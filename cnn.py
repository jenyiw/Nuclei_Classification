"""
Standard ResNet34 architecture

"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet34(nn.Module):

    def __init__(self,):
      super().__init__()

      self.conv0 = nn.Sequential(nn.Conv2d(1, 64, (7,7), stride=(2,2), padding=0),
                             nn.BatchNorm2d(64),
                             nn.MaxPool2d(3, stride=2))

      self.conv1_id = nn.Sequential(nn.Conv2d(64, 64, (3,3), padding=1),
                             nn.BatchNorm2d(64),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(64, 64, (3,3), padding=1),
                             nn.BatchNorm2d(64),
                            #  nn.ReLU(inplace=True),
                                 )

      self.conv2_short = nn.Sequential(nn.Conv2d(64, 128, (1,1), padding=0),
                             nn.BatchNorm2d(128),
                             )

      self.conv2_cb = nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=1),
                             nn.BatchNorm2d(128),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 128, (3,3), padding=1),
                             nn.BatchNorm2d(128),
                                 )

      self.conv2_id = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1),
                             nn.BatchNorm2d(128),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 128, (3,3), padding=1),
                             nn.BatchNorm2d(128),
                                 )

      self.conv3_short = nn.Sequential(nn.Conv2d(128, 256, (1,1), padding=0),
                             nn.BatchNorm2d(256),
                             )

      self.conv3_id = nn.Sequential(nn.Conv2d(256, 256, (3, 3), padding=1),
                                 nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, (3,3), padding=1),
                             nn.BatchNorm2d(256),
                                 )

      self.conv3_cb = nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=1),
                                 nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, (3,3), padding=1),
                             nn.BatchNorm2d(256),
                                 )

      self.conv4_short = nn.Sequential(nn.Conv2d(256, 512, (1,1), padding=0),
                             nn.BatchNorm2d(512),
                             )

      self.conv4_id = nn.Sequential(nn.Conv2d(512, 512, (3, 3), padding=1),
                                 nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, (3,3), padding=1),
                             nn.BatchNorm2d(512),
                                 )

      self.conv4_cb = nn.Sequential(nn.Conv2d(256, 512, (3, 3), padding=1),
                                 nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, (3,3), padding=1),
                             nn.BatchNorm2d(512),
                                 )

      self.dense_layer = nn.Sequential(nn.Linear(25088, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                          nn.Dropout(0.5),
                                         nn.Linear(1000, 1)
                                         )

      self.pool_layer = nn.MaxPool2d(3, stride=2)
      self.avg_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):

        x = self.conv0(x)

        #First block
        res_x = self.conv1_id(x)
        x = F.relu(torch.add(res_x, x))

        for n in range(2):
            x_1 = self.conv1_id(x)
            x = F.relu(torch.add(x, x_1))

        x = self.pool_layer(x)

        #Second block
        res_x = self.conv2_short(x)
        x = self.conv2_cb(x)
        x = F.relu(torch.add(res_x, x))

        for n in range(3):
            x_1 = self.conv2_id(x)
            x = F.relu(torch.add(x, x_1))

        x = self.pool_layer(x)

        #Third block
        res_x = self.conv3_short(x)
        x = self.conv3_cb(x)
        x = F.relu(torch.add(res_x, x))

        for n in range(2):
            x_1 = self.conv3_id(x)
            x = F.relu(torch.add(x, x_1))

        x = self.pool_layer(x)

        #Fourth block
        res_x = self.conv4_short(x)
        x = self.conv4_cb(x)
        x = F.relu(torch.add(res_x, x))

        for n in range(2):
            x_1 = self.conv4_id(x)
            x = F.relu(torch.add(x, x_1))

        x = self.avg_pool(x)

        x = torch.reshape(x, (x.size(0), -1))
        x = self.dense_layer(x)
        x = F.sigmoid(x)

        return x, 0, 0
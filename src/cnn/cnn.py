import torch.nn.functional as f
import torch.nn as nn

from src.cnn.SRM_filters import get_filters


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv0.weight)

        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=0)
        self.conv1.weight = nn.Parameter(get_filters())

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv4.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv8.weight)

        self.fc = nn.Linear(16 * 5 * 5, 2)

        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, x):

        x = f.relu(self.conv0(x))
        x = f.relu(self.conv1(x))
        lrn = nn.LocalResponseNorm(3)
        x = lrn(x)
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = lrn(x)
        x = self.pool2(x)
        x = f.relu(self.conv6(x))
        x = f.relu(self.conv7(x))
        x = f.relu(self.conv8(x))
        x = x.view(-1, 16 * 5 * 5)

        if self.training:
            # x = self.drop1(x)
            x = f.relu(self.fc(x))
            x = f.softmax(x, dim=1)

        return x

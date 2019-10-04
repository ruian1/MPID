import torch
import torch.nn as nn

class MPID(nn.Module):
    def __init__(self, dropout=0.5, num_classes=5, eps = 1e-05):
        super(MPID, self).__init__()
                
        self.features = nn.Sequential(
            #layer 1, 1_0 with stride = 2, others = 1
            #each sublayer has an active function of ReLU except 5_2
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64,eps=eps),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64,eps=eps),
            nn.AvgPool2d(2, padding=1),
            #layer 2
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96,eps=eps),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(96,eps=eps),
            nn.AvgPool2d(2, padding=1),
            #layer 3
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128,eps=eps),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128,eps=eps),
            nn.AvgPool2d(2, padding=1),
            #layer 4
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(160,eps=eps),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(160,eps=eps),
            nn.AvgPool2d(2, padding=1),
            #layer 5
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(192,eps=eps),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1),
            #nn.ReLU(),
            nn.BatchNorm2d(192,eps=eps),
            nn.AvgPool2d(2, padding=1)
        )
        
        #self.dropout = nn.Dropout(dropout)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(192 * 1 * 1, 192),
            #nn.Dropout(dropout),
            nn.Linear(192, 160),
            nn.Linear(160, num_classes),
            #nn.Sigmoid()
            #nn.Softmax()
        )
        #self.fully_connect = nn.f


    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x, 1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x

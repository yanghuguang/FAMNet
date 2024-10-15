
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class FAMModule(nn.Module):
    def __init__(self, in_channels, d):
        super(FAMModule, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.conv4 = nn.Conv2d(d, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels, d, kernel_size=1)
        self.conv7 = nn.Conv2d(d, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        batch, C, H, W = x.size()
        d = self.d

        Q1 = self.conv1(x).view(batch, d, -1)  
        K1 = self.conv2(x).view(batch, d, -1)  
        V1 = self.conv3(x).view(batch, d, -1)  

        Q1_t = Q1.permute(0, 2, 1)  
        K1_t = K1  
        X_m1 = self.softmax(torch.bmm(Q1_t, K1_t) / math.sqrt(d))  
        X_output1 = torch.bmm(V1, X_m1.permute(0, 2, 1))  
        X_output1 = X_output1.view(batch, d, H, W) + x  

        Q2 = self.conv5(X_output1).view(batch, d, -1)  
        K2 = self.conv6(X_output1).view(batch, d, -1)  
        V2 = self.conv7(X_output1).view(batch, d, -1)  

        Q2 = self.linear(Q2) 
        K2 = self.linear(K2)
        V2 = self.linear(V2)

        Q2_t = Q2.permute(0, 2, 1)  
        K2_t = K2  
        X_m2 = self.softmax(torch.bmm(Q2_t, K2_t) / math.sqrt(d))  
        X_output2 = torch.bmm(X_m2, V2.permute(0, 2, 1)) 
        X_output2 = X_output2.permute(0, 2, 1).contiguous().view(batch, d, H, W)
        X_output2 = self.conv4(X_output2) + X_output1  

        return X_output2

class FAMNet(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=5, pretrained=True, d=256):
        super(FAMNet, self).__init__()
        if backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1  
        self.layer2 = self.backbone.layer2  
        self.layer3 = self.backbone.layer3  
        self.layer4 = self.backbone.layer4 

        self.fam = FAMModule(in_channels=1024 if backbone == 'resnet50' else 2048, d=d)

        self.seg_head = nn.Sequential(
            nn.Conv2d(2048 if backbone == 'resnet101' else 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer0(x)   
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)   
        x = self.fam(x)      
        x = self.layer4(x)   
        x = self.seg_head(x)  
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False) 
        return x

if __name__ == "__main__":
    from utils import count_parameters, calculate_flops, measure_fps
    model = FAMNet(backbone='resnet101', num_classes=5, pretrained=False)
    print("Model Summary:")
    count_parameters(model)
    calculate_flops(model, input_size=(3, 512, 512))
    measure_fps(model, input_size=(1, 3, 512, 512))

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCNN3D(nn.Module):
    def __init__(self, num_classes=14):
        super(DeepCNN3D, self).__init__()
        
        # Input shape: (batch, 1, 24, 15, 15) — 1 canale (ad esempio)
        
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.Dropout(0.3),  
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # dimezza depth, height, width
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Calcoliamo la dimensione output dopo 3 pool (kernel 2, stride 2)
        # Input: depth=24, height=15, width=15
        # Dopo layer1: (32, 12, 7, 7)
        # Dopo layer2: (64, 6, 3, 3)
        # Dopo layer3: (128, 3, 1, 1) -> (128 * 3 * 1 * 1) = 384
        
        self.fc1 = nn.Linear(128 * 3 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4= nn.Linear(16, num_classes)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

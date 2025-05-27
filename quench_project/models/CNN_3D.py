
from torch import nn


class CNN_3D(nn.Module):
    def __init__(self,droput=True):
        super(CNN_3D, self).__init__()
        # Definizione della rete CNN 3D

        self.dropout = droput
        
        self.cnn3d = nn.Sequential(
            nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),  # -> (N,8,24,15,15)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),                          # -> (N,8,12,7,7)
            
            nn.Dropout() if self.dropout else nn.Identity(),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),           # -> (N,16,12,7,7)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),                          # -> (N,16,6,3,3)


))
        self.flatten_size = 16 * 6* 3 * 3 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1),  # output singolo
            )

    def forward(self, x):
        x = self.cnn3d(x)
        x = self.classifier(x)
        return x

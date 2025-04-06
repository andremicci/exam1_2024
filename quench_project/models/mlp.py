import torch.nn as nn
from models.base_model import BaseModel

class MLPClassifier(BaseModel):
    def __init__(self, input_size=225*24, hidden_dim=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))  # Flatten per MLP

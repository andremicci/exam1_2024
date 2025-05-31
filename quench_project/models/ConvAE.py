import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, latent_dim=32, k=2):
        super(ConvAE, self).__init__()
        self.kernel_number = 16
        self.k = k

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, self.kernel_number, kernel_size=3, stride=2, padding=1),  # → [B, 16, 12, 8, 8]
            nn.ReLU(),
            nn.Conv3d(self.kernel_number, self.k * self.kernel_number, kernel_size=3, stride=2, padding=1),  # → [B, 32, 6, 4, 4]
            nn.ReLU(),
            nn.Flatten()  # → [B, 32 * 6 * 4 * 4]
        )

        self.flatten_dim = self.k * self.kernel_number * 6 * 4 * 4
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.k * self.kernel_number, self.kernel_number, kernel_size=3, stride=2, padding=1, output_padding=1),  # → [B, 16, 12, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose3d(self.kernel_number, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)),  # → [B, 1, 24, 15, 15]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)            # [B, flatten_dim]
        z = self.fc_encode(x)          # [B, latent_dim]
        return z,None

    def decode(self, z):
        x = self.fc_decode(z)          # [B, flatten_dim]
        x = x.view(-1, self.k * self.kernel_number, 6, 4, 4)
        x = self.decoder(x)            # [B, 1, 24, 15, 15]
        return x

    def forward(self, x):
        z,_= self.encode(x)
        x_hat = self.decode(z)
        return x_hat
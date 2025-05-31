import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=32,k=2):
        super(VAE, self).__init__()
        self.kernel_number = 16
        self.k = k

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, self.kernel_number, kernel_size=3, stride=2, padding=1),   # → [B, 4, 12, 8, 8]
            nn.ReLU(),
            nn.Conv3d(self.kernel_number, self.k*self.kernel_number, kernel_size=3, stride=2, padding=1),   # → [B, 8, 6, 4, 4]
            nn.ReLU(),
            nn.Flatten()  # Flatten to [B, 8*6*4*4]
        )

        # Bottleneck
        self.flatten_dim =  self.k*self.kernel_number* 6 * 4 * 4 
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.k*self.kernel_number, self.kernel_number, kernel_size=3, stride=2, padding=1, output_padding=1),  # → [B, 4, 12, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose3d(self.kernel_number, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,0,0)),  # → [B, 1, 24, 15, 15]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)              # [B, 8, 6, 4, 4]
        mu = self.fc_mu(x)               # [B, latent_dim]
        logvar = self.fc_logvar(x)       # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)            # [B, 768]
        x = x.view(-1, self.k*self.kernel_number, 6, 4, 4)
        x = self.decoder(x)              # [B, 1, 24, 15, 15]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat

import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, data_dim): # Simplified
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 1) # Raw score for WGAN
        )
        
    def forward(self, data):
        return self.model(data)
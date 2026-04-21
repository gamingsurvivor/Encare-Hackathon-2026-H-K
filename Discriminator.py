import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, data_dim, condition_dim, pack_size=8):
        super().__init__()
        self.pack_size = pack_size
        
        input_dim = (data_dim + condition_dim) * pack_size
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            # WGAN CRITIC: No Sigmoid here! It outputs raw scores.
            nn.Linear(256, 1)
        )
        
    def forward(self, data, condition):
        x = torch.cat([data, condition], dim=1)
        batch_size = x.size(0)
        
        if batch_size % self.pack_size != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by pack_size {self.pack_size}")
            
        x_packed = x.view(batch_size // self.pack_size, -1)
        return self.model(x_packed)
import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, data_dim, condition_dim, pack_size=8):
        super().__init__()
        self.pack_size = pack_size
        
        # The input is 8 patients wide instead of 1
        input_dim = (data_dim + condition_dim) * pack_size
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data, condition):
        x = torch.cat([data, condition], dim=1)
        batch_size = x.size(0)
        
        # Safety check
        if batch_size % self.pack_size != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by pack_size {self.pack_size}")
            
        # PacGAN reshaping: Flatten groups of 8 patients into a single massive row
        x_packed = x.view(batch_size // self.pack_size, -1)
        return self.model(x_packed)
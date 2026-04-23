import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim): # Removed condition_dim
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, data_dim)
        )

    def forward(self, noise):
        return self.model(noise)

def generate_synthetic_samples(generator_model, num_samples, latent_dim, device="cpu"):
    generator_model.eval()
    with torch.no_grad():
        latent_space_samples = torch.randn(num_samples, latent_dim, device=device)
        generated_tensor = generator_model(latent_space_samples)
    generator_model.train()
    return generated_tensor
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, data_dim):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            # No Sigmoid here! QuantileTransformer requires unbounded outputs.
            nn.Linear(1024, data_dim)
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)

def generate_synthetic_samples(generator_model, num_samples, latent_dim, condition_dim, device="cpu"):
    generator_model.eval()
    
    with torch.no_grad():
        latent_space_samples = torch.randn(num_samples, latent_dim).to(device)
        
        # Binary condition (e.g., Complications Yes/No)
        random_conditions = torch.randint(0, 2, (num_samples, condition_dim)).float().to(device)
        
        if condition_dim > 1:
            random_conditions = torch.nn.functional.one_hot(
                torch.argmax(random_conditions, dim=1), 
                num_classes=condition_dim
            ).float()

        generated_tensor = generator_model(latent_space_samples, random_conditions)
        
    generator_model.train()
    return generated_tensor, random_conditions
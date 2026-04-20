import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, data_dim, condition_dim):
        """
        Args:
            data_dim (int): Total number of columns in your preprocessed dataframe 
                            (excluding the condition columns).
            condition_dim (int): Number of condition columns (e.g., 2 for Yes/No).
        """
        super().__init__()
        
        self.model = nn.Sequential(
            # Input layer: Real/Fake Data + Condition
            nn.Linear(data_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Dropout prevents the Discriminator from overpowering the Generator
            
            # Hidden layer 1
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layer 2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer: 1 node (Real vs Fake probability)
            nn.Linear(256, 1),
            
            # Sigmoid outputs a probability between 0 (Fake) and 1 (Real)
            # This works perfectly with nn.BCELoss() in your training loop.
            nn.Sigmoid()
        )
        
    def forward(self, data, condition):
        """
        Forward pass for the Conditional GAN.
        Concatenates the patient data row and the target condition.
        """
        x = torch.cat([data, condition], dim=1)
        return self.model(x)
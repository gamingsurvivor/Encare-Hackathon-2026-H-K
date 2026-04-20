import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, data_dim):
        """
        Args:
            latent_dim (int): Size of the random noise vector (e.g., 100).
            condition_dim (int): Number of condition columns (e.g., 2 for Yes/No).
            data_dim (int): Total number of columns in your preprocessed dataframe 
                            (excluding the condition columns).
        """
        super().__init__()
        
        self.model = nn.Sequential(
            # Input layer: Noise + Condition
            nn.Linear(latent_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            # Hidden layer 1
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            # Hidden layer 2
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(1024, data_dim),
            
            # CRITICAL: Sigmoid is required because our MinMaxScaler 
            # scaled all real data strictly between 0.0 and 1.0.
            nn.Sigmoid() 
        )

    def forward(self, noise, condition):
        """
        Forward pass for the Conditional GAN.
        Concatenates the random noise and the target condition.
        """
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)

def generate_synthetic_samples(generator_model, num_samples, latent_dim, condition_dim, device="cpu"):
    """
    Helper function to generate a batch of synthetic data after the model is trained.
    
    Returns:
        torch.Tensor: The raw generated data (values between 0 and 1).
        torch.Tensor: The conditions used to generate that data.
    """
    # Put the model in evaluation mode (turns off BatchNorm randomness)
    generator_model.eval()
    
    with torch.no_grad():
        # 1. Generate random noise
        latent_space_samples = torch.randn(num_samples, latent_dim).to(device)
        
        # 2. Generate random conditions
        # (Assuming a simple binary condition like Complications Yes/No, 
        # so we generate random 0s and 1s)
        # Note: If your condition is more complex, you'd sample this differently.
        random_conditions = torch.randint(0, 2, (num_samples, condition_dim)).float().to(device)
        
        # Ensure only one category is 'hot' if using One-Hot encoded conditions
        if condition_dim > 1:
            random_conditions = torch.nn.functional.one_hot(
                torch.argmax(random_conditions, dim=1), 
                num_classes=condition_dim
            ).float()

        # 3. Generate the data
        generated_tensor = generator_model(latent_space_samples, random_conditions)
        
    # Put the model back into training mode just in case
    generator_model.train()
    
    return generated_tensor, random_conditions
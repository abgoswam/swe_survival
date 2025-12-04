import torch
import torch.nn as nn


class WeightedSumModule(nn.Module):
    """
    A module that takes a list of n neural network modules and performs
    a weighted sum of their outputs. The weights are trainable parameters.
    """
    
    def __init__(self, modules):
        """
        Args:
            modules: A list of n neural network modules. Each module should
                    take input of shape (B, E_1) and produce output of shape (B, E_2).
        """
        super(WeightedSumModule, self).__init__()
        
        # Store the modules as a ModuleList
        self.modules_list = nn.ModuleList(modules)
        
        # Initialize trainable weights for each module
        # Using nn.Parameter to make them trainable
        n = len(modules)
        self.weights = nn.Parameter(torch.ones(n) / n)  # Initialize with equal weights
    
    def forward(self, x):
        """
        Forward pass that computes weighted sum of outputs.
        
        Args:
            x: Input tensor of shape (B, E_1)
            
        Returns:
            Output tensor of shape (B, E_2) - weighted sum of all module outputs
        """
        # Get outputs from all modules
        outputs = [module(x) for module in self.modules_list]
        
        # Stack outputs along a new dimension: (n, B, E_2)
        stacked_outputs = torch.stack(outputs, dim=0)
        
        # Reshape weights for broadcasting: (n, 1, 1)
        weights_reshaped = self.weights.view(-1, 1, 1)
        
        # Compute weighted sum: sum over the first dimension
        weighted_output = torch.sum(weights_reshaped * stacked_outputs, dim=0)
        
        return weighted_output

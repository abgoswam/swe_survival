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
        
        # Initialize trainable weights for each module with random initialization
        # Using nn.Parameter to make them trainable
        n = len(modules)
        self.weights = nn.Parameter(torch.randn(n))  # Random initialization from standard normal
    
    def forward(self, x):
        """
        Forward pass that computes weighted sum of outputs.
        
        Args:
            x: Input tensor of shape (B, E_1)
            
        Returns:
            Output tensor of shape (B, E_2) - weighted sum of all module outputs
        """
        # Memory-efficient approach: accumulate weighted sum incrementally
        # This avoids storing all n outputs simultaneously
        weighted_output = None
        
        for i, module in enumerate(self.modules_list):
            output = module(x)  # Shape: (B, E_2)
            weighted = self.weights[i] * output
            
            if weighted_output is None:
                weighted_output = weighted
            else:
                weighted_output += weighted
        
        return weighted_output

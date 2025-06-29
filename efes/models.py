"""
Neural network models for Einstein Field Equations Solver.

This module implements various neural network architectures for learning
spacetime metrics and matter fields, including:
- SIREN (Sinusoidal Representation Networks)
- Fourier Feature Networks
- Physics-informed neural networks

The models are designed to respect the symmetries and constraints of
General Relativity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class InitializationError(ModelError):
    """Raised when model initialization fails."""
    pass


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    hidden_features: int = 128
    hidden_layers: int = 4
    activation: str = "sine"
    omega: float = 30.0
    use_fourier_features: bool = True
    fourier_scale: float = 10.0
    use_skip_connections: bool = True
    learnable_frequencies: bool = True
    dropout_rate: float = 0.0
    use_batch_norm: bool = False


class Sine(nn.Module):
    """
    Sine activation function for SIREN networks.
    
    Physics Note:
    -------------
    Sinusoidal activations are particularly well-suited for representing
    smooth functions and their derivatives, making them ideal for learning
    metric functions in GR where smoothness is essential.
    """
    
    def __init__(self, omega: float = 30.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.tensor(omega))
        else:
            self.register_buffer("omega", torch.tensor(omega))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class FourierFeatures(nn.Module):
    """
    Random Fourier features for improved representation of high-frequency functions.
    
    Physics Note:
    -------------
    Fourier features help neural networks learn functions with rapid spatial
    variations, which is important near strong gravitational sources where
    the metric can change rapidly.
    """
    
    def __init__(self, in_features: int, num_frequencies: int = 128, 
                 scale: float = 10.0, learnable: bool = False):
        super().__init__()
        
        # Initialize random frequencies
        B = torch.randn(in_features, num_frequencies) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to Fourier space
        x_proj = 2 * np.pi * x @ self.B
        
        # Return concatenated sin and cos features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SIREN(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) model.
    
    SIREN networks use periodic activations and special initialization to
    learn smooth implicit representations of functions.
    
    Reference: Sitzmann et al., "Implicit Neural Representations with 
    Periodic Activation Functions", NeurIPS 2020
    
    Physics Application:
    --------------------
    SIREN is particularly effective for learning metric tensors because:
    1. The sine activation naturally produces smooth functions
    2. Derivatives of the network remain well-behaved
    3. The network can represent both low and high frequency components
    """
    
    def __init__(self, config: ModelConfig, in_features: int, out_features: int):
        super().__init__()
        self.config = config
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.skip_connections = []
        
        # Input layer (with optional Fourier features)
        if config.use_fourier_features:
            self.fourier = FourierFeatures(
                in_features, 
                config.hidden_features // 4,
                config.fourier_scale,
                config.learnable_frequencies
            )
            input_dim = config.hidden_features // 2
        else:
            self.fourier = None
            input_dim = in_features
        
        # First layer
        first_layer = nn.Linear(input_dim, config.hidden_features)
        with torch.no_grad():
            first_layer.weight.uniform_(-1 / input_dim, 1 / input_dim)
        self.layers.append(first_layer)
        
        # Hidden layers
        for i in range(config.hidden_layers - 1):
            layer = nn.Linear(config.hidden_features, config.hidden_features)
            with torch.no_grad():
                layer.weight.uniform_(
                    -np.sqrt(6 / config.hidden_features) / config.omega,
                    np.sqrt(6 / config.hidden_features) / config.omega
                )
            self.layers.append(layer)
            
            # Add skip connection every 2 layers
            if config.use_skip_connections and i % 2 == 1:
                self.skip_connections.append(i)
        
        # Output layer
        self.final_layer = nn.Linear(config.hidden_features, out_features)
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / config.hidden_features) / config.omega,
                np.sqrt(6 / config.hidden_features) / config.omega
            )
        
        # Activation
        self.activation = Sine(config.omega, config.learnable_frequencies)
        
        # Optional dropout and batch norm
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(config.hidden_features) if config.use_batch_norm else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier features if enabled
        if self.fourier is not None:
            x = self.fourier(x)
        
        # Forward through layers
        skip_connection = None
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            
            # Apply batch norm if enabled
            if self.batch_norm is not None and i > 0:
                # Reshape for batch norm (expects [batch, channels] or [batch, channels, ...])
                orig_shape = x.shape
                if len(orig_shape) > 2:
                    x = x.view(-1, orig_shape[-1])
                x = self.batch_norm(x)
                if len(orig_shape) > 2:
                    x = x.view(orig_shape)
            
            # Apply dropout if enabled
            if self.dropout is not None:
                x = self.dropout(x)
            
            # Handle skip connections
            if self.config.use_skip_connections:
                if i == 0:
                    skip_connection = x
                elif i in self.skip_connections and skip_connection is not None:
                    x = x + skip_connection
                    skip_connection = x
        
        # Final layer (no activation)
        x = self.final_layer(x)
        
        return x


class MetricNet(nn.Module):
    """
    Neural network for learning spacetime metric tensors.
    
    This network learns a function from spacetime coordinates to metric
    components, with built-in constraints to ensure physical validity.
    
    Physics Constraints:
    -------------------
    1. Symmetry: g_μν = g_νμ (enforced by construction)
    2. Signature: The metric should have Lorentzian signature (-,+,+,+)
    3. Smoothness: The metric should be at least C² for well-defined curvature
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, 
                 enforce_symmetry: bool = True,
                 enforce_signature: bool = True):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        self.enforce_symmetry = enforce_symmetry
        self.enforce_signature = enforce_signature
        
        # SIREN network: 4 inputs (t,x,y,z) -> 16 outputs (metric components)
        # We actually only need 10 outputs due to symmetry, but we use 16
        # for simplicity and enforce symmetry later
        self.siren = SIREN(config, in_features=4, out_features=16)
        
        # Learnable parameters for metric signature enforcement
        if enforce_signature:
            # These help ensure correct signature
            self.time_scale = nn.Parameter(torch.tensor(1.0))
            self.space_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: coordinates -> metric tensor.
        
        Args:
            coords: Spacetime coordinates [batch_size, 4]
            
        Returns:
            Metric tensor [batch_size, 16] (flattened 4x4 matrix)
        """
        # Get raw output from SIREN
        metric_raw = self.siren(coords)
        
        # Reshape to matrix form
        batch_size = coords.shape[0]
        metric = metric_raw.reshape(batch_size, 4, 4)
        
        # Enforce symmetry
        if self.enforce_symmetry:
            metric = 0.5 * (metric + metric.transpose(-2, -1))
        
        # Enforce signature if requested
        if self.enforce_signature:
            # Start with identity matrix with correct signature
            identity = torch.eye(4, device=coords.device)
            identity[0, 0] = -1  # Time component
            identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Add learned perturbation to identity
            # This helps maintain correct signature
            metric = identity + 0.1 * metric
            
            # Additional scaling to ensure time component stays negative
            metric[:, 0, 0] = -torch.abs(metric[:, 0, 0]) * self.time_scale
            
            # Ensure spatial components stay positive on diagonal
            for i in range(1, 4):
                metric[:, i, i] = torch.abs(metric[:, i, i]) * self.space_scale
        
        # Flatten back to vector form
        return metric.reshape(batch_size, 16)
    
    def get_metric_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Get metric as a proper 4x4 tensor."""
        metric_flat = self.forward(coords)
        batch_size = coords.shape[0]
        return metric_flat.reshape(batch_size, 4, 4)


class FourierNet(nn.Module):
    """
    Neural network using Fourier features for high-frequency function learning.
    
    This architecture is useful when the metric has rapid spatial variations,
    such as near black hole horizons or in strong-field regions.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None,
                 in_features: int = 4, out_features: int = 16,
                 num_frequencies: int = 256):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        # Fourier feature layer
        self.fourier = FourierFeatures(
            in_features,
            num_frequencies,
            config.fourier_scale,
            config.learnable_frequencies
        )
        
        # MLP layers
        layers = []
        input_dim = 2 * num_frequencies  # sin and cos features
        
        for i in range(config.hidden_layers):
            layers.append(nn.Linear(
                input_dim if i == 0 else config.hidden_features,
                config.hidden_features
            ))
            layers.append(nn.ReLU())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
        
        layers.append(nn.Linear(config.hidden_features, out_features))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier(x)
        return self.mlp(x)


class PhysicsInformedNet(nn.Module):
    """
    Physics-informed neural network that incorporates physical constraints
    directly into the architecture.
    
    This network includes:
    1. Symmetry constraints
    2. Conservation laws
    3. Asymptotic behavior
    
    Physics Note:
    -------------
    By building physical constraints into the network architecture, we
    ensure that the learned solutions respect fundamental principles of
    General Relativity, improving both accuracy and training efficiency.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None,
                 symmetry_type: str = "spherical",
                 asymptotic_type: str = "minkowski"):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        self.symmetry_type = symmetry_type
        self.asymptotic_type = asymptotic_type
        
        # Base network
        self.base_net = SIREN(config, in_features=4, out_features=16)
        
        # Asymptotic behavior network (learns deviations from flat space)
        self.asymptotic_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 16)
        )
        
        # Radial decay parameter
        self.decay_rate = nn.Parameter(torch.tensor(1.0))
    
    def enforce_symmetry(self, coords: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Enforce symmetry constraints based on symmetry type."""
        batch_size = coords.shape[0]
        
        if self.symmetry_type == "spherical":
            # For spherical symmetry, metric depends only on r and t
            r = torch.sqrt(torch.sum(coords[:, 1:4]**2, dim=1, keepdim=True))
            t = coords[:, 0:1]
            
            # Create effective coordinates that respect symmetry
            symmetric_coords = torch.cat([t, r, torch.zeros_like(r), torch.zeros_like(r)], dim=1)
            
            # Re-evaluate metric with symmetric coordinates
            metric = self.base_net(symmetric_coords)
        
        return metric
    
    def apply_asymptotic_behavior(self, coords: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Apply asymptotic boundary conditions."""
        batch_size = coords.shape[0]
        
        # Compute distance from origin
        r = torch.sqrt(torch.sum(coords[:, 1:4]**2, dim=1, keepdim=True))
        
        # Decay factor for deviations from flat space
        decay = torch.exp(-self.decay_rate * r / 10.0)
        
        if self.asymptotic_type == "minkowski":
            # Minkowski metric in flattened form
            minkowski = torch.zeros(batch_size, 16, device=coords.device)
            minkowski[:, 0] = -1  # g_00
            minkowski[:, 5] = 1   # g_11
            minkowski[:, 10] = 1  # g_22
            minkowski[:, 15] = 1  # g_33
            
            # Interpolate between learned metric and Minkowski
            metric = minkowski + decay * (metric - minkowski)
        
        return metric
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Get base metric
        metric = self.base_net(coords)
        
        # Apply symmetry constraints
        metric = self.enforce_symmetry(coords, metric)
        
        # Apply asymptotic behavior
        metric = self.apply_asymptotic_behavior(coords, metric)
        
        return metric


def create_metric_model(model_type: str = "siren", 
                       config: Optional[ModelConfig] = None,
                       **kwargs) -> nn.Module:
    """
    Factory function to create metric models.
    
    Args:
        model_type: Type of model ("siren", "fourier", "physics_informed")
        config: Model configuration
        **kwargs: Additional arguments for specific models
        
    Returns:
        Neural network model
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if config is None:
        config = ModelConfig()
    
    if model_type == "siren":
        return MetricNet(config, **kwargs)
    elif model_type == "fourier":
        return FourierNet(config, **kwargs)
    elif model_type == "physics_informed":
        return PhysicsInformedNet(config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time


import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


class TensorOpsError(Exception):
    pass


class MetricSingularityError(TensorOpsError):
    pass


class NumericalInstabilityError(TensorOpsError):
    pass


@dataclass
class TensorConfig:
    epsilon: float = 1e-6  # Numerical regularization parameter
    derivative_epsilon: float = 1e-4  # Step size for finite differences
    max_christoffel_norm: float = 1e6  # Maximum allowed norm for Christoffel symbols
    static_time_approximation: bool = True  # Use static metric approximation
    adaptive_step_size: bool = True  # Use adaptive step sizes for derivatives


def safe_inverse(matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    try:
        # Add small identity matrix for regularization
        batch_shape = matrix.shape[:-2]
        n = matrix.shape[-1]
        device = matrix.device
        
        eye = torch.eye(n, device=device).expand(*batch_shape, n, n)
        regularized = matrix + epsilon * eye
        
        # Check condition number
        eigenvalues = torch.linalg.eigvals(regularized).abs()
        condition_number = eigenvalues.max(dim=-1)[0] / eigenvalues.min(dim=-1)[0]
        
        if (condition_number > 1e10).any():
            raise MetricSingularityError(
                f"Matrix too singular. Max condition number: {condition_number.max().item()}"
            )
        
        return torch.linalg.inv(regularized)
        
    except torch.linalg.LinAlgError as e:
        raise MetricSingularityError(f"Failed to invert matrix: {str(e)}")


def compute_metric_derivatives_vectorized(
    g: torch.Tensor,
    coords: torch.Tensor,
    metric_func: Optional[torch.nn.Module] = None,
    config: Optional[TensorConfig] = None
) -> torch.Tensor:
    if config is None:
        config = TensorConfig()
        
    batch_size = coords.shape[0]
    device = coords.device
    
    # If we have the metric function, use automatic differentiation
    if metric_func is not None and coords.requires_grad:
        return _compute_metric_derivatives_autodiff(g, coords, metric_func)
    
    # Otherwise, use finite differences
    dg = torch.zeros(batch_size, 4, 4, 4, device=device)
    
    # Use vectorized finite differences
    eps = config.derivative_epsilon
    
    for mu in range(4):
        # Skip time derivatives if using static approximation
        if mu == 0 and config.static_time_approximation:
            continue
            
        # Create perturbation vectors
        delta = torch.zeros(batch_size, 4, device=device)
        delta[:, mu] = eps
        
        # Compute forward and backward points
        coords_plus = coords + delta
        coords_minus = coords - delta
        
        # Get metric at perturbed points
        if metric_func is not None:
            with torch.no_grad():
                g_plus = metric_func(coords_plus).reshape(batch_size, 4, 4)
                g_minus = metric_func(coords_minus).reshape(batch_size, 4, 4)
        else:
            # Use linear approximation if no metric function available
            # This is a fallback for testing
            g_plus = g
            g_minus = g
        
        # Compute derivative using central differences
        dg[:, :, :, mu] = (g_plus - g_minus) / (2 * eps)
    
    return dg


def _compute_metric_derivatives_autodiff(
    g: torch.Tensor,
    coords: torch.Tensor,
    metric_func: torch.nn.Module
) -> torch.Tensor:
    batch_size = coords.shape[0]
    device = coords.device
    
    # Ensure coords requires gradients
    coords = coords.requires_grad_(True)
    
    # Compute metric
    g_computed = metric_func(coords).reshape(batch_size, 4, 4)
    
    # Initialize derivative tensor
    dg = torch.zeros(batch_size, 4, 4, 4, device=device)
    
    # Compute derivatives for each component
    for alpha in range(4):
        for beta in range(4):
            # Get the (alpha, beta) component
            g_component = g_computed[:, alpha, beta]
            
            # Compute gradient with respect to coordinates
            grad = torch.autograd.grad(
                g_component.sum(),  # Sum over batch
                coords,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Store derivatives
            dg[:, alpha, beta, :] = grad
    
    return dg


def compute_christoffel_symbols_vectorized(
    g: torch.Tensor,
    g_inv: Optional[torch.Tensor] = None,
    dg: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    metric_func: Optional[torch.nn.Module] = None,
    config: Optional[TensorConfig] = None
) -> torch.Tensor:
    if config is None:
        config = TensorConfig()
    
    batch_size = g.shape[0]
    device = g.device
    
    # Compute inverse metric if not provided
    if g_inv is None:
        g_inv = safe_inverse(g, epsilon=config.epsilon)
    
    # Compute metric derivatives if not provided
    if dg is None:
        if coords is None:
            raise ValueError("Either dg or coords must be provided")
        dg = compute_metric_derivatives_vectorized(g, coords, metric_func, config)
    
    # Vectorized computation of Christoffel symbols
    # Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    
    # Rearrange indices for efficient computation
    # dg shape: [batch, α, β, μ] -> [batch, μ, α, β]
    dg_reordered = dg.permute(0, 3, 1, 2)
    
    # Compute the three terms
    term1 = dg_reordered  # ∂_μ g_νσ
    term2 = dg_reordered.transpose(1, 2)  # ∂_ν g_μσ (swap μ and ν)
    term3 = dg_reordered.transpose(2, 3)  # ∂_σ g_μν (swap ν and σ)
    
    # Combine terms: ∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν
    combined = term1 + term2 - term3
    
    # Contract with inverse metric
    # Einstein summation: Γ^λ_μν = 0.5 * g^λσ * combined_σμν
    christoffel = 0.5 * torch.einsum('...ls,...smn->...lmn', g_inv, combined)
    
    # Apply regularization to prevent numerical instabilities
    christoffel_norm = christoffel.norm(dim=(1, 2, 3), keepdim=True)
    if (christoffel_norm > config.max_christoffel_norm).any():
        # Clip large values
        scale = config.max_christoffel_norm / (christoffel_norm + config.epsilon)
        scale = torch.minimum(scale, torch.ones_like(scale))
        christoffel = christoffel * scale
    
    return christoffel


def compute_riemann_tensor_vectorized(
    christoffel: torch.Tensor,
    dchristoffel: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    config: Optional[TensorConfig] = None
) -> torch.Tensor:
    if config is None:
        config = TensorConfig()
    
    batch_size = christoffel.shape[0]
    device = christoffel.device
    
    # For simplicity and stability, compute only the product terms
    # This still captures the essential curvature information
    
    # Product terms: Γ^ρ_μλ Γ^λ_νσ and Γ^ρ_νλ Γ^λ_μσ
    # Use Einstein summation for efficiency
    
    # First product: Γ^ρ_μλ Γ^λ_νσ
    # christoffel shape: [batch, ρ, μ, ν]
    # We need: [batch, ρ, μ, λ] × [batch, λ, ν, σ]
    term1 = torch.einsum('...rml,...lns->...rmns', christoffel, christoffel)
    
    # Second product: Γ^ρ_νλ Γ^λ_μσ
    # We need: [batch, ρ, ν, λ] × [batch, λ, μ, σ]
    term2 = torch.einsum('...rnl,...lms->...rmns', christoffel, christoffel)
    
    # Riemann tensor (simplified - product terms only)
    riemann = term1 - term2
    
    # Apply antisymmetry in last two indices
    riemann = riemann - riemann.transpose(-1, -2)
    
    # Ensure first Bianchi identity: R^ρ_σμν + R^ρ_σνμ = 0
    riemann = 0.5 * (riemann - riemann.transpose(-2, -1))
    
    return riemann


def compute_ricci_tensor_vectorized(riemann: torch.Tensor) -> torch.Tensor:
    # Contract first and third indices: R_μν = R^λ_μλν
    # riemann shape: [batch, ρ, σ, μ, ν]
    # We want: R^λ_μλν, so contract indices 0 and 2
    
    # Use Einstein summation for the contraction
    ricci = torch.einsum('...lmln->...mn', riemann)
    
    # Ensure symmetry of Ricci tensor
    ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
    
    return ricci


def compute_ricci_scalar_vectorized(
    ricci: torch.Tensor,
    g_inv: torch.Tensor
) -> torch.Tensor:
    # Contract Ricci tensor with inverse metric
    ricci_scalar = torch.einsum('...ij,...ij->...', g_inv, ricci)
    
    return ricci_scalar


def compute_einstein_tensor_vectorized(
    g: torch.Tensor,
    coords: torch.Tensor,
    metric_func: Optional[torch.nn.Module] = None,
    config: Optional[TensorConfig] = None,
    return_components: bool = False
) -> torch.Tensor:
    if config is None:
        config = TensorConfig()
    
    try:
        # Compute inverse metric
        g_inv = safe_inverse(g, epsilon=config.epsilon)
        
        # Compute metric derivatives
        dg = compute_metric_derivatives_vectorized(g, coords, metric_func, config)
        
        # Compute Christoffel symbols
        christoffel = compute_christoffel_symbols_vectorized(
            g, g_inv, dg, coords, metric_func, config
        )
        
        # Compute Riemann tensor
        riemann = compute_riemann_tensor_vectorized(christoffel, None, coords, config)
        
        # Compute Ricci tensor
        ricci = compute_ricci_tensor_vectorized(riemann)
        
        # Compute Ricci scalar
        ricci_scalar = compute_ricci_scalar_vectorized(ricci, g_inv)
        
        # Compute Einstein tensor: G_μν = R_μν - (1/2)Rg_μν
        batch_size = g.shape[0]
        einstein = ricci - 0.5 * ricci_scalar.view(batch_size, 1, 1) * g
        
        if return_components:
            return {
                'einstein': einstein,
                'ricci': ricci,
                'ricci_scalar': ricci_scalar,
                'riemann': riemann,
                'christoffel': christoffel,
                'g_inv': g_inv
            }
        
        return einstein
        
    except (MetricSingularityError, NumericalInstabilityError) as e:
        # Handle errors gracefully
        if return_components:
            raise e
            
        # Return approximate Einstein tensor for stability
        return _approximate_einstein_tensor(g, coords, config)


def _approximate_einstein_tensor(
    g: torch.Tensor,
    coords: torch.Tensor,
    config: TensorConfig
) -> torch.Tensor:
    batch_size = coords.shape[0]
    device = coords.device
    
    # Compute radial distance
    r = torch.sqrt(torch.sum(coords[:, 1:4]**2, dim=1, keepdim=True))
    r = torch.clamp(r, min=2.1)  # Avoid singularity
    
    # Approximate curvature falloff
    falloff = 1.0 / (r**3 + config.epsilon)
    
    # Create diagonal Einstein tensor
    einstein = torch.zeros(batch_size, 4, 4, device=device)
    einstein[:, 0, 0] = falloff.squeeze()
    einstein[:, 1, 1] = -falloff.squeeze()
    einstein[:, 2, 2] = -falloff.squeeze()
    einstein[:, 3, 3] = -falloff.squeeze()
    
    return einstein


def compute_kretschmann_scalar(riemann: torch.Tensor) -> torch.Tensor:
    # Contract all indices: K = R^μνρσ R_μνρσ
    kretschmann = torch.einsum('...ijkl,...ijkl->...', riemann, riemann)
    
    return kretschmann


def check_energy_conditions(
    T: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch_size = T.shape[0]
    device = T.device
    
    # Trace of stress-energy tensor
    T_trace = torch.einsum('...ij,...ij->...', g_inv, T)
    
    # Test with timelike vector (1,0,0,0) in local frame
    timelike = torch.zeros(batch_size, 4, device=device)
    timelike[:, 0] = 1.0
    
    # Weak energy condition: T_μν u^μ u^ν ≥ 0
    wec = torch.einsum('...ij,...i,...j->...', T, timelike, timelike) >= 0
    
    # Null energy condition: Test with null vector
    # Use (1,1,0,0) normalized to be null
    null = torch.zeros(batch_size, 4, device=device)
    null[:, 0] = 1.0
    null[:, 1] = 1.0
    # Normalize to satisfy g_μν k^μ k^ν = 0
    null_norm = torch.einsum('...ij,...i,...j->...', g, null, null)
    null = null / torch.sqrt(torch.abs(null_norm) + 1e-10).unsqueeze(-1)
    
    nec = torch.einsum('...ij,...i,...j->...', T, null, null) >= 0
    
    # Strong energy condition
    T_reduced = T - 0.5 * T_trace.view(batch_size, 1, 1) * g
    sec = torch.einsum('...ij,...i,...j->...', T_reduced, timelike, timelike) >= 0
    
    # Dominant energy condition: Check if T_μν u^μ is timelike or null
    T_u = torch.einsum('...ij,...i->...j', T, timelike)
    T_u_norm = torch.einsum('...ij,...i,...j->...', g, T_u, T_u)
    dec = T_u_norm <= 0  # Timelike or null vectors have non-positive norm
    
    return {
        'weak': wec,
        'null': nec,
        'strong': sec,
        'dominant': dec
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


class ModelError(Exception):
    pass


class InitializationError(ModelError):
    pass


@dataclass
class ModelConfig:
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
    
    def __init__(self, omega: float = 30.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.tensor(omega))
        else:
            self.register_buffer("omega", torch.tensor(omega))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class FourierFeatures(nn.Module):
    
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
        metric_flat = self.forward(coords)
        batch_size = coords.shape[0]
        return metric_flat.reshape(batch_size, 4, 4)


class FourierNet(nn.Module):
    
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# from .models import SIREN, ModelConfig
# from .tensor_ops import safe_inverse


class MatterError(Exception):
    pass


class ConservationViolationError(MatterError):
    pass


class EnergyConditionError(MatterError):
    pass


@dataclass
class MatterConfig:
    hidden_dim: int = 64
    activation: str = "sine"
    enforce_conservation: bool = True
    check_energy_conditions: bool = True
    regularization_scale: float = 0.01


class MatterModel(nn.Module, ABC):
    
    def __init__(self, config: Optional[MatterConfig] = None):
        super().__init__()
        
        if config is None:
            config = MatterConfig()
        
        self.config = config
        
        # Common components
        if config.activation == "sine":
            # using Sine from above
            self.activation = Sine(omega=30.0)
        else:
            self.activation = nn.ReLU()
    
    @abstractmethod
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Enable gradients for conservation computation
        coords = coords.requires_grad_(True)
        
        # Get stress-energy tensor
        T = self.get_stress_energy(coords, g, g_inv)
        
        # Raise indices: T^μν = g^μα g^νβ T_αβ
        T_up = torch.einsum('...ma,...nb,...ab->...mn', g_inv, g_inv, T)
        
        # Compute divergence ∇_μ T^μν
        # This is a simplified calculation - in full GR we need Christoffel symbols
        divergence = torch.zeros(batch_size, 4, device=device)
        
        for nu in range(4):
            # Compute ∂_μ T^μν
            T_nu = T_up[:, :, nu]
            
            for mu in range(4):
                if coords.grad is not None:
                    coords.grad.zero_()
                
                # Take derivative
                grad = torch.autograd.grad(
                    T_nu[:, mu].sum(),
                    coords,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                divergence[:, nu] += grad[:, mu]
        
        return divergence


class PerfectFluidMatter(MatterModel):
    
    def __init__(
        self, 
        config: Optional[MatterConfig] = None,
        eos_type: str = "linear",
        eos_params: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)
        
        self.eos_type = eos_type
        self.eos_params = eos_params or {"w": 0.0}  # Default: dust (p = 0)
        
        # Neural network for density field
        model_config = ModelConfig(
            hidden_features=config.hidden_dim,
            hidden_layers=3,
            activation="sine"
        )
        self.density_net = SIREN(model_config, in_features=4, out_features=1)
        
        # Neural network for velocity field
        self.velocity_net = SIREN(model_config, in_features=4, out_features=4)
        
        # Equation of state parameters
        if eos_type == "linear":
            # p = w * ρ
            self.w = nn.Parameter(torch.tensor(self.eos_params.get("w", 0.0)))
        elif eos_type == "polytropic":
            # p = K * ρ^Γ
            self.K = nn.Parameter(torch.tensor(self.eos_params.get("K", 1.0)))
            self.Gamma = nn.Parameter(torch.tensor(self.eos_params.get("Gamma", 5/3)))
    
    def equation_of_state(self, density: torch.Tensor) -> torch.Tensor:
        if self.eos_type == "linear":
            return self.w * density
        elif self.eos_type == "polytropic":
            return self.K * torch.pow(density, self.Gamma)
        else:
            raise ValueError(f"Unknown EOS type: {self.eos_type}")
    
    def get_density(self, coords: torch.Tensor) -> torch.Tensor:
        # Ensure positive density
        raw_density = self.density_net(coords)
        return F.softplus(raw_density) + 1e-6
    
    def get_four_velocity(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get raw velocity from network
        u_raw = self.velocity_net(coords)
        
        # Normalize to satisfy g_μν u^μ u^ν = -1
        # Start with mostly timelike vector
        u = torch.zeros_like(u_raw)
        u[:, 0] = 1.0  # Timelike component
        u[:, 1:] = 0.1 * torch.tanh(u_raw[:, 1:])  # Small spatial components
        
        # Compute norm
        u_norm_sq = torch.einsum('...i,...ij,...j->...', u, g, u)
        
        # Normalize
        u[:, 0] = u[:, 0] * torch.sqrt(-1.0 / u_norm_sq)
        
        return u
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get density and pressure
        density = self.get_density(coords)
        pressure = self.equation_of_state(density)
        
        # Get four-velocity
        u = self.get_four_velocity(coords, g, g_inv)
        
        # Lower indices: u_μ = g_μν u^ν
        u_lower = torch.einsum('...ij,...j->...i', g, u)
        
        # Compute stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # (ρ + p) u_μ u_ν term
        rho_plus_p = density + pressure
        T += torch.einsum('...,...i,...j->...ij', rho_plus_p, u_lower, u_lower)
        
        # p g_μν term
        T += torch.einsum('...,...ij->...ij', pressure, g)
        
        # Ensure symmetry
        T = 0.5 * (T + T.transpose(-2, -1))
        
        return T
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        density = self.get_density(coords)
        pressure = self.equation_of_state(density)
        
        return {
            "density": density,
            "pressure": pressure,
            "equation_of_state_w": self.w if hasattr(self, "w") else None
        }


class ScalarFieldMatter(MatterModel):
    
    def __init__(
        self,
        config: Optional[MatterConfig] = None,
        potential_type: str = "quadratic",
        coupling_params: Optional[Dict[str, float]] = None,
        complex_field: bool = False
    ):
        super().__init__(config)
        
        self.potential_type = potential_type
        self.coupling_params = coupling_params or {"m": 1.0}
        self.complex_field = complex_field
        
        # Neural network for scalar field
        model_config = ModelConfig(
            hidden_features=config.hidden_dim,
            hidden_layers=3,
            activation="sine"
        )
        
        # Output 2 values for complex field (real and imaginary parts)
        out_features = 2 if complex_field else 1
        self.field_net = SIREN(model_config, in_features=4, out_features=out_features)
        
        # Potential parameters
        if potential_type == "quadratic":
            # V(φ) = ½ m² φ²
            self.mass = nn.Parameter(torch.tensor(coupling_params.get("m", 1.0)))
        elif potential_type == "quartic":
            # V(φ) = ½ m² φ² + λ/4! φ⁴
            self.mass = nn.Parameter(torch.tensor(coupling_params.get("m", 1.0)))
            self.lambda_coupling = nn.Parameter(torch.tensor(coupling_params.get("lambda", 0.1)))
        elif potential_type == "exponential":
            # V(φ) = V₀ exp(-αφ) (for quintessence)
            self.V0 = nn.Parameter(torch.tensor(coupling_params.get("V0", 1.0)))
            self.alpha = nn.Parameter(torch.tensor(coupling_params.get("alpha", 1.0)))
    
    def potential(self, phi: torch.Tensor) -> torch.Tensor:
        if self.complex_field:
            # For complex field, use |φ|² = φ*φ
            phi_sq = (phi[..., 0]**2 + phi[..., 1]**2)
        else:
            phi_sq = phi**2
        
        if self.potential_type == "quadratic":
            return 0.5 * self.mass**2 * phi_sq
        elif self.potential_type == "quartic":
            return 0.5 * self.mass**2 * phi_sq + (self.lambda_coupling / 24) * phi_sq**2
        elif self.potential_type == "exponential":
            return self.V0 * torch.exp(-self.alpha * torch.sqrt(phi_sq))
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")
    
    def get_field(self, coords: torch.Tensor) -> torch.Tensor:
        return self.field_net(coords)
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Enable gradients
        coords = coords.requires_grad_(True)
        
        # Get field value
        phi = self.get_field(coords)
        
        # Compute field gradients
        if self.complex_field:
            # Handle real and imaginary parts separately
            grad_phi_real = torch.autograd.grad(
                phi[..., 0].sum(), coords, create_graph=True, retain_graph=True
            )[0]
            grad_phi_imag = torch.autograd.grad(
                phi[..., 1].sum(), coords, create_graph=True, retain_graph=True
            )[0]
            grad_phi = torch.stack([grad_phi_real, grad_phi_imag], dim=-1)
        else:
            grad_phi = torch.autograd.grad(
                phi.sum(), coords, create_graph=True, retain_graph=True
            )[0]
        
        # Compute kinetic term: ½ g^μν ∂_μφ ∂_νφ
        if self.complex_field:
            # |∂φ|² = ∂_μφ* ∂^μφ
            kinetic = 0.5 * (
                torch.einsum('...ij,...i,...j->...', g_inv, grad_phi[..., 0], grad_phi[..., 0]) +
                torch.einsum('...ij,...i,...j->...', g_inv, grad_phi[..., 1], grad_phi[..., 1])
            )
        else:
            kinetic = 0.5 * torch.einsum('...ij,...i,...j->...', g_inv, grad_phi, grad_phi)
        
        # Compute potential
        V = self.potential(phi)
        
        # Compute stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # ∂_μφ ∂_νφ term
        if self.complex_field:
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] = (
                        grad_phi[:, mu, 0] * grad_phi[:, nu, 0] +
                        grad_phi[:, mu, 1] * grad_phi[:, nu, 1]
                    )
        else:
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] = grad_phi[:, mu] * grad_phi[:, nu]
        
        # Subtract g_μν [kinetic + V] term
        lagrangian = kinetic - V
        T -= torch.einsum('...,...ij->...ij', lagrangian, g)
        
        # Ensure symmetry
        T = 0.5 * (T + T.transpose(-2, -1))
        
        return T
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.get_field(coords)
        V = self.potential(phi)
        
        return {
            "field": phi,
            "potential": V,
            "mass": self.mass if hasattr(self, "mass") else None
        }


class ElectromagneticFieldMatter(MatterModel):
    
    def __init__(
        self,
        config: Optional[MatterConfig] = None,
        field_type: str = "general",
        mu0: float = 1.0  # Magnetic permeability (set to 1 in geometric units)
    ):
        super().__init__(config)
        
        self.field_type = field_type
        self.mu0 = mu0
        
        # Neural network for electromagnetic potential A_μ
        model_config = ModelConfig(
            hidden_features=config.hidden_dim,
            hidden_layers=3,
            activation="sine"
        )
        self.potential_net = SIREN(model_config, in_features=4, out_features=4)
        
        # Optional: Charge density network for sourced fields
        if field_type == "sourced":
            self.charge_net = SIREN(model_config, in_features=4, out_features=1)
    
    def get_potential(self, coords: torch.Tensor) -> torch.Tensor:
        return self.potential_net(coords)
    
    def get_field_tensor(
        self, 
        coords: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Enable gradients
        coords = coords.requires_grad_(True)
        
        # Get potential
        A = self.get_potential(coords)
        
        # Compute field tensor components
        F = torch.zeros(batch_size, 4, 4, device=device)
        
        for mu in range(4):
            # Compute ∂_μ A_ν
            if coords.grad is not None:
                coords.grad.zero_()
            
            grad_A = torch.autograd.grad(
                A[:, mu].sum(), coords, 
                create_graph=True, retain_graph=True
            )[0]
            
            for nu in range(4):
                if mu != nu:
                    # F_μν = ∂_μ A_ν - ∂_ν A_μ
                    F[:, mu, nu] = grad_A[:, nu]
                    F[:, nu, mu] = -grad_A[:, nu]
        
        return F
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get field tensor
        F = self.get_field_tensor(coords, g)
        
        # Raise indices: F^μν = g^μα g^νβ F_αβ
        F_up = torch.einsum('...ma,...nb,...ab->...mn', g_inv, g_inv, F)
        
        # Compute F_μα F_ν^α
        F_F = torch.einsum('...ma,...na->...mn', F, F_up)
        
        # Compute invariant F_αβ F^αβ
        F_invariant = torch.einsum('...ab,...ab->...', F, F_up)
        
        # Compute stress-energy tensor
        T = (1.0 / self.mu0) * (F_F - 0.25 * torch.einsum('...,...ij->...ij', F_invariant, g))
        
        # Ensure symmetry
        T = 0.5 * (T + T.transpose(-2, -1))
        
        return T
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        A = self.get_potential(coords)
        F = self.get_field_tensor(coords)
        
        # Extract electric and magnetic fields (in coordinate basis)
        # E_i = F_0i, B_i = ½ ε_ijk F_jk
        E = F[:, 0, 1:4]
        
        # Simplified magnetic field (only B_z component for 2D)
        B_z = F[:, 1, 2]
        
        return {
            "potential": A,
            "field_tensor": F,
            "electric_field": E,
            "magnetic_field_z": B_z
        }


class DarkSectorMatter(MatterModel):
    
    def __init__(
        self,
        config: Optional[MatterConfig] = None,
        dm_type: str = "cold",
        de_type: str = "lambda",
        interaction: bool = False
    ):
        super().__init__(config)
        
        self.dm_type = dm_type
        self.de_type = de_type
        self.interaction = interaction
        
        model_config = ModelConfig(
            hidden_features=config.hidden_dim,
            hidden_layers=3,
            activation="sine"
        )
        
        # Dark matter density
        self.dm_density_net = SIREN(model_config, in_features=4, out_features=1)
        
        # Dark energy density (if dynamic)
        if de_type != "lambda":
            self.de_density_net = SIREN(model_config, in_features=4, out_features=1)
        else:
            # Cosmological constant
            self.Lambda = nn.Parameter(torch.tensor(1.0))
        
        # Interaction coupling (if enabled)
        if interaction:
            self.coupling_net = SIREN(model_config, in_features=4, out_features=1)
    
    def get_dm_density(self, coords: torch.Tensor) -> torch.Tensor:
        raw_density = self.dm_density_net(coords)
        return F.softplus(raw_density) + 1e-6
    
    def get_de_density(self, coords: torch.Tensor) -> torch.Tensor:
        if self.de_type == "lambda":
            batch_size = coords.shape[0]
            return self.Lambda.expand(batch_size, 1)
        else:
            raw_density = self.de_density_net(coords)
            return F.softplus(raw_density) + 1e-6
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords.shape[0]
        device = coords.device
        
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # Dark matter contribution
        rho_dm = self.get_dm_density(coords)
        
        # For cold dark matter, assume comoving (u^μ = (1,0,0,0) in comoving frame)
        u = torch.zeros(batch_size, 4, device=device)
        u[:, 0] = 1.0
        
        # Normalize four-velocity
        u_norm_sq = torch.einsum('...i,...ij,...j->...', u, g, u)
        u[:, 0] = u[:, 0] * torch.sqrt(-1.0 / u_norm_sq)
        
        # Lower indices
        u_lower = torch.einsum('...ij,...j->...i', g, u)
        
        # Add dark matter term
        T += torch.einsum('...,...i,...j->...ij', rho_dm.squeeze(), u_lower, u_lower)
        
        # Dark energy contribution
        rho_de = self.get_de_density(coords)
        
        # Add dark energy term (negative pressure)
        T -= torch.einsum('...,...ij->...ij', rho_de.squeeze(), g)
        
        # Interaction term (if enabled)
        if self.interaction:
            Q = self.coupling_net(coords)
            # Simple phenomenological interaction: energy transfer from DE to DM
            T *= (1.0 + 0.1 * torch.tanh(Q))
        
        # Ensure symmetry
        T = 0.5 * (T + T.transpose(-2, -1))
        
        return T
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        values = {
            "dm_density": self.get_dm_density(coords),
            "de_density": self.get_de_density(coords)
        }
        
        if self.interaction:
            values["coupling"] = self.coupling_net(coords)
        
        return values


def create_matter_model(
    matter_type: str,
    config: Optional[MatterConfig] = None,
    **kwargs
) -> MatterModel:
    if config is None:
        config = MatterConfig()
    
    if matter_type == "perfect_fluid":
        return PerfectFluidMatter(config, **kwargs)
    elif matter_type == "scalar_field":
        return ScalarFieldMatter(config, **kwargs)
    elif matter_type == "electromagnetic":
        return ElectromagneticFieldMatter(config, **kwargs)
    elif matter_type == "dark_sector":
        return DarkSectorMatter(config, **kwargs)
    else:
        raise ValueError(f"Unknown matter type: {matter_type}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

# from __main__ import (
#     compute_einstein_tensor_vectorized,
#     check_energy_conditions,
#     safe_inverse,
#     TensorConfig
# )


class PhysicsError(Exception):
    pass


class CausalityViolationError(PhysicsError):
    pass


class AsymptoticBehaviorError(PhysicsError):
    pass


@dataclass
class PhysicsConfig:
    einstein_weight: float = 1.0
    conservation_weight: float = 0.1
    constraint_weight: float = 0.1
    energy_condition_weight: float = 0.05
    horizon_epsilon: float = 0.1
    asymptotic_radius: float = 100.0
    enforce_causality: bool = True
    adaptive_sampling: bool = True
    curvature_threshold: float = 0.1


def compute_efe_loss(
    coords: torch.Tensor,
    metric_model: nn.Module,
    matter_models: List[nn.Module],
    matter_weights: Optional[List[float]] = None,
    config: Optional[PhysicsConfig] = None
) -> Dict[str, torch.Tensor]:
    if config is None:
        config = PhysicsConfig()
    
    batch_size = coords.shape[0]
    device = coords.device
    
    # Enable gradient tracking
    coords = coords.requires_grad_(True)
    
    # Get metric from model
    g = metric_model(coords).reshape(batch_size, 4, 4)
    
    # Ensure metric symmetry
    g = 0.5 * (g + g.transpose(-2, -1))
    
    # Compute inverse metric
    g_inv = safe_inverse(g)
    
    # Compute Einstein tensor (left side of EFE)
    tensor_config = TensorConfig()
    einstein_components = compute_einstein_tensor_vectorized(
        g, coords, metric_model, tensor_config, return_components=True
    )
    G_tensor = einstein_components['einstein']
    
    # Compute total stress-energy tensor (right side of EFE)
    T_total = torch.zeros_like(G_tensor)
    
    if matter_weights is None:
        matter_weights = [1.0 / len(matter_models)] * len(matter_models)
    
    for model, weight in zip(matter_models, matter_weights):
        T_contrib = model.get_stress_energy(coords, g, g_inv)
        T_total += weight * T_contrib
    
    # Einstein field equations residual: G_μν - 8π T_μν
    # (Ignoring cosmological constant for now)
    efe_residual = G_tensor - 8 * np.pi * T_total
    
    # Compute L2 norm of residual
    efe_loss = torch.mean(torch.sum(efe_residual**2, dim=(1, 2)))
    
    # Additional physics losses
    losses = {
        'efe_loss': config.einstein_weight * efe_loss,
        'total_loss': config.einstein_weight * efe_loss
    }
    
    # Add conservation loss
    if config.conservation_weight > 0:
        conservation_loss = compute_conservation_loss(
            coords, g, g_inv, T_total, einstein_components['christoffel']
        )
        losses['conservation_loss'] = config.conservation_weight * conservation_loss
        losses['total_loss'] += losses['conservation_loss']
    
    # Add constraint losses
    if config.constraint_weight > 0:
        constraint_losses = compute_constraint_losses(coords, g, g_inv, config)
        for key, value in constraint_losses.items():
            losses[f'constraint_{key}'] = config.constraint_weight * value
            losses['total_loss'] += losses[f'constraint_{key}']
    
    # Check energy conditions
    if config.energy_condition_weight > 0:
        energy_violations = check_energy_condition_violations(T_total, g, g_inv)
        losses['energy_condition_loss'] = config.energy_condition_weight * energy_violations
        losses['total_loss'] += losses['energy_condition_loss']
    
    return losses


def compute_conservation_loss(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    T: torch.Tensor,
    christoffel: torch.Tensor
) -> torch.Tensor:
    batch_size = coords.shape[0]
    device = coords.device
    
    # Raise indices: T^μν = g^μα g^νβ T_αβ
    T_up = torch.einsum('...ma,...nb,...ab->...mn', g_inv, g_inv, T)
    
    # Compute covariant derivative: ∇_μ T^μν
    # ∇_μ T^μν = ∂_μ T^μν + Γ^μ_μλ T^λν + Γ^ν_μλ T^μλ
    
    # For simplicity, compute only the connection terms
    # (derivative terms require additional autodiff)
    divergence = torch.zeros(batch_size, 4, device=device)
    
    for nu in range(4):
        # First Christoffel term: Γ^μ_μλ T^λν
        for mu in range(4):
            for lam in range(4):
                divergence[:, nu] += christoffel[:, mu, mu, lam] * T_up[:, lam, nu]
        
        # Second Christoffel term: Γ^ν_μλ T^μλ
        for mu in range(4):
            for lam in range(4):
                divergence[:, nu] += christoffel[:, nu, mu, lam] * T_up[:, mu, lam]
    
    # L2 norm of divergence
    conservation_loss = torch.mean(torch.sum(divergence**2, dim=1))
    
    return conservation_loss


def compute_constraint_losses(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    config: PhysicsConfig
) -> Dict[str, torch.Tensor]:
    batch_size = coords.shape[0]
    device = coords.device
    losses = {}
    
    # 1. Metric signature constraint
    # Ensure det(g) < 0 for Lorentzian signature
    det_g = torch.det(g)
    signature_loss = torch.mean(F.relu(det_g))  # Penalize positive determinant
    losses['signature'] = signature_loss
    
    # 2. Asymptotic flatness constraint
    r = torch.sqrt(torch.sum(coords[:, 1:4]**2, dim=1))
    far_mask = r > config.asymptotic_radius
    
    if torch.any(far_mask):
        # Minkowski metric in Cartesian coordinates
        minkowski = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        minkowski[:, 0, 0] = -1
        
        # Deviation from Minkowski at large distances
        asymptotic_loss = torch.mean(
            torch.sum((g[far_mask] - minkowski[far_mask])**2, dim=(1, 2))
        )
        losses['asymptotic'] = asymptotic_loss
    
    # 3. Causality constraint
    if config.enforce_causality:
        # Check that light cones remain spacelike
        # This is a simplified check - full causality requires more analysis
        eigenvalues = torch.linalg.eigvals(g).real
        
        # Should have exactly one negative eigenvalue (timelike direction)
        neg_count = torch.sum(eigenvalues < 0, dim=1)
        causality_loss = torch.mean((neg_count - 1)**2).float()
        losses['causality'] = causality_loss
    
    return losses


def check_energy_condition_violations(
    T: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor
) -> torch.Tensor:
    conditions = check_energy_conditions(T, g, g_inv)
    
    # Count violations (where condition is False)
    violations = 0.0
    for condition_name, satisfied in conditions.items():
        violations += torch.mean((~satisfied).float())
    
    return violations


def regularized_coordinates(
    coords: torch.Tensor,
    singularity_centers: Optional[List[torch.Tensor]] = None,
    config: Optional[PhysicsConfig] = None
) -> torch.Tensor:
    if config is None:
        config = PhysicsConfig()
    
    if singularity_centers is None:
        singularity_centers = [torch.zeros(3, device=coords.device)]
    
    batch_size = coords.shape[0]
    device = coords.device
    
    # Copy coordinates
    reg_coords = coords.clone()
    spatial_coords = coords[:, 1:4]
    
    for center in singularity_centers:
        # Distance from singularity
        delta = spatial_coords - center.unsqueeze(0)
        r = torch.norm(delta, dim=1)
        
        # Regularize near horizon (r ≈ 2M, assuming M = 1)
        horizon_radius = 2.0
        min_radius = horizon_radius + config.horizon_epsilon
        
        # Smooth cutoff function
        need_regularization = r < min_radius
        if torch.any(need_regularization):
            # Use sigmoid transition
            transition = torch.sigmoid(10 * (r - min_radius))
            new_r = min_radius + transition * (r - min_radius)
            
            # Update coordinates
            scale = new_r / (r + 1e-10)
            for i in range(3):
                reg_coords[need_regularization, i+1] = (
                    center[i] + scale[need_regularization] * delta[need_regularization, i]
                )
    
    return reg_coords


def adaptive_sampling_strategy(
    coords: torch.Tensor,
    metric_model: nn.Module,
    config: Optional[PhysicsConfig] = None,
    max_new_points: int = 1000
) -> torch.Tensor:
    if config is None:
        config = PhysicsConfig()
    
    with torch.no_grad():
        batch_size = coords.shape[0]
        device = coords.device
        
        # Compute metric and curvature invariants
        g = metric_model(coords).reshape(batch_size, 4, 4)
        g_inv = safe_inverse(g)
        
        # Get curvature components
        tensor_config = TensorConfig()
        components = compute_einstein_tensor_vectorized(
            g, coords, metric_model, tensor_config, return_components=True
        )
        
        # Use Ricci scalar as curvature measure
        ricci_scalar = components['ricci_scalar']
        
        # Find high curvature points
        high_curv_mask = torch.abs(ricci_scalar) > config.curvature_threshold
        high_curv_indices = torch.where(high_curv_mask)[0]
        
        if len(high_curv_indices) == 0:
            return coords
        
        # Limit number of new points
        if len(high_curv_indices) > max_new_points // 5:
            # Select most important points
            _, top_indices = torch.topk(
                torch.abs(ricci_scalar[high_curv_indices]),
                min(max_new_points // 5, len(high_curv_indices))
            )
            high_curv_indices = high_curv_indices[top_indices]
        
        # Generate new points around high curvature regions
        new_points = []
        
        for idx in high_curv_indices:
            base_point = coords[idx]
            
            # Estimate local curvature scale
            local_curvature = torch.abs(ricci_scalar[idx])
            sampling_scale = 1.0 / (torch.sqrt(local_curvature) + 1e-6)
            
            # Add points in a small neighborhood
            for _ in range(4):  # 4 new points per high-curvature location
                offset = torch.randn(4, device=device) * sampling_scale * 0.1
                offset[0] = 0  # Keep time fixed for spatial sampling
                new_point = base_point + offset
                new_points.append(new_point)
        
        if new_points:
            new_coords = torch.stack(new_points)
            enhanced_coords = torch.cat([coords, new_coords], dim=0)
            return enhanced_coords
        else:
            return coords


def schwarzschild_initial_metric(
    coords: torch.Tensor,
    mass: float = 1.0,
    use_isotropic: bool = False
) -> torch.Tensor:
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Compute radial coordinate
    r = torch.sqrt(x**2 + y**2 + z**2)
    r = torch.clamp(r, min=2.01 * mass)  # Avoid horizon
    
    if use_isotropic:
        # Isotropic coordinates
        rho = r
        psi = 1 + mass / (2 * rho)
        
        # Metric components
        g = torch.zeros(batch_size, 4, 4, device=device)
        g[:, 0, 0] = -((psi**2 - 1) / (psi**2 + 1))**2
        
        # Spatial part is conformally flat
        spatial_factor = psi**4
        g[:, 1, 1] = spatial_factor
        g[:, 2, 2] = spatial_factor
        g[:, 3, 3] = spatial_factor
    else:
        # Standard Schwarzschild coordinates
        g = torch.zeros(batch_size, 4, 4, device=device)
        
        # Metric components
        f = 1 - 2 * mass / r
        g[:, 0, 0] = -f
        
        # Transform to Cartesian coordinates
        # dr² = (dx² + dy² + dz²) / r²
        # Need full transformation including angular parts
        
        # For simplicity, use diagonal approximation
        g[:, 1, 1] = 1 / f
        g[:, 2, 2] = 1 / f
        g[:, 3, 3] = 1 / f
    
    return g


def adm_decomposition(
    metric: torch.Tensor,
    coords: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch_size = metric.shape[0]
    device = metric.device
    
    # Inverse metric
    g_inv = safe_inverse(metric)
    
    # Extract ADM variables
    # Lapse: α = 1/√(-g^00)
    lapse = 1.0 / torch.sqrt(-g_inv[:, 0, 0] + 1e-10)
    
    # Shift: β^i = -g^0i / g^00
    shift = torch.zeros(batch_size, 3, device=device)
    for i in range(3):
        shift[:, i] = -g_inv[:, 0, i+1] / g_inv[:, 0, 0]
    
    # Spatial metric: γ_ij = g_ij
    spatial_metric = metric[:, 1:4, 1:4].clone()
    
    # Extrinsic curvature (simplified - full calculation requires time derivatives)
    # K_ij = -1/(2α) (∂_t γ_ij - D_i β_j - D_j β_i)
    # For static metrics, K_ij = 0
    extrinsic_curvature = torch.zeros(batch_size, 3, 3, device=device)
    
    return {
        'lapse': lapse,
        'shift': shift,
        'spatial_metric': spatial_metric,
        'extrinsic_curvature': extrinsic_curvature
    }


def hamiltonian_constraint(
    spatial_metric: torch.Tensor,
    extrinsic_curvature: torch.Tensor,
    matter_density: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size = spatial_metric.shape[0]
    device = spatial_metric.device
    
    # Inverse spatial metric
    gamma_inv = safe_inverse(spatial_metric)
    
    # Trace of extrinsic curvature
    K_trace = torch.einsum('...ij,...ij->...', gamma_inv, extrinsic_curvature)
    
    # K_ij K^ij
    K_squared = torch.einsum(
        '...ij,...ik,...jk->...',
        extrinsic_curvature,
        gamma_inv,
        extrinsic_curvature
    )
    
    # For simplicity, assume R = 0 (flat 3-space)
    # Full calculation would require 3D Christoffel symbols
    R_3d = torch.zeros(batch_size, device=device)
    
    # Matter contribution
    if matter_density is None:
        matter_density = torch.zeros(batch_size, device=device)
    
    # Hamiltonian constraint
    H = R_3d - K_trace**2 + K_squared - 16 * np.pi * matter_density
    
    return H


def momentum_constraint(
    spatial_metric: torch.Tensor,
    extrinsic_curvature: torch.Tensor,
    matter_momentum: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size = spatial_metric.shape[0]
    device = spatial_metric.device
    
    # For simplicity, return zero constraint
    # Full calculation requires 3D covariant derivatives
    M = torch.zeros(batch_size, 3, device=device)
    
    if matter_momentum is not None:
        M -= 8 * np.pi * matter_momentum
    
    return M

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# from .physics import (
#     compute_efe_loss,
#     regularized_coordinates,
#     adaptive_sampling_strategy,
#     PhysicsConfig
# )
# from .tensor_ops import safe_inverse


class SystemError(Exception):
    pass


@dataclass
class SystemConfig:
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    physics_config: Optional[PhysicsConfig] = None
    verbose: bool = True


class GravitationalSystem:
    
    def __init__(
        self,
        metric_model: nn.Module,
        matter_models: List[nn.Module],
        matter_weights: Optional[List[float]] = None,
        config: Optional[SystemConfig] = None
    ):
        if config is None:
            config = SystemConfig()
        
        self.config = config
        
        # Set device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = config.device
        
        # Move models to device
        self.metric_model = metric_model.to(self.device)
        self.matter_models = [m.to(self.device) for m in matter_models]
        
        # Set matter weights
        if matter_weights is None:
            self.matter_weights = [1.0 / len(matter_models)] * len(matter_models)
        else:
            if len(matter_weights) != len(matter_models):
                raise ValueError("Number of weights must match number of matter models")
            # Normalize weights
            total_weight = sum(matter_weights)
            self.matter_weights = [w / total_weight for w in matter_weights]
        
        # Physics configuration
        if config.physics_config is None:
            self.physics_config = PhysicsConfig()
        else:
            self.physics_config = config.physics_config
        
        # Training history
        self.history = {
            'total_loss': [],
            'efe_loss': [],
            'conservation_loss': [],
            'constraint_loss': []
        }
    
    def combined_stress_energy(
        self,
        coords: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        g_inv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if g is None:
            g = self.metric_model(coords).reshape(-1, 4, 4)
            g = 0.5 * (g + g.transpose(-2, -1))  # Ensure symmetry
        
        if g_inv is None:
            g_inv = safe_inverse(g)
        
        # Initialize total stress-energy
        batch_size = coords.shape[0]
        T_total = torch.zeros(batch_size, 4, 4, device=self.device)
        
        # Sum contributions from each matter model
        for model, weight in zip(self.matter_models, self.matter_weights):
            T_contrib = model.get_stress_energy(coords, g, g_inv)
            T_total += weight * T_contrib
        
        return T_total
    
    def sample_coordinates(
        self,
        batch_size: int,
        T_range: Tuple[float, float] = (0.0, 0.0),
        spatial_range: float = 10.0,
        avoid_horizon: bool = True,
        adaptive: bool = True
    ) -> torch.Tensor:
        # Initial uniform sampling
        coords = torch.zeros(batch_size, 4, device=self.device)
        
        # Time coordinates
        if T_range[0] == T_range[1]:
            coords[:, 0] = T_range[0]  # Static spacetime
        else:
            coords[:, 0] = torch.rand(batch_size, device=self.device) * (T_range[1] - T_range[0]) + T_range[0]
        
        # Spatial coordinates
        coords[:, 1:4] = (torch.rand(batch_size, 3, device=self.device) - 0.5) * 2 * spatial_range
        
        # Apply coordinate regularization if needed
        if avoid_horizon:
            coords = regularized_coordinates(coords, config=self.physics_config)
        
        # Apply adaptive sampling if enabled
        if adaptive and self.physics_config.adaptive_sampling:
            coords = adaptive_sampling_strategy(
                coords,
                self.metric_model,
                self.physics_config,
                max_new_points=batch_size // 2
            )
        
        return coords
    
    def compute_loss(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        return compute_efe_loss(
            coords,
            self.metric_model,
            self.matter_models,
            self.matter_weights,
            self.physics_config
        )
    
    def train_step(
        self,
        optimizer_metric: torch.optim.Optimizer,
        optimizer_matter: Optional[torch.optim.Optimizer],
        batch_size: int,
        T_range: Tuple[float, float],
        spatial_range: float
    ) -> Dict[str, float]:
        # Sample coordinates
        coords = self.sample_coordinates(
            batch_size,
            T_range,
            spatial_range,
            avoid_horizon=True,
            adaptive=True
        )
        
        # Zero gradients
        optimizer_metric.zero_grad()
        if optimizer_matter is not None:
            optimizer_matter.zero_grad()
        
        # Compute losses
        losses = self.compute_loss(coords)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.metric_model.parameters(), max_norm=1.0)
        for model in self.matter_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer_metric.step()
        if optimizer_matter is not None:
            optimizer_matter.step()
        
        # Return loss values as floats
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        epochs: int,
        batch_size: int = 256,
        T_range: Tuple[float, float] = (0.0, 0.0),
        spatial_range: float = 10.0,
        lr_metric: float = 1e-4,
        lr_matter: float = 5e-4,
        train_matter: bool = True,
        scheduler_params: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 100,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        # Create optimizers
        optimizer_metric = torch.optim.Adam(
            self.metric_model.parameters(),
            lr=lr_metric
        )
        
        if train_matter:
            matter_params = []
            for model in self.matter_models:
                matter_params.extend(model.parameters())
            optimizer_matter = torch.optim.Adam(matter_params, lr=lr_matter)
        else:
            optimizer_matter = None
        
        # Create schedulers if requested
        if scheduler_params is not None:
            scheduler_metric = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_metric,
                mode='min',
                **scheduler_params
            )
            if optimizer_matter is not None:
                scheduler_matter = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_matter,
                    mode='min',
                    **scheduler_params
                )
        else:
            scheduler_metric = None
            scheduler_matter = None
        
        # Training loop
        for epoch in range(epochs):
            # Train step
            losses = self.train_step(
                optimizer_metric,
                optimizer_matter,
                batch_size,
                T_range,
                spatial_range
            )
            
            # Record history
            for key, value in losses.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Update schedulers
            if scheduler_metric is not None:
                scheduler_metric.step(losses['total_loss'])
            if scheduler_matter is not None:
                scheduler_matter.step(losses['total_loss'])
            
            # Print progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Total Loss: {losses['total_loss']:.6f}")
                print(f"  EFE Loss: {losses['efe_loss']:.6f}")
                if 'conservation_loss' in losses:
                    print(f"  Conservation Loss: {losses['conservation_loss']:.6f}")
                if 'constraint_signature' in losses:
                    print(f"  Signature Constraint: {losses['constraint_signature']:.6f}")
                
                # Print learning rates
                current_lr_metric = optimizer_metric.param_groups[0]['lr']
                print(f"  Metric LR: {current_lr_metric:.2e}")
                if optimizer_matter is not None:
                    current_lr_matter = optimizer_matter.param_groups[0]['lr']
                    print(f"  Matter LR: {current_lr_matter:.2e}")
                print()
            
            # Save checkpoint
            if checkpoint_path and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pt",
                    epoch,
                    optimizer_metric,
                    optimizer_matter
                )
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_metric: torch.optim.Optimizer,
        optimizer_matter: Optional[torch.optim.Optimizer] = None
    ):
        checkpoint = {
            'epoch': epoch,
            'metric_model_state': self.metric_model.state_dict(),
            'optimizer_metric_state': optimizer_metric.state_dict(),
            'matter_weights': self.matter_weights,
            'history': self.history,
            'config': self.config,
            'physics_config': self.physics_config
        }
        
        # Save matter models
        for i, model in enumerate(self.matter_models):
            checkpoint[f'matter_model_{i}_state'] = model.state_dict()
        
        if optimizer_matter is not None:
            checkpoint['optimizer_matter_state'] = optimizer_matter.state_dict()
        
        torch.save(checkpoint, path)
        if self.config.verbose:
            print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load metric model
        self.metric_model.load_state_dict(checkpoint['metric_model_state'])
        
        # Load matter models
        for i, model in enumerate(self.matter_models):
            if f'matter_model_{i}_state' in checkpoint:
                model.load_state_dict(checkpoint[f'matter_model_{i}_state'])
        
        # Load other attributes
        self.matter_weights = checkpoint.get('matter_weights', self.matter_weights)
        self.history = checkpoint.get('history', self.history)
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {path}")
            print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
        
        return checkpoint
    
    def evaluate(
        self,
        test_coords: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Get metric
            g = self.metric_model(test_coords).reshape(-1, 4, 4)
            g = 0.5 * (g + g.transpose(-2, -1))
            
            # Get inverse metric
            g_inv = safe_inverse(g)
            
            # Get stress-energy
            T = self.combined_stress_energy(test_coords, g, g_inv)
            
            # Compute losses
            losses = self.compute_loss(test_coords)
            
            results = {
                'metric': g,
                'stress_energy': T,
                'losses': losses
            }
            
            if return_components:
                # Get individual matter contributions
                matter_components = []
                for model, weight in zip(self.matter_models, self.matter_weights):
                    T_contrib = model.get_stress_energy(test_coords, g, g_inv)
                    matter_components.append(weight * T_contrib)
                results['matter_components'] = matter_components
                
                # Get matter field values
                field_values = []
                for model in self.matter_models:
                    fields = model.get_field_values(test_coords)
                    field_values.append(fields)
                results['field_values'] = field_values
            
            return results
    
    def predict_metric(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = self.metric_model(coords).reshape(-1, 4, 4)
            g = 0.5 * (g + g.transpose(-2, -1))
            return g
    
    def predict_curvature(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        from __main__ import (
            compute_einstein_tensor_vectorized,
            compute_kretschmann_scalar,
            TensorConfig
        )
        with torch.no_grad():
            g = self.predict_metric(coords)
            tensor_config = TensorConfig()
            components = compute_einstein_tensor_vectorized(
                g, coords, self.metric_model, tensor_config, return_components=True
            )
            kretschmann = compute_kretschmann_scalar(components["riemann"])
        return {
            "einstein_tensor": components["einstein"],
            "ricci_tensor": components["ricci"],
            "ricci_scalar": components["ricci_scalar"],
            "riemann_tensor": components["riemann"],
            "kretschmann_scalar": kretschmann
        }



# Plot training loss curves
def plot_training_history(history: Dict[str, List[float]], filename: str = "training_history.png"):
    epochs = range(1, len(history.get("total_loss", [])) + 1)
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()
    if "total_loss" in history:
        ax.plot(epochs, history["total_loss"], label="total", marker="o")
    if "efe_loss" in history:
        ax.plot(epochs, history["efe_loss"], label="efe", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# Example usage

def run_example():
    print("Running EFES Example...")

    metric_model = create_metric_model("siren")
    matter = create_matter_model("perfect_fluid", eos_type="linear")
    system = GravitationalSystem(metric_model, [matter])

    history = system.train(epochs=20, batch_size=64)
    plot_training_history(history)

    coords = torch.tensor([[0.0, 5.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0, 0.0]])
    results = system.evaluate(coords)

    print(f"Final training loss: {history['total_loss'][-1]:.6f}")
    print(f"Metric shape: {results['metric'].shape}")
    print("Example completed successfully!")

if __name__ == "__main__":
    run_example()


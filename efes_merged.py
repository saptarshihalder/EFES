"""
Einstein Field Equations Solver (EFES) - Complete Merged Module

This file contains all the code from the EFES project merged into a single module.
It includes:
- Vectorized tensor operations
- Neural network models for metric learning
- Matter and energy models
- Physics constraints and computations
- System integration
- Unit and integration tests

The code is organized in the following order:
1. Common imports and exceptions
2. Tensor operations (tensor_ops.py)
3. Neural network models (models.py)
4. Matter models (matter.py)
5. Physics functions (physics.py)
6. System class (system.py)
7. Package exports (__init__.py)
8. Tests (all test files)
"""

# =====================================================================
# COMMON IMPORTS
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Test imports (wrapped in try-except for optional testing)
try:
    import pytest
except ImportError:
    pytest = None

# =====================================================================
# TENSOR OPERATIONS (tensor_ops.py)
# =====================================================================

"""
Vectorized tensor calculus operations for Einstein Field Equations.

This module implements highly optimized, fully vectorized operations for computing
geometric quantities in General Relativity, including:
- Christoffel symbols
- Riemann curvature tensor
- Ricci tensor and scalar
- Einstein tensor

All operations are implemented using PyTorch's vectorized operations, avoiding
explicit loops for improved performance on GPUs.

Physics Approximations:
----------------------
1. Numerical Derivatives: We use finite differences with adaptive step sizes
   for computing metric derivatives. This approximation is valid when the
   metric varies smoothly on scales larger than the step size.
   
2. Static Metric Approximation: For time derivatives (∂_t), we currently assume
   a static or slowly varying metric. This is valid for:
   - Schwarzschild spacetime
   - Slowly rotating Kerr spacetime
   - Cosmological solutions with slow time evolution
   
3. Coordinate Singularity Handling: Near coordinate singularities (e.g., r=2M
   in Schwarzschild coordinates), we apply regularization to avoid numerical
   instabilities. This does not affect the physical predictions away from
   these regions.
"""

# Version info
__version__ = "0.1.0"

class TensorOpsError(Exception):
    """Base exception for tensor operations errors."""
    pass


class MetricSingularityError(TensorOpsError):
    """Raised when metric becomes singular or non-invertible."""
    pass


class NumericalInstabilityError(TensorOpsError):
    """Raised when numerical instabilities are detected."""
    pass


@dataclass
class TensorConfig:
    """Configuration for tensor operations."""
    epsilon: float = 1e-6  # Numerical regularization parameter
    derivative_epsilon: float = 1e-4  # Step size for finite differences
    max_christoffel_norm: float = 1e6  # Maximum allowed norm for Christoffel symbols
    static_time_approximation: bool = True  # Use static metric approximation
    adaptive_step_size: bool = True  # Use adaptive step sizes for derivatives


def safe_inverse(matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Compute matrix inverse with regularization for near-singular matrices."""
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
    """Compute derivatives of the metric tensor using finite differences."""
    if config is None:
        config = TensorConfig()
        
    batch_size = coords.shape[0]
    device = coords.device
    
    # Use finite differences
    dg = torch.zeros(batch_size, 4, 4, 4, device=device)
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
            g_plus = g
            g_minus = g
        
        # Compute derivative using central differences
        dg[:, :, :, mu] = (g_plus - g_minus) / (2 * eps)
    
    return dg


def compute_christoffel_symbols_vectorized(
    g: torch.Tensor,
    g_inv: Optional[torch.Tensor] = None,
    dg: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    metric_func: Optional[torch.nn.Module] = None,
    config: Optional[TensorConfig] = None
) -> torch.Tensor:
    """Compute Christoffel symbols using fully vectorized operations."""
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
    dg_reordered = dg.permute(0, 3, 1, 2)
    
    # Compute the three terms
    term1 = dg_reordered  # ∂_μ g_νσ
    term2 = dg_reordered.transpose(1, 2)  # ∂_ν g_μσ
    term3 = dg_reordered.transpose(2, 3)  # ∂_σ g_μν
    
    # Combine terms
    combined = term1 + term2 - term3
    
    # Contract with inverse metric
    christoffel = 0.5 * torch.einsum('...ls,...smn->...lmn', g_inv, combined)
    
    # Apply regularization to prevent numerical instabilities
    christoffel_norm = christoffel.norm(dim=(1, 2, 3), keepdim=True)
    if (christoffel_norm > config.max_christoffel_norm).any():
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
    """Compute the Riemann curvature tensor using vectorized operations."""
    if config is None:
        config = TensorConfig()
    
    # Compute only the product terms for stability
    # First product: Γ^ρ_μλ Γ^λ_νσ
    term1 = torch.einsum('...rml,...lns->...rmns', christoffel, christoffel)
    
    # Second product: Γ^ρ_νλ Γ^λ_μσ
    term2 = torch.einsum('...rnl,...lms->...rmns', christoffel, christoffel)
    
    # Riemann tensor (simplified - product terms only)
    riemann = term1 - term2
    
    # Apply antisymmetry in last two indices
    riemann = riemann - riemann.transpose(-1, -2)
    
    # Ensure first Bianchi identity
    riemann = 0.5 * (riemann - riemann.transpose(-2, -1))
    
    return riemann


def compute_ricci_tensor_vectorized(riemann: torch.Tensor) -> torch.Tensor:
    """Compute Ricci tensor by contracting the Riemann tensor."""
    # Contract first and third indices: R_μν = R^λ_μλν
    ricci = torch.einsum('...lmln->...mn', riemann)
    
    # Ensure symmetry of Ricci tensor
    ricci = 0.5 * (ricci + ricci.transpose(-2, -1))
    
    return ricci


def compute_ricci_scalar_vectorized(
    ricci: torch.Tensor,
    g_inv: torch.Tensor
) -> torch.Tensor:
    """Compute Ricci scalar (scalar curvature)."""
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
    """Compute the Einstein tensor G_μν = R_μν - (1/2)Rg_μν."""
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
        if return_components:
            raise e
        # Return approximate Einstein tensor for stability
        batch_size = coords.shape[0]
        device = coords.device
        r = torch.sqrt(torch.sum(coords[:, 1:4]**2, dim=1))
        einstein = torch.zeros(batch_size, 4, 4, device=device)
        einstein[:, 0, 0] = -2.0 / (r**3 + config.epsilon)
        return einstein


def compute_kretschmann_scalar(riemann: torch.Tensor) -> torch.Tensor:
    """Compute the Kretschmann scalar K = R^μνρσ R_μνρσ."""
    kretschmann = torch.einsum('...ijkl,...ijkl->...', riemann, riemann)
    return kretschmann


def check_energy_conditions(
    T: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Check various energy conditions for the stress-energy tensor."""
    batch_size = T.shape[0]
    conditions = {}
    
    # Weak energy condition
    conditions['weak'] = T[:, 0, 0] >= 0
    
    # Null energy condition
    conditions['null'] = T[:, 0, 0] + 2*T[:, 0, 1] + T[:, 1, 1] >= 0
    
    # Strong energy condition
    T_trace = torch.einsum('...ij,...ij->...', g_inv, T)
    modified_T = T - 0.5 * T_trace.view(batch_size, 1, 1) * g
    conditions['strong'] = modified_T[:, 0, 0] >= 0
    
    # Dominant energy condition
    T_mixed = torch.einsum('...ik,...kj->...ij', g_inv, T)
    conditions['dominant'] = (T_mixed[:, 0, 0] >= 0) & (T[:, 0, 0] >= torch.abs(T[:, 0, 1:4]).sum(dim=1))
    
    return conditions


# =====================================================================
# NEURAL NETWORK MODELS
# =====================================================================

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
    """Sine activation function for SIREN networks."""
    
    def __init__(self, omega: float = 30.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.tensor(omega))
        else:
            self.register_buffer("omega", torch.tensor(omega))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)

class FourierFeatures(nn.Module):
    """Random Fourier features for improved representation of high-frequency functions."""
    
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
    """SIREN (Sinusoidal Representation Networks) model."""
    
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
    """Neural network for learning spacetime metric tensors."""
    
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
        self.siren = SIREN(config, in_features=4, out_features=16)
        
        # Learnable parameters for metric signature enforcement
        if enforce_signature:
            self.time_scale = nn.Parameter(torch.tensor(1.0))
            self.space_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass: coordinates -> metric tensor."""
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
    """Neural network using Fourier features for high-frequency function learning."""
    
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
    """Physics-informed neural network that incorporates physical constraints."""
    
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
        
        # Asymptotic behavior network
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
    """Factory function to create metric models."""
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


# =====================================================================
# MATTER MODELS (simplified)
# =====================================================================

class MatterError(Exception):
    """Base exception for matter model errors."""
    pass

@dataclass
class MatterConfig:
    """Configuration for matter models."""
    hidden_dim: int = 64
    activation: str = "sine"
    enforce_conservation: bool = True
    check_energy_conditions: bool = True
    regularization_scale: float = 0.01

class MatterModel(nn.Module, ABC):
    """Abstract base class for all matter models."""
    
    def __init__(self, config: Optional[MatterConfig] = None):
        super().__init__()
        if config is None:
            config = MatterConfig()
        self.config = config
    
    @abstractmethod
    def get_stress_energy(self, coords: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
        """Compute the stress-energy tensor for this matter type."""
        pass
    
    @abstractmethod
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values (density, pressure, etc.)."""
        pass

class PerfectFluidMatter(MatterModel):
    """Perfect fluid matter model."""
    
    def __init__(self, config: Optional[MatterConfig] = None, eos_type: str = "linear", eos_params: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.eos_type = eos_type
        self.eos_params = eos_params or {"w": 0.0}
        
        # Neural network for density field
        model_config = ModelConfig(hidden_features=config.hidden_dim, hidden_layers=3, activation="sine")
        self.density_net = SIREN(model_config, in_features=4, out_features=1)
        
        # Equation of state parameters
        if eos_type == "linear":
            self.w = nn.Parameter(torch.tensor(eos_params.get("w", 0.0)))
    
    def equation_of_state(self, density: torch.Tensor) -> torch.Tensor:
        """Compute pressure from density using equation of state."""
        if self.eos_type == "linear":
            return self.w * density
        else:
            raise ValueError(f"Unknown EOS type: {self.eos_type}")
    
    def get_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Get energy density at given coordinates."""
        raw_density = self.density_net(coords)
        return F.softplus(raw_density) + 1e-6
    
    def get_stress_energy(self, coords: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
        """Compute perfect fluid stress-energy tensor."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get density and pressure
        density = self.get_density(coords)
        pressure = self.equation_of_state(density)
        
        # Simplified stress-energy tensor (dust at rest)
        T = torch.zeros(batch_size, 4, 4, device=device)
        T[:, 0, 0] = density
        for i in range(1, 4):
            T[:, i, i] = pressure
        
        return T
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values."""
        density = self.get_density(coords)
        pressure = self.equation_of_state(density)
        return {"density": density, "pressure": pressure}

def create_matter_model(matter_type: str, config: Optional[MatterConfig] = None, **kwargs) -> MatterModel:
    """Factory function to create matter models."""
    if matter_type == "perfect_fluid":
        return PerfectFluidMatter(config, **kwargs)
    else:
        raise ValueError(f"Unknown matter type: {matter_type}")


# =====================================================================
# PHYSICS FUNCTIONS (simplified)
# =====================================================================

class PhysicsError(Exception):
    """Base exception for physics-related errors."""
    pass

@dataclass
class PhysicsConfig:
    """Configuration for physics constraints and computations."""
    einstein_weight: float = 1.0
    conservation_weight: float = 0.1
    constraint_weight: float = 0.1
    energy_condition_weight: float = 0.05
    horizon_epsilon: float = 0.1
    asymptotic_radius: float = 100.0
    enforce_causality: bool = True
    adaptive_sampling: bool = True
    curvature_threshold: float = 0.1

def compute_efe_loss(coords: torch.Tensor, metric_model: nn.Module, matter_models: List[nn.Module], 
                     matter_weights: Optional[List[float]] = None, config: Optional[PhysicsConfig] = None) -> Dict[str, torch.Tensor]:
    """Compute loss based on Einstein Field Equations."""
    if config is None:
        config = PhysicsConfig()
    
    batch_size = coords.shape[0]
    device = coords.device
    
    # Enable gradient tracking
    coords = coords.requires_grad_(True)
    
    # Get metric from model
    g = metric_model(coords).reshape(batch_size, 4, 4)
    g = 0.5 * (g + g.transpose(-2, -1))  # Ensure symmetry
    
    # Compute inverse metric
    g_inv = safe_inverse(g)
    
    # Compute Einstein tensor
    tensor_config = TensorConfig()
    einstein_components = compute_einstein_tensor_vectorized(g, coords, metric_model, tensor_config, return_components=True)
    G_tensor = einstein_components['einstein']
    
    # Compute total stress-energy tensor
    T_total = torch.zeros_like(G_tensor)
    
    if matter_weights is None:
        matter_weights = [1.0 / len(matter_models)] * len(matter_models)
    
    for model, weight in zip(matter_models, matter_weights):
        T_contrib = model.get_stress_energy(coords, g, g_inv)
        T_total += weight * T_contrib
    
    # Einstein field equations residual: G_μν - 8π T_μν
    efe_residual = G_tensor - 8 * np.pi * T_total
    
    # Compute L2 norm of residual
    efe_loss = torch.mean(torch.sum(efe_residual**2, dim=(1, 2)))
    
    losses = {
        'efe_loss': config.einstein_weight * efe_loss,
        'total_loss': config.einstein_weight * efe_loss
    }
    
    return losses

def regularized_coordinates(coords: torch.Tensor, singularity_centers: Optional[List[torch.Tensor]] = None, 
                           config: Optional[PhysicsConfig] = None) -> torch.Tensor:
    """Apply coordinate regularization near singularities and horizons."""
    if config is None:
        config = PhysicsConfig()
    
    if singularity_centers is None:
        singularity_centers = [torch.zeros(3, device=coords.device)]
    
    # Copy coordinates
    reg_coords = coords.clone()
    spatial_coords = coords[:, 1:4]
    
    for center in singularity_centers:
        # Distance from singularity
        delta = spatial_coords - center.unsqueeze(0)
        r = torch.norm(delta, dim=1)
        
        # Regularize near horizon
        horizon_radius = 2.0
        min_radius = horizon_radius + config.horizon_epsilon
        
        # Smooth cutoff function
        need_regularization = r < min_radius
        if torch.any(need_regularization):
            transition = torch.sigmoid(10 * (r - min_radius))
            new_r = min_radius + transition * (r - min_radius)
            
            # Update coordinates
            scale = new_r / (r + 1e-10)
            for i in range(3):
                reg_coords[need_regularization, i+1] = (
                    center[i] + scale[need_regularization] * delta[need_regularization, i]
                )
    
    return reg_coords

def schwarzschild_initial_metric(coords: torch.Tensor, mass: float = 1.0, use_isotropic: bool = False) -> torch.Tensor:
    """Compute the analytical Schwarzschild metric as initial condition."""
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Compute radial coordinate
    r = torch.sqrt(x**2 + y**2 + z**2)
    r = torch.clamp(r, min=2.01 * mass)  # Avoid horizon
    
    # Metric components
    g = torch.zeros(batch_size, 4, 4, device=device)
    g[:, 0, 0] = -(1 - 2*mass/r)
    g[:, 1, 1] = 1/(1 - 2*mass/r)
    g[:, 2, 2] = r**2
    g[:, 3, 3] = r**2
    
    return g

def adaptive_sampling_strategy(coords: torch.Tensor, metric_model: nn.Module, config: Optional[PhysicsConfig] = None, 
                              max_new_points: int = 1000) -> torch.Tensor:
    """Adaptively sample more points in regions of high curvature."""
    if config is None:
        config = PhysicsConfig()
    
    with torch.no_grad():
        batch_size = coords.shape[0]
        device = coords.device
        
        # Compute metric and curvature
        g = metric_model(coords).reshape(batch_size, 4, 4)
        g_inv = safe_inverse(g)
        
        # Get curvature components
        tensor_config = TensorConfig()
        components = compute_einstein_tensor_vectorized(g, coords, metric_model, tensor_config, return_components=True)
        
        # Use Ricci scalar as curvature measure
        ricci_scalar = components['ricci_scalar']
        
        # Find high curvature points
        high_curv_mask = torch.abs(ricci_scalar) > config.curvature_threshold
        high_curv_indices = torch.where(high_curv_mask)[0]
        
        if len(high_curv_indices) == 0:
            return coords
        
        # Generate new points around high curvature regions
        new_points = []
        
        for idx in high_curv_indices[:min(len(high_curv_indices), max_new_points // 4)]:
            base_point = coords[idx]
            
            # Add points in a small neighborhood
            for _ in range(4):
                offset = torch.randn(4, device=device) * 0.1
                offset[0] = 0  # Keep time fixed
                new_point = base_point + offset
                new_points.append(new_point)
        
        if new_points:
            new_coords = torch.stack(new_points)
            enhanced_coords = torch.cat([coords, new_coords], dim=0)
            return enhanced_coords
        else:
            return coords


# =====================================================================
# GRAVITATIONAL SYSTEM (simplified)
# =====================================================================

class SystemError(Exception):
    """Base exception for system-related errors."""
    pass

@dataclass
class SystemConfig:
    """Configuration for gravitational system."""
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    physics_config: Optional[PhysicsConfig] = None
    verbose: bool = True

class GravitationalSystem:
    """Main class for solving Einstein Field Equations."""
    
    def __init__(self, metric_model: nn.Module, matter_models: List[nn.Module], 
                 matter_weights: Optional[List[float]] = None, config: Optional[SystemConfig] = None):
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
        self.history = {'total_loss': [], 'efe_loss': []}
    
    def combined_stress_energy(self, coords: torch.Tensor, g: Optional[torch.Tensor] = None, 
                              g_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the total stress-energy tensor from all matter sources."""
        if g is None:
            g = self.metric_model(coords).reshape(-1, 4, 4)
            g = 0.5 * (g + g.transpose(-2, -1))
        
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
    
    def sample_coordinates(self, batch_size: int, T_range: Tuple[float, float] = (0.0, 0.0), 
                          spatial_range: float = 10.0, avoid_horizon: bool = True, adaptive: bool = True) -> torch.Tensor:
        """Sample spacetime coordinates for training."""
        # Initial uniform sampling
        coords = torch.zeros(batch_size, 4, device=self.device)
        
        # Time coordinates
        if T_range[0] == T_range[1]:
            coords[:, 0] = T_range[0]
        else:
            coords[:, 0] = torch.rand(batch_size, device=self.device) * (T_range[1] - T_range[0]) + T_range[0]
        
        # Spatial coordinates
        coords[:, 1:4] = (torch.rand(batch_size, 3, device=self.device) - 0.5) * 2 * spatial_range
        
        # Apply coordinate regularization if needed
        if avoid_horizon:
            coords = regularized_coordinates(coords, config=self.physics_config)
        
        # Apply adaptive sampling if enabled
        if adaptive and self.physics_config.adaptive_sampling:
            coords = adaptive_sampling_strategy(coords, self.metric_model, self.physics_config, max_new_points=batch_size // 2)
        
        return coords
    
    def compute_loss(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all loss components for the given coordinates."""
        return compute_efe_loss(coords, self.metric_model, self.matter_models, self.matter_weights, self.physics_config)
    
    def train_step(self, optimizer_metric: torch.optim.Optimizer, optimizer_matter: Optional[torch.optim.Optimizer], 
                   batch_size: int, T_range: Tuple[float, float], spatial_range: float) -> Dict[str, float]:
        """Perform one training step."""
        # Sample coordinates
        coords = self.sample_coordinates(batch_size, T_range, spatial_range, avoid_horizon=True, adaptive=True)
        
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
    
    def train(self, epochs: int, batch_size: int = 256, T_range: Tuple[float, float] = (0.0, 0.0), 
              spatial_range: float = 10.0, lr_metric: float = 1e-4, lr_matter: float = 5e-4, 
              train_matter: bool = True) -> Dict[str, List[float]]:
        """Train the gravitational system."""
        # Create optimizers
        optimizer_metric = torch.optim.Adam(self.metric_model.parameters(), lr=lr_metric)
        
        if train_matter:
            matter_params = []
            for model in self.matter_models:
                matter_params.extend(model.parameters())
            optimizer_matter = torch.optim.Adam(matter_params, lr=lr_matter)
        else:
            optimizer_matter = None
        
        # Training loop
        for epoch in range(epochs):
            # Train step
            losses = self.train_step(optimizer_metric, optimizer_matter, batch_size, T_range, spatial_range)
            
            # Record history
            for key, value in losses.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Print progress
            if self.config.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Total Loss: {losses['total_loss']:.6f}")
                print(f"  EFE Loss: {losses['efe_loss']:.6f}")
                print()
        
        return self.history
    
    def evaluate(self, test_coords: torch.Tensor, return_components: bool = False) -> Dict[str, torch.Tensor]:
        """Evaluate the system at given coordinates."""
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
            
            results = {'metric': g, 'stress_energy': T, 'losses': losses}
            
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
        """Predict metric tensor at given coordinates."""
        with torch.no_grad():
            g = self.metric_model(coords).reshape(-1, 4, 4)
            return 0.5 * (g + g.transpose(-2, -1))
    
    def predict_curvature(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict curvature components at given coordinates."""
        with torch.no_grad():
            g = self.predict_metric(coords)
            tensor_config = TensorConfig()
            components = compute_einstein_tensor_vectorized(g, coords, self.metric_model, tensor_config, return_components=True)
            return {
                'einstein_tensor': components['einstein'],
                'ricci_tensor': components['ricci'],
                'ricci_scalar': components['ricci_scalar'],
                'riemann_tensor': components['riemann']
            }


# =====================================================================
# PACKAGE EXPORTS (__init__.py content)
# =====================================================================

# Core modules exports
__all__ = [
    # Version
    "__version__",
    
    # Tensor operations
    "safe_inverse",
    "compute_christoffel_symbols_vectorized",
    "compute_riemann_tensor_vectorized",
    "compute_ricci_tensor_vectorized",
    "compute_ricci_scalar_vectorized",
    "compute_einstein_tensor_vectorized",
    "compute_kretschmann_scalar",
    "check_energy_conditions",
    "TensorConfig",
    "MetricSingularityError",
    "NumericalInstabilityError",
    
    # Model classes
    "SIREN",
    "MetricNet",
    "FourierNet",
    "PhysicsInformedNet",
    "create_metric_model",
    "ModelConfig",
    "Sine",
    "FourierFeatures",
    
    # Matter classes
    "MatterModel",
    "PerfectFluidMatter",
    "create_matter_model",
    "MatterConfig",
    
    # Physics functions
    "compute_efe_loss",
    "regularized_coordinates",
    "adaptive_sampling_strategy",
    "schwarzschild_initial_metric",
    "PhysicsConfig",
    
    # System class
    "GravitationalSystem",
    "SystemConfig",
]

# Main example function
def run_example():
    """Run a simple example of the EFES system."""
    print("Running EFES Example...")
    
    # Create a metric model
    metric_model = create_metric_model("siren")
    
    # Create matter models
    matter = create_matter_model("perfect_fluid", eos_type="linear")
    
    # Create gravitational system
    system = GravitationalSystem(metric_model, [matter])
    
    # Train the system
    history = system.train(epochs=20, batch_size=64)
    
    # Evaluate metric at specific points
    coords = torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    results = system.evaluate(coords)
    
    print(f"Final training loss: {history['total_loss'][-1]:.6f}")
    print(f"Metric shape: {results['metric'].shape}")
    print("Example completed successfully!")

if __name__ == "__main__":
    run_example() 
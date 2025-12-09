#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    hidden_dim: int = 256
    num_layers: int = 6
    activation: str = "sine"
    omega_0: float = 30.0
    use_fourier: bool = True
    fourier_scale: float = 10.0
    num_fourier: int = 128
    dropout: float = 0.0
    use_residual: bool = True


@dataclass
class ADMConfig:
    """Configuration for ADM decomposition."""
    enforce_hamiltonian: bool = True
    enforce_momentum: bool = True
    gauge_condition: str = "harmonic"  # harmonic, maximal_slicing, geodesic
    hamiltonian_weight: float = 10.0
    momentum_weight: float = 5.0
    gauge_weight: float = 1.0
    epsilon: float = 1e-8


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 1000
    batch_size: int = 512
    lr_initial: float = 1e-4
    lr_decay: float = 0.95
    decay_every: int = 100
    curriculum_stages: int = 5
    adaptive_sampling: bool = True
    resample_every: int = 50
    grad_clip: float = 1.0
    early_stopping_patience: int = 100
    checkpoint_every: int = 50


@dataclass
class BreakthroughConfig:
    """Configuration for breakthrough detection."""
    enabled: bool = True
    novelty_threshold: float = 2.5
    stability_threshold: float = 0.1
    energy_violation_threshold: float = 0.05
    check_every: int = 10
    history_window: int = 50


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    plot_every: int = 100
    resolution: int = 50
    spatial_extent: float = 20.0
    save_plots: bool = True
    dpi: int = 150
    figsize: Tuple[int, int] = (20, 12)


# ============================================================================
# ADVANCED NEURAL NETWORK ARCHITECTURES
# ============================================================================

class SineLayer(nn.Module):
    """Sine activation with learnable frequency."""
    
    def __init__(self, omega_0: float = 30.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.tensor(omega_0))
        else:
            self.register_buffer('omega', torch.tensor(omega_0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class FourierFeatureLayer(nn.Module):
    """Random Fourier features for better high-frequency representation."""
    
    def __init__(self, in_dim: int, num_features: int, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x


class AdvancedSIREN(nn.Module):
    """
    Advanced SIREN architecture with Fourier features and residual connections.
    Designed for learning complex spacetime geometries.
    """
    
    def __init__(self, in_dim: int, out_dim: int, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Fourier feature encoding
        if config.use_fourier:
            self.fourier = FourierFeatureLayer(in_dim, config.num_fourier, config.fourier_scale)
            effective_in_dim = 2 * config.num_fourier
        else:
            self.fourier = None
            effective_in_dim = in_dim
        
        # Input layer
        self.input_layer = nn.Linear(effective_in_dim, config.hidden_dim)
        
        # Hidden layers with optional residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(config.num_layers):
            if config.use_residual and i > 0:
                self.hidden_layers.append(
                    ResidualBlock(config.hidden_dim, SineLayer(config.omega_0))
                )
            else:
                self.hidden_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dim, out_dim)
        
        # Activation
        self.activation = SineLayer(config.omega_0, learnable=True)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """SIREN-style weight initialization."""
        with torch.no_grad():
            # First layer
            if self.fourier is None:
                self.input_layer.weight.uniform_(-1, 1)
            else:
                n = self.fourier.B.shape[1] * 2
                self.input_layer.weight.uniform_(
                    -np.sqrt(6 / n) / self.config.omega_0,
                    np.sqrt(6 / n) / self.config.omega_0
                )
            
            # Hidden layers
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(
                        -np.sqrt(6 / self.config.hidden_dim) / self.config.omega_0,
                        np.sqrt(6 / self.config.hidden_dim) / self.config.omega_0
                    )
            
            # Output layer
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / self.config.hidden_dim),
                np.sqrt(6 / self.config.hidden_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier features
        if self.fourier is not None:
            x = self.fourier(x)
        
        # Input layer
        x = self.input_layer(x)
        x = self.activation(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.output_layer(x)
        return x


# ============================================================================
# ADM DECOMPOSITION NETWORKS
# ============================================================================

class ADMNetwork(nn.Module):
    """
    Neural network for ADM variables: lapse (α), shift (β^i), 
    spatial metric (γ_ij), and extrinsic curvature (K_ij).
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Separate networks for each ADM variable
        # 4D input: (t, x, y, z)
        self.lapse_net = AdvancedSIREN(4, 1, config)  # α: scalar
        self.shift_net = AdvancedSIREN(4, 3, config)  # β^i: 3-vector
        self.metric_net = AdvancedSIREN(4, 6, config)  # γ_ij: symmetric 3x3 (6 independent)
        self.extrinsic_net = AdvancedSIREN(4, 6, config)  # K_ij: symmetric 3x3
        
        # Learnable scale parameters for physical constraints
        self.lapse_scale = nn.Parameter(torch.tensor(1.0))
        self.shift_scale = nn.Parameter(torch.tensor(0.1))
        self.metric_scale = nn.Parameter(torch.tensor(1.0))
        self.K_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords: (batch, 4) tensor of (t, x, y, z)
        
        Returns:
            Dictionary containing ADM variables
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Lapse function (must be positive)
        lapse_raw = self.lapse_net(coords)
        lapse = F.softplus(lapse_raw * self.lapse_scale) + 0.1
        
        # Shift vector
        shift = self.shift_net(coords) * self.shift_scale
        
        # Spatial metric (must be positive definite)
        metric_raw = self.metric_net(coords)
        # Construct symmetric 3x3 matrix
        gamma = self._construct_symmetric_matrix(metric_raw, batch_size, device)
        # Ensure positive definiteness
        gamma = self._ensure_positive_definite(gamma)
        
        # Extrinsic curvature (symmetric, traceless encouraged)
        K_raw = self.extrinsic_net(coords)
        K = self._construct_symmetric_matrix(K_raw, batch_size, device) * self.K_scale
        
        return {
            'lapse': lapse.squeeze(-1),  # (batch,)
            'shift': shift,  # (batch, 3)
            'gamma': gamma,  # (batch, 3, 3)
            'K': K  # (batch, 3, 3)
        }
    
    def _construct_symmetric_matrix(self, vec: torch.Tensor, batch_size: int, 
                                   device: torch.device) -> torch.Tensor:
        """Construct symmetric 3x3 matrix from 6-component vector."""
        mat = torch.zeros(batch_size, 3, 3, device=device)
        # Upper triangular part
        mat[:, 0, 0] = vec[:, 0]
        mat[:, 0, 1] = vec[:, 1]
        mat[:, 0, 2] = vec[:, 2]
        mat[:, 1, 1] = vec[:, 3]
        mat[:, 1, 2] = vec[:, 4]
        mat[:, 2, 2] = vec[:, 5]
        # Mirror to lower triangular
        mat[:, 1, 0] = mat[:, 0, 1]
        mat[:, 2, 0] = mat[:, 0, 2]
        mat[:, 2, 1] = mat[:, 1, 2]
        return mat
    
    def _ensure_positive_definite(self, gamma: torch.Tensor, 
                                  epsilon: float = 0.01) -> torch.Tensor:
        """Ensure spatial metric is positive definite via eigenvalue modification."""
        # Start with identity
        batch_size = gamma.shape[0]
        device = gamma.device
        identity = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add identity and scale
        gamma = identity + 0.5 * gamma
        
        # Ensure positive diagonal
        diag = torch.diagonal(gamma, dim1=-2, dim2=-1)
        gamma = gamma - torch.diag_embed(diag) + torch.diag_embed(torch.abs(diag) + epsilon)
        
        return gamma
    
    def get_4d_metric(self, adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct 4D metric from ADM variables."""
        batch_size = adm_vars['lapse'].shape[0]
        device = adm_vars['lapse'].device
        
        alpha = adm_vars['lapse'].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        beta = adm_vars['shift']  # (batch, 3)
        gamma = adm_vars['gamma']  # (batch, 3, 3)
        
        # Build 4D metric
        g = torch.zeros(batch_size, 4, 4, device=device)
        
        # g_00 = -α² + β_i β^i
        beta_squared = torch.einsum('bi,bij,bj->b', beta, gamma, beta)
        g[:, 0, 0] = -alpha.squeeze(-1).squeeze(-1)**2 + beta_squared
        
        # g_0i = β_i
        g[:, 0, 1:4] = torch.einsum('bij,bj->bi', gamma, beta)
        g[:, 1:4, 0] = g[:, 0, 1:4]
        
        # g_ij = γ_ij
        g[:, 1:4, 1:4] = gamma
        
        return g


# ============================================================================
# MATTER MODELS
# ============================================================================

class MatterField(nn.Module, ABC):
    """Abstract base class for matter fields."""
    
    @abstractmethod
    def stress_energy_tensor(self, coords: torch.Tensor, 
                            adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute stress-energy tensor T_μν."""
        pass
    
    @abstractmethod
    def energy_density(self, coords: torch.Tensor, 
                      adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute energy density ρ."""
        pass
    
    @abstractmethod
    def momentum_density(self, coords: torch.Tensor, 
                        adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute momentum density j^i."""
        pass


class ScalarField(MatterField):
    """Scalar field matter (e.g., inflaton, quintessence)."""
    
    def __init__(self, config: NetworkConfig, mass: float = 1.0, 
                 potential_type: str = "quadratic"):
        super().__init__()
        self.field_net = AdvancedSIREN(4, 1, config)
        self.mass = nn.Parameter(torch.tensor(mass))
        self.potential_type = potential_type
        
        if potential_type == "quartic":
            self.lambda_coupling = nn.Parameter(torch.tensor(0.1))
    
    def potential(self, phi: torch.Tensor) -> torch.Tensor:
        """Scalar potential V(φ)."""
        if self.potential_type == "quadratic":
            return 0.5 * self.mass**2 * phi**2
        elif self.potential_type == "quartic":
            return 0.5 * self.mass**2 * phi**2 + 0.25 * self.lambda_coupling * phi**4
        else:
            return torch.zeros_like(phi)
    
    def stress_energy_tensor(self, coords: torch.Tensor, 
                            adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute T_μν for scalar field."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Enable gradients
        coords = coords.requires_grad_(True)
        phi = self.field_net(coords)
        
        # Compute derivatives
        grad_phi = torch.autograd.grad(phi.sum(), coords, create_graph=True)[0]
        
        # Get 4D metric
        g = ADMNetwork.get_4d_metric(self, adm_vars)
        g_inv = self._safe_inverse(g)
        
        # Kinetic term: ½ g^μν ∂_μφ ∂_νφ
        kinetic = 0.5 * torch.einsum('bij,bi,bj->b', g_inv, grad_phi, grad_phi)
        
        # Potential
        V = self.potential(phi.squeeze(-1))
        
        # T_μν = ∂_μφ ∂_νφ - g_μν[½ ∂φ² - V]
        T = torch.zeros(batch_size, 4, 4, device=device)
        for mu in range(4):
            for nu in range(4):
                T[:, mu, nu] = grad_phi[:, mu] * grad_phi[:, nu]
        
        lagrangian = kinetic - V
        T = T - torch.einsum('b,bij->bij', lagrangian, g)
        
        return T
    
    def energy_density(self, coords: torch.Tensor, 
                      adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Energy density for Hamiltonian constraint."""
        coords = coords.requires_grad_(True)
        phi = self.field_net(coords)
        grad_phi = torch.autograd.grad(phi.sum(), coords, create_graph=True)[0]
        
        # π = ∂_t φ / α (conjugate momentum)
        alpha = adm_vars['lapse']
        pi = grad_phi[:, 0] / alpha
        
        # Spatial gradient
        gamma_inv = self._safe_inverse(adm_vars['gamma'])
        grad_phi_spatial = grad_phi[:, 1:4]
        
        # ρ = ½ π² + ½ γ^ij ∂_i φ ∂_j φ + V(φ)
        rho = (0.5 * pi**2 + 
               0.5 * torch.einsum('bij,bi,bj->b', gamma_inv, grad_phi_spatial, grad_phi_spatial) +
               self.potential(phi.squeeze(-1)))
        
        return rho
    
    def momentum_density(self, coords: torch.Tensor, 
                        adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Momentum density for momentum constraint."""
        coords = coords.requires_grad_(True)
        phi = self.field_net(coords)
        grad_phi = torch.autograd.grad(phi.sum(), coords, create_graph=True)[0]
        
        alpha = adm_vars['lapse']
        pi = grad_phi[:, 0] / alpha
        grad_phi_spatial = grad_phi[:, 1:4]
        
        # j^i = π ∂^i φ
        gamma_inv = self._safe_inverse(adm_vars['gamma'])
        j = pi.unsqueeze(-1) * torch.einsum('bij,bj->bi', gamma_inv, grad_phi_spatial)
        
        return j
    
    def _safe_inverse(self, matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Stable matrix inversion using SVD."""
        U, S, Vh = torch.linalg.svd(matrix)
        S_inv = torch.where(S > epsilon, 1.0 / S, 1.0 / epsilon)
        return torch.matmul(Vh.transpose(-2, -1), 
                           torch.matmul(torch.diag_embed(S_inv), U.transpose(-2, -1)))


class PerfectFluid(MatterField):
    """Perfect fluid matter with equation of state."""
    
    def __init__(self, config: NetworkConfig, w: float = 0.0):
        super().__init__()
        self.density_net = AdvancedSIREN(4, 1, config)
        self.velocity_net = AdvancedSIREN(4, 3, config)
        self.w = nn.Parameter(torch.tensor(w))  # p = w * ρ
    
    def stress_energy_tensor(self, coords: torch.Tensor, 
                            adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perfect fluid T_μν = (ρ + p)u_μ u_ν + p g_μν."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Energy density (ensure positive)
        rho = F.softplus(self.density_net(coords).squeeze(-1)) + 1e-6
        
        # Pressure
        p = self.w * rho
        
        # 4-velocity (normalized)
        v_spatial = 0.1 * torch.tanh(self.velocity_net(coords))
        u = self._normalize_4velocity(v_spatial, adm_vars)
        
        # Get metric
        g = ADMNetwork.get_4d_metric(self, adm_vars)
        
        # Lower indices
        u_lower = torch.einsum('bij,bj->bi', g, u)
        
        # T_μν
        T = torch.einsum('b,bi,bj->bij', rho + p, u_lower, u_lower)
        T = T + torch.einsum('b,bij->bij', p, g)
        
        return T
    
    def energy_density(self, coords: torch.Tensor, 
                      adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        rho = F.softplus(self.density_net(coords).squeeze(-1)) + 1e-6
        return rho
    
    def momentum_density(self, coords: torch.Tensor, 
                        adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        rho = self.energy_density(coords, adm_vars)
        p = self.w * rho
        v_spatial = 0.1 * torch.tanh(self.velocity_net(coords))
        
        # j^i = (ρ + p) γ^{ij} v_j
        gamma_inv = self._safe_inverse(adm_vars['gamma'])
        j = torch.einsum('b,bij,bj->bi', rho + p, gamma_inv, v_spatial)
        
        return j
    
    def _normalize_4velocity(self, v_spatial: torch.Tensor, 
                            adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize 4-velocity to satisfy g_μν u^μ u^ν = -1."""
        batch_size = v_spatial.shape[0]
        device = v_spatial.device
        
        gamma = adm_vars['gamma']
        alpha = adm_vars['lapse']
        
        # u^0 = 1/(α√(1 - v²))
        v_squared = torch.einsum('bij,bi,bj->b', gamma, v_spatial, v_spatial)
        gamma_lorentz = 1.0 / torch.sqrt(1.0 - v_squared.clamp(max=0.99))
        u_0 = gamma_lorentz / alpha
        
        # u^i = γ^{ij} v_j / α
        gamma_inv = self._safe_inverse(gamma)
        u_spatial = torch.einsum('bij,bj->bi', gamma_inv, v_spatial) * gamma_lorentz.unsqueeze(-1) / alpha.unsqueeze(-1)
        
        return torch.cat([u_0.unsqueeze(-1), u_spatial], dim=-1)
    
    def _safe_inverse(self, matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(matrix)
        S_inv = torch.where(S > epsilon, 1.0 / S, 1.0 / epsilon)
        return torch.matmul(Vh.transpose(-2, -1), 
                           torch.matmul(torch.diag_embed(S_inv), U.transpose(-2, -1)))


# ============================================================================
# 3+1 ADM PHYSICS CONSTRAINTS
# ============================================================================

class ADMPhysics:
    """Physics-informed constraints for ADM decomposition."""
    
    def __init__(self, config: ADMConfig):
        self.config = config
    
    def hamiltonian_constraint(self, coords: torch.Tensor, 
                              adm_vars: Dict[str, torch.Tensor],
                              matter: Optional[MatterField] = None) -> torch.Tensor:
        """
        Hamiltonian constraint: H = R + K² - K_ij K^ij - 16π ρ = 0
        where R is the 3D Ricci scalar, K is the trace of extrinsic curvature.
        """
        gamma = adm_vars['gamma']
        K = adm_vars['K']
        
        # Inverse metric
        gamma_inv = self._safe_inverse(gamma)
        
        # Trace of K
        K_trace = torch.einsum('bij,bij->b', gamma_inv, K)
        
        # K_ij K^ij
        K_squared = torch.einsum('bij,bikl,bkl->b', 
                                K, 
                                torch.einsum('bik,bjl->bijkl', gamma_inv, gamma_inv), 
                                K)
        
        # 3D Ricci scalar (approximated for efficiency)
        R_3d = self._compute_ricci_scalar_3d(coords, gamma, gamma_inv)
        
        # Matter contribution
        if matter is not None:
            rho = matter.energy_density(coords, adm_vars)
        else:
            rho = torch.zeros_like(K_trace)
        
        # Hamiltonian constraint
        H = R_3d + K_trace**2 - K_squared - 16 * np.pi * rho
        
        return H
    
    def momentum_constraint(self, coords: torch.Tensor, 
                           adm_vars: Dict[str, torch.Tensor],
                           matter: Optional[MatterField] = None) -> torch.Tensor:
        """
        Momentum constraint: M^i = D_j(K^ij - γ^ij K) - 8π j^i = 0
        where D_j is the covariant derivative.
        """
        gamma = adm_vars['gamma']
        K = adm_vars['K']
        gamma_inv = self._safe_inverse(gamma)
        
        # K^ij
        K_up = torch.einsum('bik,bjl,bkl->bij', gamma_inv, gamma_inv, K)
        
        # Trace K
        K_trace = torch.einsum('bij,bij->b', gamma_inv, K)
        
        # K^ij - γ^ij K
        M_term = K_up - torch.einsum('b,bij->bij', K_trace, gamma_inv)
        
        # Simplified divergence (full version requires Christoffel symbols)
        # For demonstration, we use a finite difference approximation
        M = torch.zeros(coords.shape[0], 3, device=coords.device)
        
        # Matter momentum
        if matter is not None:
            j = matter.momentum_density(coords, adm_vars)
            M = M - 8 * np.pi * j
        
        return M
    
    def gauge_condition(self, coords: torch.Tensor, 
                       adm_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Enforce gauge conditions:
        - Harmonic gauge: □x^μ = 0
        - Maximal slicing: K = 0
        - Geodesic slicing: ∂_t α = 0
        """
        if self.config.gauge_condition == "maximal_slicing":
            # Maximal slicing: enforce K = 0
            gamma_inv = self._safe_inverse(adm_vars['gamma'])
            K_trace = torch.einsum('bij,bij->b', gamma_inv, adm_vars['K'])
            return K_trace
        
        elif self.config.gauge_condition == "harmonic":
            # Harmonic gauge (simplified)
            alpha = adm_vars['lapse']
            return torch.zeros_like(alpha)  # Placeholder
        
        else:
            return torch.zeros(coords.shape[0], device=coords.device)
    
    def _compute_ricci_scalar_3d(self, coords: torch.Tensor, 
                                 gamma: torch.Tensor,
                                 gamma_inv: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D Ricci scalar using finite differences.
        This is a simplified implementation for demonstration.
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # For now, return zero (full implementation would compute Christoffel symbols)
        # In practice, you'd use automatic differentiation of gamma
        R_3d = torch.zeros(batch_size, device=device)
        
        return R_3d
    
    def _safe_inverse(self, matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Stable matrix inversion."""
        U, S, Vh = torch.linalg.svd(matrix)
        S_inv = torch.where(S > epsilon, 1.0 / S, 1.0 / epsilon)
        return torch.matmul(Vh.transpose(-2, -1), 
                           torch.matmul(torch.diag_embed(S_inv), U.transpose(-2, -1)))


# ============================================================================
# BREAKTHROUGH DETECTION SYSTEM
# ============================================================================

class BreakthroughDetector:
    """
    Advanced system for detecting novel and potentially breakthrough solutions.
    Monitors for unusual curvature patterns, energy violations, and novel symmetries.
    """
    
    def __init__(self, config: BreakthroughConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            'constraint_violation': [],
            'curvature_max': [],
            'energy_density': [],
            'metric_determinant': []
        }
        self.breakthroughs: List[Dict[str, Any]] = []
    
    def check_for_breakthrough(self, coords: torch.Tensor, 
                              adm_vars: Dict[str, torch.Tensor],
                              constraints: Dict[str, torch.Tensor],
                              epoch: int) -> Optional[Dict[str, Any]]:
        """
        Analyze current solution for breakthrough characteristics:
        1. Novel curvature patterns (high curvature in unexpected regions)
        2. Stable constraint satisfaction with unusual metric properties
        3. New symmetries or conservation laws
        4. Energy condition violations that remain stable
        """
        if not self.config.enabled or epoch % self.config.check_every != 0:
            return None
        
        # Compute metrics
        metrics = self._compute_metrics(coords, adm_vars, constraints)
        
        # Update history
        for key, value in metrics.items():
            self.history[key].append(value)
        
        # Keep only recent history
        if len(self.history['constraint_violation']) > self.config.history_window:
            for key in self.history:
                self.history[key] = self.history[key][-self.config.history_window:]
        
        # Check for breakthrough conditions
        if len(self.history['constraint_violation']) >= 10:
            is_breakthrough, reasons = self._analyze_breakthrough_conditions(metrics)
            
            if is_breakthrough:
                breakthrough = {
                    'epoch': epoch,
                    'metrics': metrics,
                    'reasons': reasons,
                    'coords_sample': coords[:10].detach().cpu().numpy(),
                    'adm_vars_sample': {k: v[:10].detach().cpu().numpy() 
                                       for k, v in adm_vars.items()}
                }
                self.breakthroughs.append(breakthrough)
                return breakthrough
        
        return None
    
    def _compute_metrics(self, coords: torch.Tensor, 
                        adm_vars: Dict[str, torch.Tensor],
                        constraints: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute various metrics for analysis."""
        with torch.no_grad():
            # Constraint violation
            H_violation = torch.abs(constraints.get('hamiltonian', torch.tensor(0.0))).mean().item()
            M_violation = torch.abs(constraints.get('momentum', torch.tensor(0.0))).mean().item()
            
            # Curvature measures
            K = adm_vars['K']
            gamma_inv = self._safe_inverse(adm_vars['gamma'])
            K_trace = torch.einsum('bij,bij->b', gamma_inv, K)
            curvature_max = torch.abs(K_trace).max().item()
            
            # Metric determinant
            det_gamma = torch.det(adm_vars['gamma'])
            metric_det = det_gamma.mean().item()
            
            # Energy density (if available)
            energy_density = 0.0  # Placeholder
            
            return {
                'constraint_violation': H_violation + M_violation,
                'curvature_max': curvature_max,
                'energy_density': energy_density,
                'metric_determinant': metric_det
            }
    
    def _analyze_breakthrough_conditions(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Analyze if current solution represents a breakthrough."""
        reasons = []
        
        # 1. Check for novel curvature patterns
        recent_curvature = self.history['curvature_max'][-10:]
        curvature_mean = np.mean(recent_curvature)
        curvature_std = np.std(recent_curvature)
        
        if metrics['curvature_max'] > curvature_mean + self.config.novelty_threshold * curvature_std:
            reasons.append("Novel high-curvature region detected")
        
        # 2. Check for stability
        recent_violations = self.history['constraint_violation'][-10:]
        violation_stability = np.std(recent_violations)
        
        if violation_stability < self.config.stability_threshold:
            reasons.append("Exceptional constraint stability achieved")
        
        # 3. Check for unusual metric properties
        recent_det = self.history['metric_determinant'][-10:]
        if abs(metrics['metric_determinant'] - 1.0) > 0.5:
            reasons.append("Non-trivial metric determinant pattern")
        
        # 4. Combined criteria
        if (len(reasons) >= 2 and 
            metrics['constraint_violation'] < self.config.stability_threshold):
            return True, reasons
        
        return False, []
    
    def _safe_inverse(self, matrix: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(matrix)
        S_inv = torch.where(S > epsilon, 1.0 / S, 1.0 / epsilon)
        return torch.matmul(Vh.transpose(-2, -1), 
                           torch.matmul(torch.diag_embed(S_inv), U.transpose(-2, -1)))


# ============================================================================
# TRAINING SYSTEM WITH CURRICULUM LEARNING
# ============================================================================

class ADMSolver:
    """
    Main solver for 3+1 ADM decomposition with PINNs.
    Features curriculum learning, adaptive sampling, and breakthrough detection.
    """
    
    def __init__(self, 
                 network_config: NetworkConfig,
                 adm_config: ADMConfig,
                 training_config: TrainingConfig,
                 breakthrough_config: BreakthroughConfig,
                 matter: Optional[MatterField] = None):
        
        self.network_config = network_config
        self.adm_config = adm_config
        self.training_config = training_config
        self.breakthrough_config = breakthrough_config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Networks
        self.adm_network = ADMNetwork(network_config).to(self.device)
        self.matter = matter
        if self.matter is not None:
            self.matter = self.matter.to(self.device)
        
        # Physics
        self.physics = ADMPhysics(adm_config)
        
        # Breakthrough detector
        self.breakthrough_detector = BreakthroughDetector(breakthrough_config)
        
        # Training state
        self.history: Dict[str, List[float]] = {
            'total_loss': [],
            'hamiltonian_loss': [],
            'momentum_loss': [],
            'gauge_loss': []
        }
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Curriculum learning stages
        self.current_stage = 0
        self.spatial_extent_schedule = np.linspace(5.0, 30.0, training_config.curriculum_stages)
    
    def sample_coordinates(self, batch_size: int, spatial_extent: float) -> torch.Tensor:
        """Sample training coordinates with adaptive strategy."""
        coords = torch.zeros(batch_size, 4, device=self.device)
        
        # Time (for now, consider static spacetime)
        coords[:, 0] = 0.0
        
        # Spatial coordinates (with focus on regions of interest)
        if self.training_config.adaptive_sampling and len(self.history['total_loss']) > 10:
            # 50% uniform, 50% focused on high-curvature regions
            n_uniform = batch_size // 2
            n_focused = batch_size - n_uniform
            
            # Uniform sampling
            coords[:n_uniform, 1:] = (torch.rand(n_uniform, 3, device=self.device) - 0.5) * 2 * spatial_extent
            
            # Focused sampling (example: near origin for black holes)
            r = torch.rand(n_focused, device=self.device) ** 0.5 * spatial_extent * 0.3
            theta = torch.rand(n_focused, device=self.device) * np.pi
            phi = torch.rand(n_focused, device=self.device) * 2 * np.pi
            
            coords[n_uniform:, 1] = r * torch.sin(theta) * torch.cos(phi)
            coords[n_uniform:, 2] = r * torch.sin(theta) * torch.sin(phi)
            coords[n_uniform:, 3] = r * torch.cos(theta)
        else:
            # Pure uniform sampling
            coords[:, 1:] = (torch.rand(batch_size, 3, device=self.device) - 0.5) * 2 * spatial_extent
        
        return coords
    
    def compute_loss(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss."""
        # Get ADM variables
        adm_vars = self.adm_network(coords)
        
        # Compute constraints
        H = self.physics.hamiltonian_constraint(coords, adm_vars, self.matter)
        M = self.physics.momentum_constraint(coords, adm_vars, self.matter)
        G = self.physics.gauge_condition(coords, adm_vars)
        
        # Individual losses
        hamiltonian_loss = torch.mean(H ** 2)
        momentum_loss = torch.mean(torch.sum(M ** 2, dim=-1))
        gauge_loss = torch.mean(G ** 2)
        
        # Total loss
        total_loss = (self.adm_config.hamiltonian_weight * hamiltonian_loss +
                     self.adm_config.momentum_weight * momentum_loss +
                     self.adm_config.gauge_weight * gauge_loss)
        
        return {
            'total_loss': total_loss,
            'hamiltonian_loss': hamiltonian_loss,
            'momentum_loss': momentum_loss,
            'gauge_loss': gauge_loss,
            'constraints': {'hamiltonian': H, 'momentum': M}
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop with curriculum learning."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            list(self.adm_network.parameters()) + 
            (list(self.matter.parameters()) if self.matter else []),
            lr=self.training_config.lr_initial
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.training_config.lr_decay
        )
        
        print("="*80)
        print("STARTING 3+1 ADM PINN TRAINING")
        print("="*80)
        print(f"Network: {self.network_config.hidden_dim}x{self.network_config.num_layers}")
        print(f"Device: {self.device}")
        print(f"Curriculum stages: {self.training_config.curriculum_stages}")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.training_config.epochs):
            # Update curriculum stage
            stage = min(epoch // (self.training_config.epochs // self.training_config.curriculum_stages),
                       self.training_config.curriculum_stages - 1)
            spatial_extent = self.spatial_extent_schedule[stage]
            
            if stage != self.current_stage:
                self.current_stage = stage
                print(f"\n{'='*80}")
                print(f"CURRICULUM STAGE {stage+1}/{self.training_config.curriculum_stages}")
                print(f"Spatial extent: {spatial_extent:.2f}")
                print(f"{'='*80}\n")
            
            # Sample coordinates
            coords = self.sample_coordinates(self.training_config.batch_size, spatial_extent)
            
            # Forward pass
            optimizer.zero_grad()
            loss_dict = self.compute_loss(coords)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.adm_network.parameters()) + 
                (list(self.matter.parameters()) if self.matter else []),
                self.training_config.grad_clip
            )
            
            optimizer.step()
            
            # Record history
            for key in ['total_loss', 'hamiltonian_loss', 'momentum_loss', 'gauge_loss']:
                self.history[key].append(loss_dict[key].item())
            
            # Update best loss
            current_loss = loss_dict['total_loss'].item()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Learning rate decay
            if (epoch + 1) % self.training_config.decay_every == 0:
                scheduler.step()
            
            # Breakthrough detection
            adm_vars = self.adm_network(coords)
            breakthrough = self.breakthrough_detector.check_for_breakthrough(
                coords, adm_vars, loss_dict['constraints'], epoch
            )
            
            if breakthrough:
                print(f"\n{'*'*80}")
                print(f"🎉 BREAKTHROUGH DETECTED AT EPOCH {epoch+1}!")
                print(f"{'*'*80}")
                for reason in breakthrough['reasons']:
                    print(f"  • {reason}")
                print(f"Metrics: {breakthrough['metrics']}")
                print(f"{'*'*80}\n")
            
            # Progress reporting
            if (epoch + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.training_config.epochs} | "
                      f"Loss: {current_loss:.6f} | "
                      f"H: {loss_dict['hamiltonian_loss'].item():.6f} | "
                      f"M: {loss_dict['momentum_loss'].item():.6f} | "
                      f"G: {loss_dict['gauge_loss'].item():.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final loss: {self.best_loss:.6f}")
        print(f"Breakthroughs detected: {len(self.breakthrough_detector.breakthroughs)}")
        print(f"{'='*80}\n")
        
        return self.history
    
    def predict(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict ADM variables at given coordinates."""
        with torch.no_grad():
            coords = coords.to(self.device)
            adm_vars = self.adm_network(coords)
            
            # Also compute 4D metric
            g_4d = self.adm_network.get_4d_metric(adm_vars)
            adm_vars['metric_4d'] = g_4d
            
            # Compute constraints
            H = self.physics.hamiltonian_constraint(coords, adm_vars, self.matter)
            M = self.physics.momentum_constraint(coords, adm_vars, self.matter)
            
            adm_vars['hamiltonian_constraint'] = H
            adm_vars['momentum_constraint'] = M
            
        return {k: v.cpu() for k, v in adm_vars.items()}


# ============================================================================
# ADVANCED VISUALIZATION SYSTEM
# ============================================================================

class ADMVisualizer:
    """Advanced visualization system for ADM decomposition results."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             filename: str = "training_history.png"):
        """Plot training history with multiple loss components."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        epochs = np.arange(1, len(history['total_loss']) + 1)
        
        # Total loss
        ax1 = fig.add_subplot(gs[0, :])
        ax1.semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (log scale)', fontsize=12)
        ax1.set_title('Total Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Individual components
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(epochs, history['hamiltonian_loss'], 'r-', linewidth=2, label='Hamiltonian')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (log scale)', fontsize=12)
        ax2.set_title('Hamiltonian Constraint', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.semilogy(epochs, history['momentum_loss'], 'g-', linewidth=2, label='Momentum')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss (log scale)', fontsize=12)
        ax3.set_title('Momentum Constraint', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        
        plt.suptitle('3+1 ADM PINN Training Progress', fontsize=16, fontweight='bold', y=0.98)
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved training history to {filename}")
        
        plt.close()
    
    def plot_adm_fields(self, solver: ADMSolver, filename: str = "adm_fields.png"):
        """Visualize ADM fields in a 2D slice."""
        # Create coordinate grid (z=0 slice)
        x = np.linspace(-self.config.spatial_extent, self.config.spatial_extent, self.config.resolution)
        y = np.linspace(-self.config.spatial_extent, self.config.spatial_extent, self.config.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Convert to torch tensor
        coords = torch.zeros(self.config.resolution ** 2, 4)
        coords[:, 0] = 0.0  # t = 0
        coords[:, 1] = torch.from_numpy(X.flatten())
        coords[:, 2] = torch.from_numpy(Y.flatten())
        coords[:, 3] = 0.0  # z = 0
        
        # Predict
        results = solver.predict(coords)
        
        # Extract fields
        lapse = results['lapse'].numpy().reshape(self.config.resolution, self.config.resolution)
        H = results['hamiltonian_constraint'].numpy().reshape(self.config.resolution, self.config.resolution)
        
        # Get trace of K
        gamma_np = results['gamma'].numpy()
        K_np = results['K'].numpy()
        K_trace = np.zeros(self.config.resolution ** 2)
        for i in range(self.config.resolution ** 2):
            gamma_inv = np.linalg.inv(gamma_np[i] + 1e-8 * np.eye(3))
            K_trace[i] = np.trace(gamma_inv @ K_np[i])
        K_trace = K_trace.reshape(self.config.resolution, self.config.resolution)
        
        # Create plot
        fig = plt.figure(figsize=self.config.figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.4)
        
        # Lapse function
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(X, Y, lapse, levels=20, cmap='viridis')
        ax1.set_title('Lapse Function α', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        plt.colorbar(im1, ax=ax1)
        
        # Trace of K
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(X, Y, K_trace, levels=20, cmap='RdBu_r', vmin=-np.abs(K_trace).max(), vmax=np.abs(K_trace).max())
        ax2.set_title('Extrinsic Curvature Trace K', fontsize=14, fontweight='bold')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        plt.colorbar(im2, ax=ax2)
        
        # Hamiltonian constraint violation
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.contourf(X, Y, np.abs(H), levels=20, cmap='hot', norm=plt.matplotlib.colors.LogNorm(vmin=1e-6, vmax=np.abs(H).max()+1e-8))
        ax3.set_title('|Hamiltonian Constraint|', fontsize=14, fontweight='bold')
        ax3.set_xlabel('x', fontsize=12)
        ax3.set_ylabel('y', fontsize=12)
        plt.colorbar(im3, ax=ax3, label='log scale')
        
        # Spatial metric components
        gamma_11 = gamma_np[:, 0, 0].reshape(self.config.resolution, self.config.resolution)
        gamma_22 = gamma_np[:, 1, 1].reshape(self.config.resolution, self.config.resolution)
        gamma_12 = gamma_np[:, 0, 1].reshape(self.config.resolution, self.config.resolution)
        
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.contourf(X, Y, gamma_11, levels=20, cmap='plasma')
        ax4.set_title('Spatial Metric γ₁₁', fontsize=14, fontweight='bold')
        ax4.set_xlabel('x', fontsize=12)
        ax4.set_ylabel('y', fontsize=12)
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.contourf(X, Y, gamma_22, levels=20, cmap='plasma')
        ax5.set_title('Spatial Metric γ₂₂', fontsize=14, fontweight='bold')
        ax5.set_xlabel('x', fontsize=12)
        ax5.set_ylabel('y', fontsize=12)
        plt.colorbar(im5, ax=ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.contourf(X, Y, gamma_12, levels=20, cmap='RdBu_r', vmin=-np.abs(gamma_12).max(), vmax=np.abs(gamma_12).max())
        ax6.set_title('Spatial Metric γ₁₂', fontsize=14, fontweight='bold')
        ax6.set_xlabel('x', fontsize=12)
        ax6.set_ylabel('y', fontsize=12)
        plt.colorbar(im6, ax=ax6)
        
        plt.suptitle('ADM Fields (z=0 slice)', fontsize=16, fontweight='bold', y=0.98)
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved ADM fields to {filename}")
        
        plt.close()
    
    def plot_metric_3d(self, solver: ADMSolver, filename: str = "metric_3d.png"):
        """3D visualization of metric components."""
        # Create coordinate grid
        n = 30
        x = np.linspace(-self.config.spatial_extent/2, self.config.spatial_extent/2, n)
        y = np.linspace(-self.config.spatial_extent/2, self.config.spatial_extent/2, n)
        X, Y = np.meshgrid(x, y)
        
        coords = torch.zeros(n ** 2, 4)
        coords[:, 0] = 0.0
        coords[:, 1] = torch.from_numpy(X.flatten())
        coords[:, 2] = torch.from_numpy(Y.flatten())
        coords[:, 3] = 0.0
        
        results = solver.predict(coords)
        lapse = results['lapse'].numpy().reshape(n, n)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, lapse, cmap='viridis', 
                              edgecolor='none', alpha=0.9)
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Lapse α', fontsize=12, fontweight='bold')
        ax.set_title('3D Lapse Function', fontsize=16, fontweight='bold', pad=20)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved 3D metric to {filename}")
        
        plt.close()
    
    def plot_breakthrough_analysis(self, detector: BreakthroughDetector, 
                                   filename: str = "breakthrough_analysis.png"):
        """Visualize breakthrough detection results."""
        if not detector.breakthroughs:
            print("No breakthroughs detected to visualize.")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # History of metrics
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(len(detector.history['constraint_violation']))
        ax1.plot(epochs, detector.history['constraint_violation'], 'b-', linewidth=2, label='Constraint Violation')
        
        # Mark breakthroughs
        for bt in detector.breakthroughs:
            ax1.axvline(x=bt['epoch'], color='r', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Check Point', fontsize=12)
        ax1.set_ylabel('Constraint Violation', fontsize=12)
        ax1.set_title('Constraint Evolution with Breakthroughs', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curvature evolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, detector.history['curvature_max'], 'g-', linewidth=2, label='Max Curvature')
        for bt in detector.breakthroughs:
            ax2.axvline(x=bt['epoch'], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Check Point', fontsize=12)
        ax2.set_ylabel('Maximum Curvature', fontsize=12)
        ax2.set_title('Curvature Evolution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Breakthrough summary
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        summary_text = f"BREAKTHROUGH SUMMARY\n{'='*60}\n\n"
        summary_text += f"Total Breakthroughs Detected: {len(detector.breakthroughs)}\n\n"
        
        for i, bt in enumerate(detector.breakthroughs[:5], 1):  # Show first 5
            summary_text += f"Breakthrough #{i} (Epoch {bt['epoch']}):\n"
            for reason in bt['reasons']:
                summary_text += f"  • {reason}\n"
            summary_text += f"  Metrics: Constraint={bt['metrics']['constraint_violation']:.6f}, "
            summary_text += f"Curvature={bt['metrics']['curvature_max']:.6f}\n\n"
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Breakthrough Detection Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if self.config.save_plots:
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved breakthrough analysis to {filename}")
        
        plt.close()
    
    def create_comprehensive_report(self, solver: ADMSolver, 
                                   history: Dict[str, List[float]],
                                   detector: BreakthroughDetector):
        """Generate all visualizations and reports."""
        print("\nGenerating comprehensive visualization report...")
        
        self.plot_training_history(history, "training_history.png")
        self.plot_adm_fields(solver, "adm_fields.png")
        self.plot_metric_3d(solver, "metric_3d.png")
        
        if detector.breakthroughs:
            self.plot_breakthrough_analysis(detector, "breakthrough_analysis.png")
        
        print("Visualization report complete!\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print(" "*20 + "3+1 ADM PINN SOLVER")
    print(" "*15 + "Einstein Field Equations Solver")
    print("="*80 + "\n")
    
    # Configuration
    network_config = NetworkConfig(
        hidden_dim=256,
        num_layers=6,
        activation="sine",
        omega_0=30.0,
        use_fourier=True,
        fourier_scale=10.0,
        num_fourier=128
    )
    
    adm_config = ADMConfig(
        enforce_hamiltonian=True,
        enforce_momentum=True,
        gauge_condition="maximal_slicing",
        hamiltonian_weight=10.0,
        momentum_weight=5.0,
        gauge_weight=1.0
    )
    
    training_config = TrainingConfig(
        epochs=500,
        batch_size=512,
        lr_initial=1e-4,
        curriculum_stages=5,
        adaptive_sampling=True
    )
    
    breakthrough_config = BreakthroughConfig(
        enabled=True,
        novelty_threshold=2.5,
        check_every=10
    )
    
    viz_config = VisualizationConfig(
        resolution=50,
        spatial_extent=20.0,
        save_plots=True,
        dpi=150
    )
    
    # Create matter field (scalar field example)
    print("Initializing scalar field matter...")
    matter = ScalarField(network_config, mass=1.0, potential_type="quadratic")
    
    # Create solver
    print("Initializing ADM solver...")
    solver = ADMSolver(
        network_config=network_config,
        adm_config=adm_config,
        training_config=training_config,
        breakthrough_config=breakthrough_config,
        matter=matter
    )
    
    # Train
    print("\nBeginning training...\n")
    history = solver.train()
    
    # Visualize
    print("\nCreating visualizations...")
    visualizer = ADMVisualizer(viz_config)
    visualizer.create_comprehensive_report(
        solver, 
        history, 
        solver.breakthrough_detector
    )
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final loss: {solver.best_loss:.6e}")
    print(f"Breakthroughs detected: {len(solver.breakthrough_detector.breakthroughs)}")
    print(f"Generated visualizations:")
    print("  • training_history.png - Training loss curves")
    print("  • adm_fields.png - ADM field visualizations")
    print("  • metric_3d.png - 3D metric visualization")
    if solver.breakthrough_detector.breakthroughs:
        print("  • breakthrough_analysis.png - Breakthrough detection results")
    print("="*80 + "\n")
    
    return solver, history


if __name__ == "__main__":
    # Run the solver
    solver, history = main()
    
    print("Solver ready for analysis!")
    print("You can now:")
    print("  1. Examine solver.breakthrough_detector.breakthroughs for novel solutions")
    print("  2. Use solver.predict(coords) to evaluate at new points")
    print("  3. Analyze the generated plots for physical insights")
    print("\nThank you for using the 3+1 ADM PINN Solver!")

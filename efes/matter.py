"""
Matter models for Einstein Field Equations Solver.

This module implements various matter and energy sources that contribute
to the stress-energy tensor in Einstein's field equations, including:
- Perfect fluids
- Scalar fields
- Electromagnetic fields
- Dark matter and dark energy

Each matter model computes its contribution to the stress-energy tensor
T_μν, which appears on the right-hand side of Einstein's equations.

Physics Note:
-------------
The stress-energy tensor T_μν encodes the density and flux of energy
and momentum in spacetime. It must satisfy:
1. Symmetry: T_μν = T_νμ
2. Conservation: ∇_μ T^μν = 0
3. Energy conditions (for physical matter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .models import SIREN, ModelConfig
from .tensor_ops import safe_inverse


class MatterError(Exception):
    """Base exception for matter model errors."""
    pass


class ConservationViolationError(MatterError):
    """Raised when stress-energy conservation is violated."""
    pass


class EnergyConditionError(MatterError):
    """Raised when energy conditions are violated."""
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
    """
    Abstract base class for all matter models.
    
    Each matter model must implement:
    1. get_stress_energy: Compute the stress-energy tensor
    2. get_field_values: Return physical field values
    3. compute_conservation: Check conservation laws
    """
    
    def __init__(self, config: Optional[MatterConfig] = None):
        super().__init__()
        
        if config is None:
            config = MatterConfig()
        
        self.config = config
        
        # Common components
        if config.activation == "sine":
            from .models import Sine
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
        """
        Compute the stress-energy tensor for this matter type.
        
        Args:
            coords: Spacetime coordinates [batch_size, 4]
            g: Metric tensor [batch_size, 4, 4]
            g_inv: Inverse metric [batch_size, 4, 4]
            
        Returns:
            Stress-energy tensor [batch_size, 4, 4]
        """
        pass
    
    @abstractmethod
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get physical field values (density, pressure, etc.).
        
        Args:
            coords: Spacetime coordinates [batch_size, 4]
            
        Returns:
            Dictionary of field values
        """
        pass
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conservation equation ∇_μ T^μν = 0.
        
        This is a fundamental requirement for any physical stress-energy tensor,
        following from the contracted Bianchi identity and Einstein's equations.
        
        Args:
            coords: Spacetime coordinates [batch_size, 4]
            g: Metric tensor [batch_size, 4, 4]
            g_inv: Inverse metric [batch_size, 4, 4]
            
        Returns:
            Conservation violation [batch_size, 4]
            
        Physics Note:
        -------------
        Conservation of stress-energy is automatic in GR when the matter
        equations of motion are satisfied. Violations indicate either
        numerical errors or unphysical matter configurations.
        """
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
    """
    Perfect fluid matter model.
    
    A perfect fluid is characterized by:
    - Energy density ρ
    - Pressure p
    - Four-velocity u^μ (normalized: g_μν u^μ u^ν = -1)
    
    The stress-energy tensor is:
    T_μν = (ρ + p) u_μ u_ν + p g_μν
    
    Physics Applications:
    --------------------
    - Cosmological models (radiation, matter, dark energy)
    - Stellar interiors
    - Accretion disks
    - Early universe physics
    """
    
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
            self.w = nn.Parameter(torch.tensor(eos_params.get("w", 0.0)))
        elif eos_type == "polytropic":
            # p = K * ρ^Γ
            self.K = nn.Parameter(torch.tensor(eos_params.get("K", 1.0)))
            self.Gamma = nn.Parameter(torch.tensor(eos_params.get("Gamma", 5/3)))
    
    def equation_of_state(self, density: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure from density using equation of state.
        
        Physics Note:
        -------------
        Different values of w in p = wρ correspond to:
        - w = 0: Dust (non-relativistic matter)
        - w = 1/3: Radiation
        - w = -1: Cosmological constant
        - -1 < w < -1/3: Quintessence
        """
        if self.eos_type == "linear":
            return self.w * density
        elif self.eos_type == "polytropic":
            return self.K * torch.pow(density, self.Gamma)
        else:
            raise ValueError(f"Unknown EOS type: {self.eos_type}")
    
    def get_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Get energy density at given coordinates."""
        # Ensure positive density
        raw_density = self.density_net(coords)
        return F.softplus(raw_density) + 1e-6
    
    def get_four_velocity(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Get normalized four-velocity.
        
        The four-velocity must satisfy g_μν u^μ u^ν = -1.
        """
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
        """
        Compute perfect fluid stress-energy tensor.
        
        T_μν = (ρ + p) u_μ u_ν + p g_μν
        """
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
        """Get physical field values."""
        density = self.get_density(coords)
        pressure = self.equation_of_state(density)
        
        return {
            "density": density,
            "pressure": pressure,
            "equation_of_state_w": self.w if hasattr(self, "w") else None
        }


class ScalarFieldMatter(MatterModel):
    """
    Scalar field matter model.
    
    Scalar fields are fundamental in:
    - Inflation models
    - Dark energy (quintessence)
    - Higgs mechanism
    - Axion dark matter
    
    The stress-energy tensor for a scalar field φ with potential V(φ) is:
    T_μν = ∂_μφ ∂_νφ - g_μν [½ g^αβ ∂_αφ ∂_βφ + V(φ)]
    """
    
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
        """
        Compute the scalar field potential V(φ).
        
        Physics Note:
        -------------
        The choice of potential determines the dynamics:
        - Quadratic: Simple massive scalar field
        - Quartic: Self-interacting field (φ⁴ theory)
        - Exponential: Quintessence dark energy models
        """
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
        """Get scalar field value at given coordinates."""
        return self.field_net(coords)
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scalar field stress-energy tensor.
        
        T_μν = ∂_μφ ∂_νφ - g_μν [½ g^αβ ∂_αφ ∂_βφ + V(φ)]
        """
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
        """Get physical field values."""
        phi = self.get_field(coords)
        V = self.potential(phi)
        
        return {
            "field": phi,
            "potential": V,
            "mass": self.mass if hasattr(self, "mass") else None
        }


class ElectromagneticFieldMatter(MatterModel):
    """
    Electromagnetic field matter model.
    
    The electromagnetic field is described by the field tensor F_μν and
    contributes to spacetime curvature through its stress-energy tensor:
    
    T_μν = 1/μ₀ [F_μα F_ν^α - ¼ g_μν F_αβ F^αβ]
    
    Physics Applications:
    --------------------
    - Charged black holes (Reissner-Nordström)
    - Magnetized neutron stars
    - Cosmological magnetic fields
    - Plasma dynamics in curved spacetime
    """
    
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
        """Get electromagnetic four-potential A_μ."""
        return self.potential_net(coords)
    
    def get_field_tensor(
        self, 
        coords: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute electromagnetic field tensor F_μν = ∂_μ A_ν - ∂_ν A_μ.
        
        Physics Note:
        -------------
        The field tensor is antisymmetric and gauge-invariant. Its components
        encode the electric and magnetic fields in the given reference frame.
        """
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
        """
        Compute electromagnetic stress-energy tensor.
        
        T_μν = 1/μ₀ [F_μα F_ν^α - ¼ g_μν F_αβ F^αβ]
        """
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
        """Get physical field values."""
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
    """
    Dark matter and dark energy model.
    
    This model can represent:
    - Cold dark matter (pressureless fluid)
    - Warm/hot dark matter
    - Dark energy (cosmological constant or dynamic)
    - Interacting dark sector
    
    Physics Note:
    -------------
    Dark matter and dark energy constitute ~95% of the universe's energy
    content. While their fundamental nature is unknown, their gravitational
    effects are well-described by their stress-energy contributions.
    """
    
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
        """Get dark matter density."""
        raw_density = self.dm_density_net(coords)
        return F.softplus(raw_density) + 1e-6
    
    def get_de_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Get dark energy density."""
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
        """
        Compute dark sector stress-energy tensor.
        
        For dark matter: T_μν = ρ_dm u_μ u_ν (dust)
        For dark energy: T_μν = -ρ_de g_μν (cosmological constant)
        """
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
        """Get physical field values."""
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
    """
    Factory function to create matter models.
    
    Args:
        matter_type: Type of matter ("perfect_fluid", "scalar_field", etc.)
        config: Matter configuration
        **kwargs: Additional arguments for specific models
        
    Returns:
        Matter model instance
        
    Raises:
        ValueError: If matter_type is not recognized
    """
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
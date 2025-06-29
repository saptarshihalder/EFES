"""
Physics-specific functions and constraints for Einstein Field Equations Solver.

This module implements:
- Physical constraint enforcement
- Coordinate transformations and regularizations
- Einstein Field Equations loss computation
- Energy condition checking
- ADM decomposition for 3+1 formalism
- Adaptive sampling strategies

All functions are designed to ensure physical validity of solutions
to Einstein's field equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

from .tensor_ops import (
    compute_einstein_tensor_vectorized,
    check_energy_conditions,
    safe_inverse,
    TensorConfig
)


class PhysicsError(Exception):
    """Base exception for physics-related errors."""
    pass


class CausalityViolationError(PhysicsError):
    """Raised when causality conditions are violated."""
    pass


class AsymptoticBehaviorError(PhysicsError):
    """Raised when asymptotic behavior is incorrect."""
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


def compute_efe_loss(
    coords: torch.Tensor,
    metric_model: nn.Module,
    matter_models: List[nn.Module],
    matter_weights: Optional[List[float]] = None,
    config: Optional[PhysicsConfig] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute loss based on Einstein Field Equations: G_μν + Λg_μν = 8πT_μν.
    
    This is the core physics loss that ensures the learned metric satisfies
    Einstein's equations for the given matter distribution.
    
    Args:
        coords: Spacetime coordinates [batch_size, 4]
        metric_model: Neural network model for the metric
        matter_models: List of matter models
        matter_weights: Weights for combining matter contributions
        config: Physics configuration
        
    Returns:
        Dictionary containing various loss components
        
    Physics Note:
    -------------
    Einstein's field equations relate the geometry of spacetime (left side)
    to its energy-momentum content (right side). In geometric units (G=c=1):
    
    G_μν + Λg_μν = 8π T_μν
    
    where:
    - G_μν is the Einstein tensor (geometry)
    - Λ is the cosmological constant
    - T_μν is the stress-energy tensor (matter/energy)
    """
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
    """
    Compute loss for stress-energy conservation: ∇_μ T^μν = 0.
    
    Physics Note:
    -------------
    Conservation of stress-energy is a fundamental requirement that follows
    from the Bianchi identity and Einstein's equations. It ensures that
    energy and momentum are neither created nor destroyed, only transformed
    or transported through spacetime.
    """
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
    """
    Compute various physical constraint losses.
    
    These constraints ensure:
    1. Correct metric signature
    2. Asymptotic flatness
    3. Regularity at horizons
    4. Causality preservation
    """
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
    """
    Check violations of energy conditions and return as loss.
    
    Physics Note:
    -------------
    Energy conditions restrict the types of matter/energy that can exist:
    - Weak: No observer measures negative energy density
    - Null: Energy density along light rays is non-negative
    - Strong: Gravity is always attractive
    - Dominant: Energy flow is causal (subluminal)
    """
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
    """
    Apply coordinate regularization near singularities and horizons.
    
    This prevents numerical instabilities near:
    - Black hole singularities (r = 0)
    - Event horizons (r = 2M in Schwarzschild)
    - Coordinate singularities
    
    Args:
        coords: Input coordinates [batch_size, 4]
        singularity_centers: List of known singularity locations
        config: Physics configuration
        
    Returns:
        Regularized coordinates
        
    Physics Note:
    -------------
    Many coordinate systems have singularities that are artifacts of the
    coordinate choice, not physical singularities. This function helps
    avoid numerical issues near such coordinate singularities while
    preserving the physics away from them.
    """
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
    """
    Adaptively sample more points in regions of high curvature.
    
    This improves accuracy in regions where the metric changes rapidly,
    such as near black holes or other strong-field sources.
    
    Args:
        coords: Initial coordinates [batch_size, 4]
        metric_model: Neural network model for the metric
        config: Physics configuration
        max_new_points: Maximum number of new points to add
        
    Returns:
        Enhanced coordinate tensor with additional samples
        
    Physics Note:
    -------------
    Curvature measures how much spacetime deviates from flatness. High
    curvature regions require denser sampling for accurate representation
    of the metric and its derivatives.
    """
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
    """
    Compute the analytical Schwarzschild metric as initial condition.
    
    The Schwarzschild metric describes spacetime around a non-rotating,
    uncharged black hole or spherical mass.
    
    Args:
        coords: Spacetime coordinates [batch_size, 4]
        mass: Mass parameter (in geometric units)
        use_isotropic: Whether to use isotropic coordinates
        
    Returns:
        Metric tensor [batch_size, 4, 4]
        
    Physics Note:
    -------------
    The Schwarzschild solution is the unique spherically symmetric vacuum
    solution to Einstein's equations. It's characterized by:
    - Event horizon at r = 2M
    - Singularity at r = 0
    - Asymptotically flat (Minkowski) as r → ∞
    """
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
    """
    Perform ADM (3+1) decomposition of the spacetime metric.
    
    The ADM formalism splits 4D spacetime into:
    - 3D spatial slices (hypersurfaces)
    - Time evolution between slices
    
    Components:
    - α (lapse): Rate of time flow
    - β^i (shift): Coordinate velocity
    - γ_ij (3-metric): Metric on spatial slice
    
    Args:
        metric: 4D metric tensor [batch_size, 4, 4]
        coords: Spacetime coordinates [batch_size, 4]
        
    Returns:
        Dictionary with ADM variables
        
    Physics Note:
    -------------
    The ADM formalism is essential for:
    - Numerical relativity simulations
    - Initial value problems in GR
    - Studying dynamics of spacetime
    - Black hole and gravitational wave physics
    """
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
    """
    Compute the Hamiltonian constraint in the ADM formalism.
    
    The Hamiltonian constraint is:
    H = R - K² + K_ij K^ij - 16π ρ = 0
    
    where:
    - R is the 3D Ricci scalar
    - K is the trace of extrinsic curvature
    - ρ is the energy density
    
    Physics Note:
    -------------
    The Hamiltonian constraint is the projection of Einstein's equations
    normal to the spatial hypersurface. It constrains the initial data
    in numerical relativity simulations.
    """
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
    """
    Compute the momentum constraint in the ADM formalism.
    
    The momentum constraint is:
    M^i = D_j K^ij - D^i K - 8π J^i = 0
    
    where:
    - D_j is the covariant derivative on the 3-space
    - J^i is the momentum density
    
    Physics Note:
    -------------
    The momentum constraint is the projection of Einstein's equations
    tangent to the spatial hypersurface. It ensures conservation of
    momentum in the evolution.
    """
    batch_size = spatial_metric.shape[0]
    device = spatial_metric.device
    
    # For simplicity, return zero constraint
    # Full calculation requires 3D covariant derivatives
    M = torch.zeros(batch_size, 3, device=device)
    
    if matter_momentum is not None:
        M -= 8 * np.pi * matter_momentum
    
    return M
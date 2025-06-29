"""
Gravitational system class for combining metric and matter models.

This module provides the main interface for setting up and training
Einstein Field Equations solvers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .physics import (
    compute_efe_loss,
    regularized_coordinates,
    adaptive_sampling_strategy,
    PhysicsConfig
)
from .tensor_ops import safe_inverse


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
    """
    Main class for solving Einstein Field Equations.
    
    This class combines:
    - A neural network model for the spacetime metric
    - One or more matter models contributing to stress-energy
    - Training procedures that enforce Einstein's equations
    - Analysis and visualization tools
    
    Example:
    --------
    ```python
    # Create metric model
    metric_model = create_metric_model("siren")
    
    # Create matter models
    matter1 = create_matter_model("perfect_fluid", eos_type="dust")
    matter2 = create_matter_model("scalar_field", potential_type="quadratic")
    
    # Create system
    system = GravitationalSystem(
        metric_model=metric_model,
        matter_models=[matter1, matter2],
        matter_weights=[0.7, 0.3]  # 70% dust, 30% scalar field
    )
    
    # Train
    history = system.train(epochs=1000, batch_size=512)
    ```
    
    Physics Note:
    -------------
    The system solves the coupled Einstein-matter equations:
    - Geometry side: G_μν + Λg_μν (from the metric model)
    - Matter side: 8πT_μν (from matter models)
    The training process finds a metric that satisfies G_μν + Λg_μν = 8πT_μν
    """
    
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
        """
        Compute the total stress-energy tensor from all matter sources.
        
        Args:
            coords: Spacetime coordinates
            g: Metric tensor (computed if not provided)
            g_inv: Inverse metric (computed if not provided)
            
        Returns:
            Total stress-energy tensor
        """
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
        """
        Sample spacetime coordinates for training.
        
        Args:
            batch_size: Number of points to sample
            T_range: Time coordinate range
            spatial_range: Spatial coordinate range [-L, L]³
            avoid_horizon: Whether to avoid sampling near horizons
            adaptive: Whether to use adaptive sampling
            
        Returns:
            Sampled coordinates [batch_size, 4]
        """
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
        """
        Compute all loss components for the given coordinates.
        
        Args:
            coords: Spacetime coordinates
            
        Returns:
            Dictionary of loss components
        """
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
        """
        Perform one training step.
        
        Args:
            optimizer_metric: Optimizer for metric model
            optimizer_matter: Optimizer for matter models (optional)
            batch_size: Batch size
            T_range: Time range for sampling
            spatial_range: Spatial range for sampling
            
        Returns:
            Dictionary of loss values
        """
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
        """
        Train the gravitational system.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            T_range: Time coordinate range
            spatial_range: Spatial coordinate range
            lr_metric: Learning rate for metric model
            lr_matter: Learning rate for matter models
            train_matter: Whether to train matter models
            scheduler_params: Parameters for learning rate scheduler
            checkpoint_interval: Save checkpoint every N epochs
            checkpoint_path: Path to save checkpoints
            
        Returns:
            Training history dictionary
            
        Example:
        --------
        ```python
        history = system.train(
            epochs=1000,
            batch_size=512,
            spatial_range=20.0,
            lr_metric=1e-4,
            scheduler_params={'factor': 0.5, 'patience': 100}
        )
        ```
        """
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
        """Save training checkpoint."""
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
        """Load training checkpoint."""
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
        """
        Evaluate the system at given coordinates.
        
        Args:
            test_coords: Test coordinates
            return_components: Whether to return individual components
            
        Returns:
            Dictionary of evaluation results
        """
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
        """
        Predict the metric at given coordinates.
        
        Args:
            coords: Spacetime coordinates
            
        Returns:
            Metric tensor
        """
        with torch.no_grad():
            g = self.metric_model(coords).reshape(-1, 4, 4)
            g = 0.5 * (g + g.transpose(-2, -1))
            return g
    
    def predict_curvature(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict curvature quantities at given coordinates.
        
        Args:
            coords: Spacetime coordinates
            
        Returns:
            Dictionary of curvature tensors and scalars
        """
        from .tensor_ops import (
            compute_einstein_tensor_vectorized,
            compute_kretschmann_scalar,
            TensorConfig
        )
        
        with torch.no_grad():
            # Get metric
            g = self.predict_metric(coords)
            
            # Compute curvature components
            tensor_config = TensorConfig()
            components = compute_einstein_tensor_vectorized(
                g, coords, self.metric_model, tensor_config, return_components=True
            )
            
            # Compute Kretschmann scalar
            kretschmann = compute_kretschmann_scalar(components['riemann'])
            
            return {
                'einstein_tensor': components['einstein'],
                'ricci_tensor': components['ricci'],
                'ricci_scalar': components['ricci_scalar'],
                'riemann_tensor': components['riemann'],
                'kretschmann_scalar': kretschmann
            }
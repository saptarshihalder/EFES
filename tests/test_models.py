"""
Unit tests for neural network models.

Tests model architectures, forward passes, and physics constraints.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from efes.models import (
    Sine,
    FourierFeatures,
    SIREN,
    MetricNet,
    FourierNet,
    PhysicsInformedNet,
    create_metric_model,
    ModelConfig
)


class TestSineActivation:
    """Test sine activation function."""
    
    def test_forward(self):
        """Test forward pass of sine activation."""
        activation = Sine(omega=1.0)
        x = torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        y = activation(x)
        
        expected = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0])
        assert torch.allclose(y, expected, atol=1e-5)
    
    def test_learnable_omega(self):
        """Test learnable frequency parameter."""
        activation = Sine(omega=30.0, learnable=True)
        
        # Check that omega is a parameter
        assert isinstance(activation.omega, nn.Parameter)
        assert activation.omega.item() == 30.0
        
        # Check gradient flow
        x = torch.randn(10, requires_grad=True)
        y = activation(x)
        loss = y.sum()
        loss.backward()
        
        assert activation.omega.grad is not None
        assert x.grad is not None


class TestFourierFeatures:
    """Test Fourier feature embedding."""
    
    def test_output_dimension(self):
        """Test that output has correct dimension."""
        in_features = 4
        num_frequencies = 128
        
        fourier = FourierFeatures(in_features, num_frequencies)
        
        batch_size = 10
        x = torch.randn(batch_size, in_features)
        y = fourier(x)
        
        # Output should have 2 * num_frequencies (sin and cos)
        assert y.shape == (batch_size, 2 * num_frequencies)
    
    def test_bounded_output(self):
        """Test that outputs are bounded [-1, 1]."""
        fourier = FourierFeatures(4, 64, scale=10.0)
        
        x = torch.randn(100, 4) * 10  # Large inputs
        y = fourier(x)
        
        # Sin and cos are bounded
        assert y.abs().max() <= 1.0 + 1e-6
    
    def test_learnable_frequencies(self):
        """Test learnable frequency matrix."""
        fourier = FourierFeatures(4, 32, learnable=True)
        
        assert isinstance(fourier.B, nn.Parameter)
        
        # Check gradient flow
        x = torch.randn(5, 4, requires_grad=True)
        y = fourier(x)
        loss = y.sum()
        loss.backward()
        
        assert fourier.B.grad is not None


class TestSIREN:
    """Test SIREN architecture."""
    
    def test_initialization(self):
        """Test SIREN weight initialization."""
        config = ModelConfig(hidden_features=64, hidden_layers=3)
        model = SIREN(config, in_features=4, out_features=16)
        
        # Check first layer initialization
        first_layer = model.layers[0]
        weight_std = first_layer.weight.std().item()
        
        # Should be initialized with uniform distribution
        expected_std = 1 / (4 * np.sqrt(3))  # std of uniform(-1/n, 1/n)
        assert abs(weight_std - expected_std) < 0.1
    
    def test_forward_shape(self):
        """Test output shape of SIREN."""
        config = ModelConfig(hidden_features=128, hidden_layers=4)
        model = SIREN(config, in_features=4, out_features=16)
        
        batch_size = 32
        x = torch.randn(batch_size, 4)
        y = model(x)
        
        assert y.shape == (batch_size, 16)
    
    def test_skip_connections(self):
        """Test skip connections in SIREN."""
        config = ModelConfig(
            hidden_features=64,
            hidden_layers=4,
            use_skip_connections=True
        )
        model = SIREN(config, in_features=4, out_features=1)
        
        # Check that skip connections are registered
        assert len(model.skip_connections) > 0
        
        # Forward pass should work
        x = torch.randn(10, 4)
        y = model(x)
        assert y.shape == (10, 1)
    
    def test_fourier_features_integration(self):
        """Test SIREN with Fourier features."""
        config = ModelConfig(
            hidden_features=128,
            use_fourier_features=True,
            fourier_scale=10.0
        )
        model = SIREN(config, in_features=4, out_features=16)
        
        # Should have Fourier feature layer
        assert model.fourier is not None
        
        # Forward pass
        x = torch.randn(20, 4)
        y = model(x)
        assert y.shape == (20, 16)


class TestMetricNet:
    """Test metric neural network."""
    
    def test_metric_output_shape(self):
        """Test that MetricNet outputs correct shape."""
        model = MetricNet()
        
        batch_size = 10
        coords = torch.randn(batch_size, 4)
        metric = model(coords)
        
        # Output should be flattened 4x4 metric
        assert metric.shape == (batch_size, 16)
        
        # Test tensor form
        metric_tensor = model.get_metric_tensor(coords)
        assert metric_tensor.shape == (batch_size, 4, 4)
    
    def test_metric_symmetry(self):
        """Test that output metric is symmetric."""
        model = MetricNet(enforce_symmetry=True)
        
        coords = torch.randn(5, 4)
        g = model.get_metric_tensor(coords)
        
        # Check g_μν = g_νμ
        assert torch.allclose(g, g.transpose(-2, -1), atol=1e-6)
    
    def test_metric_signature(self):
        """Test enforcement of Lorentzian signature."""
        model = MetricNet(enforce_signature=True)
        
        coords = torch.randn(20, 4)
        g = model.get_metric_tensor(coords)
        
        # Time component should be negative
        assert (g[:, 0, 0] < 0).all()
        
        # Spatial diagonal components should be positive
        for i in range(1, 4):
            assert (g[:, i, i] > 0).all()
    
    def test_gradient_flow(self):
        """Test gradient flow through MetricNet."""
        model = MetricNet()
        coords = torch.randn(5, 4, requires_grad=True)
        
        metric = model(coords)
        loss = metric.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
        assert coords.grad is not None


class TestFourierNet:
    """Test Fourier feature network."""
    
    def test_high_frequency_capability(self):
        """Test that FourierNet can represent high-frequency functions."""
        model = FourierNet(num_frequencies=256)
        
        # Train on high-frequency target
        coords = torch.linspace(-1, 1, 100).unsqueeze(1).repeat(1, 4)
        target = torch.sin(50 * coords[:, 0])  # High frequency
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Quick training
        for _ in range(100):
            pred = model(coords)[:, 0]
            loss = nn.MSELoss()(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should approximate high-frequency function
        final_loss = loss.item()
        assert final_loss < 0.1  # Reasonable approximation


class TestPhysicsInformedNet:
    """Test physics-informed neural network."""
    
    def test_spherical_symmetry(self):
        """Test enforcement of spherical symmetry."""
        model = PhysicsInformedNet(symmetry_type="spherical")
        
        # Points at same radius should give same metric
        r = 5.0
        coords1 = torch.tensor([[0.0, r, 0.0, 0.0]])
        coords2 = torch.tensor([[0.0, 0.0, r, 0.0]])
        coords3 = torch.tensor([[0.0, 0.0, 0.0, r]])
        
        metric1 = model(coords1)
        metric2 = model(coords2)
        metric3 = model(coords3)
        
        # Metrics should be similar (not exactly equal due to network architecture)
        assert torch.allclose(metric1, metric2, atol=0.1)
        assert torch.allclose(metric1, metric3, atol=0.1)
    
    def test_asymptotic_flatness(self):
        """Test asymptotic behavior at large distances."""
        model = PhysicsInformedNet(asymptotic_type="minkowski")
        
        # Far from origin
        coords = torch.tensor([[0.0, 1000.0, 0.0, 0.0]])
        metric = model(coords).reshape(4, 4)
        
        # Should approach Minkowski metric
        minkowski = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
        
        # Check diagonal components are close to Minkowski
        for i in range(4):
            assert abs(metric[i, i] - minkowski[i, i]) < 0.1
    
    def test_decay_rate_parameter(self):
        """Test learnable decay rate."""
        model = PhysicsInformedNet()
        
        # Decay rate should be a parameter
        assert hasattr(model, 'decay_rate')
        assert isinstance(model.decay_rate, nn.Parameter)
        
        # Should participate in gradient computation
        coords = torch.randn(5, 4, requires_grad=True)
        output = model(coords)
        loss = output.sum()
        loss.backward()
        
        assert model.decay_rate.grad is not None


class TestModelFactory:
    """Test model creation factory."""
    
    def test_create_siren(self):
        """Test creation of SIREN model."""
        model = create_metric_model("siren")
        assert isinstance(model, MetricNet)
        
        # Test with custom config
        config = ModelConfig(hidden_features=256, hidden_layers=6)
        model = create_metric_model("siren", config=config)
        assert model.config.hidden_features == 256
    
    def test_create_fourier(self):
        """Test creation of Fourier model."""
        model = create_metric_model("fourier")
        assert isinstance(model, FourierNet)
    
    def test_create_physics_informed(self):
        """Test creation of physics-informed model."""
        model = create_metric_model(
            "physics_informed",
            symmetry_type="spherical",
            asymptotic_type="minkowski"
        )
        assert isinstance(model, PhysicsInformedNet)
        assert model.symmetry_type == "spherical"
    
    def test_invalid_model_type(self):
        """Test error for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_metric_model("invalid_type")


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.hidden_features == 128
        assert config.hidden_layers == 4
        assert config.activation == "sine"
        assert config.omega == 30.0
        assert config.use_fourier_features == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            hidden_features=256,
            hidden_layers=8,
            dropout_rate=0.1,
            use_batch_norm=True
        )
        
        assert config.hidden_features == 256
        assert config.hidden_layers == 8
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
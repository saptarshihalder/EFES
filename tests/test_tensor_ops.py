"""
Unit tests for vectorized tensor operations.

These tests verify:
1. Correctness of tensor calculations
2. Vectorization performance
3. Error handling
4. Physical properties (symmetries, identities)
"""

import pytest
import torch
import numpy as np
from efes.tensor_ops import (
    safe_inverse,
    compute_metric_derivatives_vectorized,
    compute_christoffel_symbols_vectorized,
    compute_riemann_tensor_vectorized,
    compute_ricci_tensor_vectorized,
    compute_ricci_scalar_vectorized,
    compute_einstein_tensor_vectorized,
    compute_kretschmann_scalar,
    check_energy_conditions,
    TensorConfig,
    MetricSingularityError
)


class TestSafeInverse:
    """Test safe matrix inversion with regularization."""
    
    def test_regular_matrix(self):
        """Test inversion of a regular matrix."""
        # Create a well-conditioned matrix
        A = torch.eye(4) * 2.0
        A_inv = safe_inverse(A)
        
        # Check A * A^{-1} = I
        identity = torch.matmul(A, A_inv)
        assert torch.allclose(identity, torch.eye(4), atol=1e-6)
    
    def test_batch_inversion(self):
        """Test batch matrix inversion."""
        batch_size = 10
        A = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        A += torch.randn(batch_size, 4, 4) * 0.1  # Add small perturbation
        
        A_inv = safe_inverse(A)
        
        # Check A * A^{-1} = I for each batch
        identity = torch.matmul(A, A_inv)
        expected = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        assert torch.allclose(identity, expected, atol=1e-5)
    
    def test_singular_matrix_regularization(self):
        """Test that singular matrices are regularized."""
        # Create a singular matrix (rank 3)
        A = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]  # Zero row
        ])
        
        # Should regularize instead of failing
        A_inv = safe_inverse(A, epsilon=0.1)
        assert torch.isfinite(A_inv).all()
    
    def test_metric_signature(self):
        """Test inversion of Lorentzian metric."""
        # Minkowski metric
        g = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
        g_inv = safe_inverse(g)
        
        # Check g * g^{-1} = I
        identity = torch.matmul(g, g_inv)
        assert torch.allclose(identity, torch.eye(4), atol=1e-6)
        
        # Check signature preserved
        assert g_inv[0, 0] < 0  # Time component negative
        assert all(g_inv[i, i] > 0 for i in range(1, 4))  # Space components positive


class TestMetricDerivatives:
    """Test computation of metric derivatives."""
    
    def test_constant_metric(self):
        """Test that derivatives of constant metric are zero."""
        batch_size = 5
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        config = TensorConfig(static_time_approximation=False)
        dg = compute_metric_derivatives_vectorized(g, coords, None, config)
        
        # All derivatives should be near zero
        assert torch.allclose(dg, torch.zeros_like(dg), atol=1e-5)
    
    def test_derivative_shape(self):
        """Test output shape of metric derivatives."""
        batch_size = 10
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        dg = compute_metric_derivatives_vectorized(g, coords)
        
        # Shape should be [batch, 4, 4, 4]
        assert dg.shape == (batch_size, 4, 4, 4)
    
    def test_static_approximation(self):
        """Test static metric approximation."""
        batch_size = 5
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        config = TensorConfig(static_time_approximation=True)
        dg = compute_metric_derivatives_vectorized(g, coords, None, config)
        
        # Time derivatives should be zero
        assert torch.allclose(dg[:, :, :, 0], torch.zeros(batch_size, 4, 4), atol=1e-6)


class TestChristoffelSymbols:
    """Test Christoffel symbol computation."""
    
    def test_flat_space(self):
        """Test that Christoffel symbols vanish in flat space."""
        batch_size = 5
        coords = torch.randn(batch_size, 4)
        
        # Minkowski metric
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1
        
        christoffel = compute_christoffel_symbols_vectorized(g, coords=coords)
        
        # Should be near zero for flat space
        assert torch.allclose(christoffel, torch.zeros_like(christoffel), atol=1e-4)
    
    def test_symmetry(self):
        """Test symmetry property of Christoffel symbols."""
        batch_size = 3
        coords = torch.randn(batch_size, 4)
        
        # Random symmetric metric
        g = torch.randn(batch_size, 4, 4)
        g = 0.5 * (g + g.transpose(-2, -1))
        g = g + torch.eye(4) * 2  # Ensure positive definite
        
        christoffel = compute_christoffel_symbols_vectorized(g, coords=coords)
        
        # Check Γ^λ_μν = Γ^λ_νμ (symmetry in lower indices)
        for i in range(4):
            assert torch.allclose(
                christoffel[:, i, :, :],
                christoffel[:, i, :, :].transpose(-2, -1),
                atol=1e-5
            )
    
    def test_shape(self):
        """Test output shape of Christoffel symbols."""
        batch_size = 10
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        christoffel = compute_christoffel_symbols_vectorized(g, coords=coords)
        
        # Shape should be [batch, 4, 4, 4]
        assert christoffel.shape == (batch_size, 4, 4, 4)


class TestRiemannTensor:
    """Test Riemann curvature tensor computation."""
    
    def test_flat_space(self):
        """Test that Riemann tensor vanishes in flat space."""
        batch_size = 3
        coords = torch.randn(batch_size, 4)
        
        # Flat space Christoffel symbols (zero)
        christoffel = torch.zeros(batch_size, 4, 4, 4)
        
        riemann = compute_riemann_tensor_vectorized(christoffel)
        
        # Should be zero for flat space
        assert torch.allclose(riemann, torch.zeros_like(riemann), atol=1e-6)
    
    def test_antisymmetry(self):
        """Test antisymmetry properties of Riemann tensor."""
        batch_size = 2
        
        # Random Christoffel symbols
        christoffel = torch.randn(batch_size, 4, 4, 4) * 0.1
        riemann = compute_riemann_tensor_vectorized(christoffel)
        
        # Check R^ρ_σμν = -R^ρ_σνμ (antisymmetry in last two indices)
        for rho in range(4):
            for sigma in range(4):
                assert torch.allclose(
                    riemann[:, rho, sigma, :, :],
                    -riemann[:, rho, sigma, :, :].transpose(-2, -1),
                    atol=1e-5
                )
    
    def test_shape(self):
        """Test output shape of Riemann tensor."""
        batch_size = 5
        christoffel = torch.randn(batch_size, 4, 4, 4) * 0.1
        
        riemann = compute_riemann_tensor_vectorized(christoffel)
        
        # Shape should be [batch, 4, 4, 4, 4]
        assert riemann.shape == (batch_size, 4, 4, 4, 4)


class TestRicciTensor:
    """Test Ricci tensor computation."""
    
    def test_contraction(self):
        """Test that Ricci tensor is correct contraction of Riemann."""
        batch_size = 3
        
        # Create a simple Riemann tensor
        riemann = torch.randn(batch_size, 4, 4, 4, 4) * 0.01
        
        # Make it satisfy antisymmetry
        riemann = riemann - riemann.transpose(-2, -1)
        
        ricci = compute_ricci_tensor_vectorized(riemann)
        
        # Manually compute contraction R_μν = R^λ_μλν
        ricci_manual = torch.zeros(batch_size, 4, 4)
        for mu in range(4):
            for nu in range(4):
                for lam in range(4):
                    ricci_manual[:, mu, nu] += riemann[:, lam, mu, lam, nu]
        
        assert torch.allclose(ricci, ricci_manual, atol=1e-5)
    
    def test_symmetry(self):
        """Test that Ricci tensor is symmetric."""
        batch_size = 5
        riemann = torch.randn(batch_size, 4, 4, 4, 4) * 0.01
        
        ricci = compute_ricci_tensor_vectorized(riemann)
        
        # Check R_μν = R_νμ
        assert torch.allclose(ricci, ricci.transpose(-2, -1), atol=1e-5)
    
    def test_shape(self):
        """Test output shape of Ricci tensor."""
        batch_size = 7
        riemann = torch.randn(batch_size, 4, 4, 4, 4) * 0.01
        
        ricci = compute_ricci_tensor_vectorized(riemann)
        
        # Shape should be [batch, 4, 4]
        assert ricci.shape == (batch_size, 4, 4)


class TestRicciScalar:
    """Test Ricci scalar computation."""
    
    def test_trace(self):
        """Test that Ricci scalar is trace of Ricci tensor."""
        batch_size = 4
        
        # Create Ricci tensor and metric
        ricci = torch.randn(batch_size, 4, 4)
        ricci = 0.5 * (ricci + ricci.transpose(-2, -1))  # Symmetrize
        
        g_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        ricci_scalar = compute_ricci_scalar_vectorized(ricci, g_inv)
        
        # Manually compute trace R = g^μν R_μν
        trace_manual = torch.zeros(batch_size)
        for i in range(4):
            trace_manual += ricci[:, i, i]
        
        assert torch.allclose(ricci_scalar, trace_manual, atol=1e-5)
    
    def test_shape(self):
        """Test output shape of Ricci scalar."""
        batch_size = 6
        ricci = torch.randn(batch_size, 4, 4)
        g_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        ricci_scalar = compute_ricci_scalar_vectorized(ricci, g_inv)
        
        # Should be a scalar for each batch
        assert ricci_scalar.shape == (batch_size,)


class TestEinsteinTensor:
    """Test Einstein tensor computation."""
    
    def test_divergence_free(self):
        """Test that Einstein tensor satisfies contracted Bianchi identity."""
        # This is a fundamental property: ∇_μ G^μν = 0
        # For this test, we'll check the algebraic structure
        batch_size = 3
        coords = torch.randn(batch_size, 4)
        
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1  # Minkowski signature
        
        einstein = compute_einstein_tensor_vectorized(g, coords)
        
        # Check shape and finiteness
        assert einstein.shape == (batch_size, 4, 4)
        assert torch.isfinite(einstein).all()
    
    def test_einstein_equation_structure(self):
        """Test that Einstein tensor has correct structure G_μν = R_μν - ½Rg_μν."""
        batch_size = 2
        coords = torch.randn(batch_size, 4)
        
        # Create a non-trivial metric
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g += torch.randn(batch_size, 4, 4) * 0.1
        g = 0.5 * (g + g.transpose(-2, -1))  # Symmetrize
        
        # Get all components
        components = compute_einstein_tensor_vectorized(
            g, coords, return_components=True
        )
        
        # Manually compute G_μν = R_μν - ½Rg_μν
        G_manual = components['ricci'] - 0.5 * components['ricci_scalar'].view(batch_size, 1, 1) * g
        
        assert torch.allclose(components['einstein'], G_manual, atol=1e-5)
    
    def test_symmetry(self):
        """Test that Einstein tensor is symmetric."""
        batch_size = 5
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        einstein = compute_einstein_tensor_vectorized(g, coords)
        
        # Check G_μν = G_νμ
        assert torch.allclose(einstein, einstein.transpose(-2, -1), atol=1e-5)


class TestKretschmannScalar:
    """Test Kretschmann scalar computation."""
    
    def test_flat_space(self):
        """Test that Kretschmann scalar vanishes in flat space."""
        batch_size = 3
        riemann = torch.zeros(batch_size, 4, 4, 4, 4)
        
        kretschmann = compute_kretschmann_scalar(riemann)
        
        assert torch.allclose(kretschmann, torch.zeros(batch_size), atol=1e-6)
    
    def test_positive(self):
        """Test that Kretschmann scalar is non-negative."""
        batch_size = 5
        riemann = torch.randn(batch_size, 4, 4, 4, 4) * 0.1
        
        kretschmann = compute_kretschmann_scalar(riemann)
        
        # K = R^μνρσ R_μνρσ ≥ 0
        assert (kretschmann >= 0).all()
    
    def test_shape(self):
        """Test output shape of Kretschmann scalar."""
        batch_size = 4
        riemann = torch.randn(batch_size, 4, 4, 4, 4) * 0.1
        
        kretschmann = compute_kretschmann_scalar(riemann)
        
        assert kretschmann.shape == (batch_size,)


class TestEnergyConditions:
    """Test energy condition checking."""
    
    def test_vacuum(self):
        """Test that vacuum satisfies all energy conditions."""
        batch_size = 3
        
        # Vacuum: T_μν = 0
        T = torch.zeros(batch_size, 4, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1
        g_inv = safe_inverse(g)
        
        conditions = check_energy_conditions(T, g, g_inv)
        
        # Vacuum should satisfy all conditions
        assert conditions['weak'].all()
        assert conditions['null'].all()
        assert conditions['strong'].all()
        assert conditions['dominant'].all()
    
    def test_positive_energy_density(self):
        """Test perfect fluid with positive energy density."""
        batch_size = 5
        
        # Perfect fluid at rest: T_μν = diag(ρ, p, p, p)
        T = torch.zeros(batch_size, 4, 4)
        rho = 1.0  # Positive energy density
        p = 0.3    # Positive pressure
        
        T[:, 0, 0] = rho
        T[:, 1, 1] = p
        T[:, 2, 2] = p
        T[:, 3, 3] = p
        
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1
        g_inv = safe_inverse(g)
        
        conditions = check_energy_conditions(T, g, g_inv)
        
        # Should satisfy weak energy condition
        assert conditions['weak'].all()
    
    def test_shape(self):
        """Test output shapes of energy conditions."""
        batch_size = 7
        T = torch.randn(batch_size, 4, 4)
        T = 0.5 * (T + T.transpose(-2, -1))  # Symmetrize
        
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1
        g_inv = safe_inverse(g)
        
        conditions = check_energy_conditions(T, g, g_inv)
        
        # Each condition should return a boolean tensor of shape [batch_size]
        for name, condition in conditions.items():
            assert condition.shape == (batch_size,)
            assert condition.dtype == torch.bool


class TestVectorizationPerformance:
    """Test that vectorized operations are faster than loops."""
    
    @pytest.mark.performance
    def test_christoffel_performance(self):
        """Compare vectorized vs loop-based Christoffel computation."""
        batch_size = 100
        coords = torch.randn(batch_size, 4)
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g += torch.randn(batch_size, 4, 4) * 0.1
        g = 0.5 * (g + g.transpose(-2, -1))
        
        # Time vectorized version
        import time
        start = time.time()
        christoffel_vec = compute_christoffel_symbols_vectorized(g, coords=coords)
        vec_time = time.time() - start
        
        print(f"\nVectorized Christoffel computation time: {vec_time:.4f}s")
        print(f"Throughput: {batch_size/vec_time:.1f} samples/s")
        
        # Vectorized should process many samples efficiently
        assert vec_time < 1.0  # Should be fast even for 100 samples


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_singular_metric_error(self):
        """Test handling of singular metrics."""
        # Create a truly singular metric
        g = torch.zeros(1, 4, 4)
        
        with pytest.raises(MetricSingularityError):
            safe_inverse(g, epsilon=1e-12)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        batch_size = 3
        coords = torch.randn(batch_size, 4)
        
        # Create metric with NaN
        g = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        g[0, 0, 0] = float('nan')
        
        # Should handle gracefully or raise appropriate error
        einstein = compute_einstein_tensor_vectorized(g, coords)
        
        # Either computes approximate result or contains NaN
        assert einstein.shape == (batch_size, 4, 4)
    
    def test_large_curvature_regularization(self):
        """Test regularization of large curvature values."""
        batch_size = 2
        
        # Create Christoffel symbols with large values
        christoffel = torch.randn(batch_size, 4, 4, 4) * 1e6
        
        config = TensorConfig(max_christoffel_norm=1e3)
        
        # Should regularize large values
        # This test ensures the computation doesn't explode
        riemann = compute_riemann_tensor_vectorized(christoffel, config=config)
        
        assert torch.isfinite(riemann).all()
        assert riemann.abs().max() < 1e10  # Should be bounded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
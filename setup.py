"""
Setup script for Einstein Field Equations Solver (EFES).
"""

from setuptools import setup, find_packages

with open("README.md", "w") as f:
    f.write("""# Einstein Field Equations Solver (EFES)

A PyTorch-based library for solving Einstein's field equations using neural networks.

## Features

- **Fully Vectorized Tensor Operations**: High-performance computation of geometric quantities
- **Neural Network Models**: SIREN, Fourier networks, and physics-informed architectures
- **Multiple Matter Models**: Perfect fluids, scalar fields, electromagnetic fields, dark matter/energy
- **Physics Constraints**: Energy conditions, conservation laws, asymptotic behavior
- **Adaptive Sampling**: Intelligent sampling in high-curvature regions

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import efes

# Create a metric model
metric_model = efes.models.create_metric_model("siren")

# Create matter models
matter = efes.matter.create_matter_model("perfect_fluid", eos_type="dust")

# Create gravitational system
system = efes.GravitationalSystem(metric_model, [matter])

# Train the system
history = system.train(epochs=1000, batch_size=256)

# Evaluate metric at specific points
coords = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
metric = system.predict_metric(coords)
```

## Physics Approximations

1. **Numerical Derivatives**: Finite differences with adaptive step sizes
2. **Static Metric Approximation**: Optional for time-independent solutions
3. **Coordinate Regularization**: Handling of coordinate singularities
4. **Simplified Curvature**: Product terms only in Riemann tensor for stability

## Documentation

See the docstrings in each module for detailed physics explanations and usage examples.
""")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="efes",
    version="0.1.0",
    author="EFES Development Team",
    description="Einstein Field Equations Solver using Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/efes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.13.0",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
)
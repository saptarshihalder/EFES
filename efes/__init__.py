"""
Einstein Field Equations Solver (EFES)

A PyTorch-based library for solving Einstein's field equations using neural networks.
This package provides tools for learning spacetime metrics that satisfy the
Einstein field equations for various matter distributions.

Key Features:
- Fully vectorized tensor calculus operations
- Neural network models for metric learning (SIREN, Fourier networks)
- Various matter models (perfect fluids, scalar fields, electromagnetic fields)
- Physics-informed constraints and regularizations
- Visualization tools for spacetime geometry

Example Usage:
```python
import efes

# Create a metric model
metric_model = efes.models.create_metric_model("siren")

# Create matter models
matter = efes.matter.create_matter_model("perfect_fluid", eos_type="radiation")

# Create gravitational system
system = efes.GravitationalSystem(metric_model, [matter])

# Train the system
history = system.train(epochs=1000)
```
"""

__version__ = "0.1.0"

# Core modules
from . import tensor_ops
from . import models
from . import matter
from . import physics

# Main classes
from .models import (
    SIREN,
    MetricNet,
    FourierNet,
    PhysicsInformedNet,
    create_metric_model
)

from .matter import (
    MatterModel,
    PerfectFluidMatter,
    ScalarFieldMatter,
    ElectromagneticFieldMatter,
    DarkSectorMatter,
    create_matter_model
)

from .tensor_ops import (
    compute_christoffel_symbols_vectorized,
    compute_riemann_tensor_vectorized,
    compute_ricci_tensor_vectorized,
    compute_ricci_scalar_vectorized,
    compute_einstein_tensor_vectorized,
    compute_kretschmann_scalar,
    check_energy_conditions
)

from .physics import (
    compute_efe_loss,
    regularized_coordinates,
    adaptive_sampling_strategy,
    schwarzschild_initial_metric,
    adm_decomposition
)

# Import system class if we create it
try:
    from .system import GravitationalSystem, SystemConfig
except ImportError:
    pass

# Import visualization if we create it
try:
    from . import visualization
except ImportError:
    pass

# Import training utilities if we create them
try:
    from . import training
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    
    # Modules
    "tensor_ops",
    "models",
    "matter",
    "physics",
    
    # Model classes
    "SIREN",
    "MetricNet",
    "FourierNet",
    "PhysicsInformedNet",
    "create_metric_model",
    
    # Matter classes
    "MatterModel",
    "PerfectFluidMatter",
    "ScalarFieldMatter",
    "ElectromagneticFieldMatter",
    "DarkSectorMatter",
    "create_matter_model",
    
    # Tensor operations
    "compute_christoffel_symbols_vectorized",
    "compute_riemann_tensor_vectorized",
    "compute_ricci_tensor_vectorized",
    "compute_ricci_scalar_vectorized",
    "compute_einstein_tensor_vectorized",
    "compute_kretschmann_scalar",
    "check_energy_conditions",
    
    # Physics functions
    "compute_efe_loss",
    "regularized_coordinates",
    "adaptive_sampling_strategy",
    "schwarzschild_initial_metric",
    "adm_decomposition",
]
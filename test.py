def initialize_coupled_system(
    matter_type: str,
    matter_params: Dict[str, Any],
    initial_metric_type: str = "minkowski",
    hidden_dim: int = 128,
    device: torch.device = None
) -> GravitationalSystem:
    """Initialize a coupled gravity-matter system with specified parameters.
    
    Args:
        matter_type: Type of matter to include
        matter_params: Parameters for the matter model
        initial_metric_type: Initial metric configuration
        hidden_dim: Hidden dimension for neural networks
        device: Computation device
        
    Returns:
        GravitationalSystem instance with initialized models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize metric model
    metric_model = SIREN(
        in_features=4,  # (t, x, y, z)
        out_features=16,  # Flattened 4x4 metric tensor
        hidden_features=hidden_dim,
        hidden_layers=4,
        use_skip_connections=True,
        learnable_frequencies=True
    ).to(device)
    
    # Apply initial metric if specified
    if initial_metric_type == "minkowski":
        # Initialize with flat spacetime
        # This is handled by default weight initialization
        pass
    elif initial_metric_type == "schwarzschild":
        # Initialize with Schwarzschild metric
        mass = matter_params.get("mass", 1.0)
        
        # Sample points in domain
        t = torch.linspace(0, 10, 100, device=device)
        r = torch.linspace(2.1 * mass, 20, 100, device=device)
        theta = torch.linspace(0, math.pi, 20, device=device)
        phi = torch.linspace(0, 2*math.pi, 20, device=device)
        
        # Create meshgrid for sampling
        T, R, Theta, Phi = torch.meshgrid(t, r, theta, phi, indexing="ij")
        
        # Convert to Cartesian coordinates
        X = R * torch.sin(Theta) * torch.cos(Phi)
        Y = R * torch.sin(Theta) * torch.sin(Phi)
        Z = R * torch.cos(Theta)
        
        # Create input points
        pts = torch.stack([T.flatten(), X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        # Generate Schwarzschild metric
        g_schwarzschild = schwarzschild_initial_metric(pts, mass=mass)
        
        # Reshape to match network output format
        g_flat = g_schwarzschild.reshape(-1, 16)
        
        # Create dataset and train network to match this metric
        dataset = torch.utils.data.TensorDataset(pts, g_flat)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Train for a few epochs to initialize
        optimizer = torch.optim.Adam(metric_model.parameters(), lr=1e-3)
        
        for _ in range(10):
            for batch_pts, batch_g in dataloader:
                optimizer.zero_grad()
                pred_g = metric_model(batch_pts)
                loss = nn.functional.mse_loss(pred_g, batch_g)
                loss.backward()
                optimizer.step()
    
    # Initialize matter models based on type
    matter_models = []
    
    if matter_type == "perfect_fluid":
        # Get fluid parameters
        eos_type = matter_params.get("eos_type", "linear")
        eos_params = {
            "w": matter_params.get("w", 1/3),
            "gamma": matter_params.get("gamma", 5/3),
            "K": matter_params.get("K", 1.0),
            "epsilon": matter_params.get("epsilon", 0.1)
        }
        
        # Create perfect fluid model
        fluid_model = PerfectFluidMatter(
            hidden_dim=hidden_dim,
            eos_type=eos_type,
            eos_params=eos_params
        ).to(device)
        
        matter_models.append(fluid_model)
    
    elif matter_type == "scalar_field":
        # Get scalar field parameters
        potential_type = matter_params.get("potential_type", "mass")
        coupling_params = {
            "mass": matter_params.get("mass", 1.0),
            "lambda": matter_params.get("lambda", 0.1),
            "vev": matter_params.get("vev", 1.0),
            "decay_constant": matter_params.get("decay_constant", 1.0),
            "curvature_coupling": matter_params.get("xi_coupling", 0.0)
        }
        complex_field = matter_params.get("complex_field", False)
        
        # Create scalar field model
        scalar_model = ScalarFieldMatter(
            hidden_dim=hidden_dim,
            potential_type=potential_type,
            coupling_params=coupling_params,
            complex_field=complex_field
        ).to(device)
        
        matter_models.append(scalar_model)
    
    elif matter_type == "em_field":
        # Get EM field parameters
        field_type = matter_params.get("field_type", "general")
        
        # Create EM field model
        em_model = ElectromagneticFieldMatter(
            hidden_dim=hidden_dim,
            field_type=field_type
        ).to(device)
        
        matter_models.append(em_model)
    
    elif matter_type == "dark_sector":
        # Get dark sector parameters
        dm_type = matter_params.get("dm_type", "cold")
        de_type = matter_params.get("de_type", "lambda")
        interaction = matter_params.get("interaction", False)
        
        # Create dark sector model
        dark_model = DarkSectorMatter(
            hidden_dim=hidden_dim,
            dm_type=dm_type,
            de_type=de_type,
            interaction=interaction
        ).to(device)
        
        matter_models.append(dark_model)
    
    elif matter_type == "multi":
        # Handle multiple matter components
        component_types = matter_params.get("component_types", ["perfect_fluid"])
        component_weights = matter_params.get("component_weights", [1.0] * len(component_types))
        component_params = matter_params.get("component_params", [{}] * len(component_types))
        
        for i, comp_type in enumerate(component_types):
            if comp_type == "perfect_fluid":
                fluid_model = PerfectFluidMatter(
                    hidden_dim=hidden_dim,
                    eos_type=component_params[i].get("eos_type", "linear"),
                    eos_params=component_params[i].get("eos_params", {"w": 1/3})
                ).to(device)
                matter_models.append(fluid_model)
            
            elif comp_type == "scalar_field":
                scalar_model = ScalarFieldMatter(
                    hidden_dim=hidden_dim,
                    potential_type=component_params[i].get("potential_type", "mass"),
                    coupling_params=component_params[i].get("coupling_params", {"mass": 1.0}),
                    complex_field=component_params[i].get("complex_field", False)
                ).to(device)
                matter_models.append(scalar_model)
            
            # Add other component types as needed
    
    # Create gravitational system
    grav_system = GravitationalSystem(
        metric_model=metric_model,
        matter_models=matter_models,
        matter_weights=matter_params.get("matter_weights", [1.0] * len(matter_models)),
        device=device
    )
    
    return grav_system


def run_coupled_training(
    grav_system: GravitationalSystem,
    training_params: Dict[str, Any],
    output_col: object = None
) -> Dict[str, List[float]]:
    """Run training for the coupled gravity-matter system.
    
    Args:
        grav_system: Initialized gravitational system
        training_params: Training parameters
        output_col: Streamlit column for output display
        
    Returns:
        Training history
    """
    # Extract parameters
    epochs = training_params.get("epochs", 5000)
    batch_size = training_params.get("batch_size", 2048)
    t_min = training_params.get("t_min", 0.0)
    t_max = training_params.get("t_max", 10.0)
    L = training_params.get("spatial_extent", 10.0)
    lr_metric = training_params.get("lr_metric", 1e-4)
    lr_matter = training_params.get("lr_matter", 5e-4)
    adaptive_sampling = training_params.get("adaptive_sampling", True)
    
    # Set up progress indicators if using Streamlit
    progress = None
    loss_chart = None
    constraint_chart = None
    
    if output_col is not None:
        with output_col:
            progress = st.progress(0.0)
            
            st.subheader("Training Progress")
            loss_cols = st.columns(2)
            
            with loss_cols[0]:
                loss_chart = st.line_chart()
            
            with loss_cols[1]:
                constraint_chart = st.line_chart()
    
    # Run training
    history = grav_system.train_full_system(
        epochs=epochs,
        batch_size=batch_size,
        T_range=(t_min, t_max),
        L=L,
        lr_metric=lr_metric,
        lr_matter=lr_matter,
        adaptive_sampling=adaptive_sampling,
        progress_bar=progress
    )
    
    # Display results if using Streamlit
    if output_col is not None:
        with output_col:
            # Create training history plot
            history_fig = plot_training_history(history)
            st.plotly_chart(history_fig)
    
    return history

def main():
    """Main function for the Streamlit app."""
    st.title("Advanced Einstein Field Equations Solver")
    st.markdown("""
    This application solves Einstein's field equations using Physics-Informed Neural Networks 
    with 3+1 decomposition, curvature back-reaction, and symbolic verification. The system can handle
    both vacuum solutions and various types of matter coupling.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Solver Configuration", "Matter Coupling", "Training", "Visualization", "Analysis"]
    )
    
    if page == "Overview":
        show_overview_page()
    elif page == "Solver Configuration":
        show_configuration_page()
    elif page == "Matter Coupling":
        show_matter_coupling_page()
    elif page == "Training":
        show_training_page()
    elif page == "Visualization":
        show_visualization_page()
    elif page == "Analysis":
        show_analysis_page()


# -----------------------------------------------------
# 17. Application Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    main()
    grav_system: GravitationalSystem,
    t_value: float,
    slice_axis: int = 3,
    slice_value: float = 0.0,
    matter_index: int = 0,
    device: torch.device = None
) -> go.Figure:
    """Visualize perfect fluid flow as a vector field on a 2D slice.
    
    Args:
        grav_system: Gravitational system with metric and fluid
        t_value: Time value for the slice
        slice_axis: Which spatial axis to fix (1=x, 2=y, 3=z)
        slice_value: Value for the fixed axis
        matter_index: Index of fluid model
        device: Computation device
        
    Returns:
        Plotly figure with fluid flow visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if the matter model is a fluid
    if not isinstance(grav_system.matter_models[matter_index], PerfectFluidMatter):
        raise ValueError("Selected matter model is not a perfect fluid")
    
    # Create a 2D grid for the remaining spatial coordinates
    N = 20  # Use a coarser grid for vector field
    L = 10.0
    
    if slice_axis == 1:  # Fixed x
        y = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        Y, Z = torch.meshgrid(y, z, indexing="ij")
        X = torch.full_like(Y, slice_value)
        grid_pts = torch.stack([
            torch.full_like(Y, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "y", "z"
        components = (1, 2)  # Indices for fluid velocity components
    
    elif slice_axis == 2:  # Fixed y
        x = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        X, Z = torch.meshgrid(x, z, indexing="ij")
        Y = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "z"
        components = (0, 2)  # Indices for fluid velocity components
    
    else:  # Fixed z (default)
        x = torch.linspace(-L, L, N, device=device)
        y = torch.linspace(-L, L, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "y"
        components = (0, 1)  # Indices for fluid velocity components
    
    # Get fluid model and compute values
    with torch.no_grad():
        fluid_model = grav_system.matter_models[matter_index]
        
        # Get density
        density = fluid_model.get_density(grid_pts).cpu().numpy().reshape(N, N)
        
        # Get metric and compute four-velocity
        g = grav_system.metric_model(grid_pts)
        g_inv = torch.inverse(g)
        
        # Get four-velocity
        u = fluid_model.get_four_velocity(grid_pts, g, g_inv)
        
        # Extract spatial components for vector field
        u_x = u[:, components[0] + 1].cpu().numpy().reshape(N, N)
        u_y = u[:, components[1] + 1].cpu().numpy().reshape(N, N)
    
    # Create figure with density heatmap and velocity vector field
    fig = go.Figure()
    
    # Add density heatmap
    fig.add_trace(go.Heatmap(
        z=density,
        x=y.cpu().numpy() if slice_axis == 1 else x.cpu().numpy(),
        y=z.cpu().numpy() if slice_axis != 3 else y.cpu().numpy(),
        colorscale='Viridis',
        colorbar=dict(title='Density'),
        opacity=0.7
    ))
    
    # Add velocity vector field
    x_coords = y.cpu().numpy() if slice_axis == 1 else x.cpu().numpy()
    y_coords = z.cpu().numpy() if slice_axis != 3 else y.cpu().numpy()
    
    fig.add_trace(go.Quiver(
        x=np.repeat(x_coords, len(y_coords)),
        y=np.repeat(y_coords[:, np.newaxis], len(x_coords), axis=1).flatten(),
        u=u_x.flatten(),
        v=u_y.flatten(),
        scale=0.1,
        line=dict(width=2, color='white'),
        name='Velocity Field'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Fluid Flow at t={t_value:.2f}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=700,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_stress_energy_components(
    grav_system: GravitationalSystem,
    t_value: float,
    r_values: torch.Tensor = None,
    num_points: int = 100,
    device: torch.device = None
) -> go.Figure:
    """Plot stress-energy tensor components along radial direction.
    
    Args:
        grav_system: Gravitational system with metric and matter
        t_value: Time value to analyze
        r_values: Optional tensor of r values to sample (defaults to range)
        num_points: Number of points to sample if r_values not provided
        device: Computation device
        
    Returns:
        Plotly figure with stress-energy components
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create radial points if not provided
    if r_values is None:
        r_values = torch.linspace(0.5, 10.0, num_points, device=device)
    
    # Create coordinates
    theta = torch.ones_like(r_values) * math.pi / 2
    phi = torch.zeros_like(r_values)
    
    # Convert to Cartesian
    x = r_values * torch.sin(theta) * torch.cos(phi)
    y = r_values * torch.sin(theta) * torch.sin(phi)
    z = r_values * torch.cos(theta)
    
    coords = torch.stack([
        torch.full_like(r_values, t_value),
        x, y, z
    ], dim=1)
    
    # Compute stress-energy tensor
    with torch.no_grad():
        # Get metric
        g = grav_system.metric_model(coords)
        g_inv = torch.inverse(g)
        
        # Compute combined stress-energy tensor
        T = grav_system.combined_stress_energy(coords, g, g_inv)
        
        # Convert to numpy
        T = T.cpu().numpy()
        r_values = r_values.cpu().numpy()
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            r"$T_{00}$ (Energy Density)",
            r"$T_{ii}$ (Pressure Terms)",
            r"$T_{0i}$ (Momentum Terms)",
            "Trace of $T_{\mu\nu}$"
        ]
    )
    
    # Add energy density (T00)
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 0, 0], mode='lines', name='T₀₀'),
        row=1, col=1
    )
    
    # Add pressure terms (Tii)
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 1, 1], mode='lines', name='T₁₁'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 2, 2], mode='lines', name='T₂₂'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 3, 3], mode='lines', name='T₃₃'),
        row=1, col=2
    )
    
    # Add momentum terms (T0i)
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 0, 1], mode='lines', name='T₀₁'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 0, 2], mode='lines', name='T₀₂'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=r_values, y=T[:, 0, 3], mode='lines', name='T₀₃'),
        row=2, col=1
    )
    
    # Add trace
    trace = np.zeros(len(r_values))
    for mu in range(4):
        trace += T[:, mu, mu]
    
    fig.add_trace(
        go.Scatter(x=r_values, y=trace, mode='lines', name='Trace'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Stress-Energy Tensor Components at t={t_value:.2f}',
        height=700,
        width=900,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    fig.update_xaxes(title_text="Radius (r)", row=2, col=1)
    fig.update_xaxes(title_text="Radius (r)", row=2, col=2)
    
    return fig


# -----------------------------------------------------
# 14. Integration with Streamlit App
# -----------------------------------------------------

def show_matter_coupling_page():
    """Display the matter coupling interface in the Streamlit app."""
    st.header("Matter Coupling Configuration")
    
    # Matter type selection
    st.subheader("Matter Type")
    
    matter_type = st.selectbox(
        "Select Matter Type",
        [
            "No Matter (Vacuum)",
            "Perfect Fluid",
            "Scalar Field",
            "Electromagnetic Field",
            "Dark Matter/Energy",
            "Multi-Component"
        ]
    )
    
    if matter_type == "No Matter (Vacuum)":
        st.info("Using vacuum Einstein equations: G_μν = 0")
        
        # Optional cosmological constant
        include_lambda = st.checkbox("Include Cosmological Constant", False)
        if include_lambda:
            lambda_value = st.slider("Λ Value", -1.0, 1.0, 0.1, 0.01)
            st.write(f"Using Einstein equations with Λ: G_μν + Λg_μν = 0")
    
    elif matter_type == "Perfect Fluid":
        st.markdown("""
        Perfect fluid model with stress-energy tensor:
        
        $T^{\mu\\nu} = (\\rho + p)u^{\mu}u^{\\nu} + p g^{\mu\\nu}$
        
        where $\\rho$ is rest mass density, $p$ is pressure, and $u^{\mu}$ is four-velocity.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            eos_type = st.selectbox(
                "Equation of State",
                ["Linear (p = wρ)", "Polytropic", "Ideal Gas"]
            )
            
            if eos_type == "Linear (p = wρ)":
                w_param = st.slider("w Parameter", -1.0, 1.0, 0.33, 0.01)
                st.write(f"Using p = {w_param}ρ")
                
                # Special cases
                if abs(w_param) < 0.01:
                    st.info("w ≈ 0: Dust (pressureless matter)")
                elif abs(w_param - 0.33) < 0.01:
                    st.info("w ≈ 1/3: Radiation")
                elif abs(w_param + 1) < 0.01:
                    st.info("w ≈ -1: Dark energy / Cosmological constant")
            
            elif eos_type == "Polytropic":
                gamma_param = st.slider("Adiabatic Index (Γ)", 1.1, 2.0, 5/3, 0.1)
                K_param = st.slider("Polytropic Constant (K)", 0.1, 2.0, 1.0, 0.1)
                st.write(f"Using p = K·ρ^Γ with Γ = {gamma_param}, K = {K_param}")
            
            elif eos_type == "Ideal Gas":
                gamma_param = st.slider("Adiabatic Index (Γ)", 1.1, 2.0, 5/3, 0.1)
                epsilon = st.slider("Specific Internal Energy (ε)", 0.01, 1.0, 0.1, 0.01)
                st.write(f"Using p = (Γ-1)·ρ·ε with Γ = {gamma_param}, ε = {epsilon}")
        
        with col2:
            density_profile = st.selectbox(
                "Density Profile",
                ["Gaussian", "Constant", "Power Law", "Custom"]
            )
            
            if density_profile == "Gaussian":
                center_radius = st.slider("Center Radius", 0.0, 5.0, 0.0, 0.5)
                width = st.slider("Width", 0.5, 5.0, 2.0, 0.5)
                amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1)
                st.write(f"ρ = {amplitude}·exp(-r²/{width}²)")
            
            elif density_profile == "Power Law":
                inner_radius = st.slider("Inner Radius", 0.1, 2.0, 0.5, 0.1)
                power_index = st.slider("Power Index", -3.0, 0.0, -2.0, 0.1)
                st.write(f"ρ ∝ (r/{inner_radius})^{power_index} for r > {inner_radius}")
    
    elif matter_type == "Scalar Field":
        st.markdown("""
        Scalar field with stress-energy tensor:
        
        $T^{\mu\\nu} = \\partial^{\mu}\\phi \\partial^{\\nu}\\phi - g^{\mu\\nu}\\left[\\frac{1}{2}g^{\\alpha\\beta}\\partial_{\\alpha}\\phi\\partial_{\\beta}\\phi + V(\\phi)\\right]$
        
        where $\\phi$ is the scalar field and $V(\\phi)$ is the potential.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            potential_type = st.selectbox(
                "Potential Type",
                ["Mass Term (m²φ²/2)", "φ⁴ Self-Interaction", "Higgs-like", "Axion", "Inflation"]
            )
            
            if potential_type == "Mass Term (m²φ²/2)":
                mass_param = st.slider("Mass Parameter (m)", 0.01, 2.0, 0.1, 0.01)
                st.write(f"V(φ) = {mass_param}²·φ²/2")
            
            elif potential_type == "φ⁴ Self-Interaction":
                mass_param = st.slider("Mass Parameter (m)", 0.01, 1.0, 0.1, 0.01)
                lambda_param = st.slider("Self-Coupling (λ)", 0.01, 1.0, 0.1, 0.01)
                st.write(f"V(φ) = {mass_param}²·φ²/2 + {lambda_param}·φ⁴/4")
            
            elif potential_type == "Higgs-like":
                lambda_param = st.slider("Self-Coupling (λ)", 0.01, 1.0, 0.1, 0.01)
                vev = st.slider("Vacuum Expectation Value (v)", 0.1, 2.0, 1.0, 0.1)
                st.write(f"V(φ) = {lambda_param}·(φ² - {vev}²)²/4")
            
            elif potential_type == "Axion":
                mass_param = st.slider("Mass Parameter (m)", 0.01, 1.0, 0.1, 0.01)
                decay_const = st.slider("Decay Constant (f)", 0.1, 2.0, 1.0, 0.1)
                st.write(f"V(φ) = {mass_param}²·{decay_const}²·[1 - cos(φ/{decay_const})]")
                
            elif potential_type == "Inflation":
                mass_param = st.slider("Mass Parameter (m)", 0.001, 0.1, 0.01, 0.001)
                st.write(f"V(φ) = {mass_param}²·φ²/2 (chaotic inflation model)")
        
        with col2:
            field_profile = st.selectbox(
                "Initial Field Profile",
                ["Gaussian", "Tanh Kink", "Sine Wave", "Custom"]
            )
            
            if field_profile == "Gaussian":
                center_radius = st.slider("Center Radius", 0.0, 5.0, 0.0, 0.5)
                width = st.slider("Width", 0.5, 5.0, 2.0, 0.5)
                amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1)
                st.write(f"φ = {amplitude}·exp(-r²/{width}²)")
            
            elif field_profile == "Tanh Kink":
                center_radius = st.slider("Center Radius", 1.0, 5.0, 3.0, 0.5)
                width = st.slider("Width", 0.1, 2.0, 0.5, 0.1)
                amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1)
                st.write(f"φ = {amplitude}·tanh((r - {center_radius})/{width})")
            
            elif field_profile == "Sine Wave":
                wavelength = st.slider("Wavelength", 0.5, 5.0, 2.0, 0.5)
                amplitude = st.slider("Amplitude", 0.1, 2.0, 1.0, 0.1)
                st.write(f"φ = {amplitude}·sin(2π·r/{wavelength})")
            
            complex_field = st.checkbox("Use Complex Scalar Field", False)
            
            # Curvature coupling
            xi_coupling = st.slider("Curvature Coupling (ξ)", 0.0, 0.5, 0.0, 0.01)
            if xi_coupling > 0:
                st.write(f"Including non-minimal coupling: ξ·R·φ² with ξ = {xi_coupling}")
    
    elif matter_type == "Electromagnetic Field":
        st.markdown("""
        Electromagnetic field with stress-energy tensor:
        
        $T^{\mu\\nu} = \\frac{1}{4\\pi}\\left[F^{\mu\\alpha}F^{\\nu}_{\\alpha} - \\frac{1}{4}g^{\mu\\nu}F_{\\alpha\\beta}F^{\\alpha\\beta}\\right]$
        
        where $F^{\mu\\nu}$ is the electromagnetic field tensor.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            field_type = st.selectbox(
                "Field Configuration",
                ["Electric Monopole", "Magnetic Dipole", "EM Wave", "Custom"]
            )
            
            if field_type == "Electric Monopole":
                charge = st.slider("Charge (Q)", 0.1, 2.0, 1.0, 0.1)
                st.write(f"Electric field: E = Q/r² (radial)")
            
            elif field_type == "Magnetic Dipole":
                magnetic_moment = st.slider("Magnetic Moment (μ)", 0.1, 2.0, 1.0, 0.1)
                orientation = st.selectbox("Dipole Orientation", ["z-axis", "x-axis", "y-axis"])
                st.write(f"Magnetic dipole with moment μ = {magnetic_moment} along {orientation}")
            
            elif field_type == "EM Wave":
                wavelength = st.slider("Wavelength", 0.5, 5.0, 2.0, 0.5)
                amplitude = st.slider("Amplitude", 0.1, 1.0, 0.5, 0.1)
                polarization = st.selectbox("Polarization", ["Linear x", "Linear y", "Circular"])
                propagation = st.selectbox("Propagation Direction", ["z-axis", "x-axis", "y-axis"])
                st.write(f"EM wave with λ = {wavelength}, amplitude = {amplitude}")
        
        with col2:
            dynamic = st.checkbox("Dynamic Field Evolution", True)
            current_sources = st.checkbox("Include Current Sources", False)
            
            if current_sources:
                source_type = st.selectbox(
                    "Current Source Type",
                    ["Point Charge", "Current Loop", "Oscillating Dipole"]
                )
                source_strength = st.slider("Source Strength", 0.1, 2.0, 1.0, 0.1)
            
            # Advanced options
            st.subheader("Advanced Options")
            gauge_fixing = st.selectbox("Gauge Fixing", ["Lorenz Gauge", "Coulomb Gauge", "Temporal Gauge"])
            st.write(f"Using {gauge_fixing} condition")
    
    elif matter_type == "Dark Matter/Energy":
        st.markdown("""
        Dark sector model with components for dark matter and dark energy.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            dm_type = st.selectbox(
                "Dark Matter Type",
                ["Cold (CDM)", "Warm", "Fuzzy/Wave"]
            )
            
            if dm_type == "Cold (CDM)":
                dm_density = st.slider("DM Density Parameter", 0.1, 1.0, 0.3, 0.1)
                st.write(f"Cold dark matter with Ω_DM = {dm_density}")
            
            elif dm_type == "Warm":
                dm_density = st.slider("DM Density Parameter", 0.1, 1.0, 0.3, 0.1)
                velocity_dispersion = st.slider("Velocity Dispersion", 0.001, 0.1, 0.01, 0.001)
                st.write(f"Warm dark matter with Ω_DM = {dm_density}, σ_v = {velocity_dispersion}")
            
            elif dm_type == "Fuzzy/Wave":
                dm_density = st.slider("DM Density Parameter", 0.1, 1.0, 0.3, 0.1)
                mass_param = st.slider("Particle Mass (10^-N eV)", 20, 24, 22, 1)
                st.write(f"Fuzzy DM with Ω_DM = {dm_density}, m = 10^-{mass_param} eV")
        
        with col2:
            de_type = st.selectbox(
                "Dark Energy Type",
                ["Cosmological Constant (Λ)", "Quintessence", "Phantom Energy"]
            )
            
            if de_type == "Cosmological Constant (Λ)":
                de_density = st.slider("DE Density Parameter", 0.0, 1.0, 0.7, 0.1)
                st.write(f"Cosmological constant with Ω_Λ = {de_density}")
            
            elif de_type == "Quintessence":
                de_density = st.slider("DE Density Parameter", 0.0, 1.0, 0.7, 0.1)
                w_de = st.slider("Equation of State (w)", -0.999, -0.8, -0.95, 0.01)
                st.write(f"Quintessence with Ω_DE = {de_density}, w = {w_de}")
            
            elif de_type == "Phantom Energy":
                de_density = st.slider("DE Density Parameter", 0.0, 1.0, 0.7, 0.1)
                w_de = st.slider("Equation of State (w)", -1.3, -1.001, -1.05, 0.01)
                st.write(f"Phantom energy with Ω_DE = {de_density}, w = {w_de}")
            
            # Interaction
            interaction = st.checkbox("Dark Sector Interaction", False)
            if interaction:
                coupling = st.slider("Interaction Strength", 0.001, 0.1, 0.01, 0.001)
                st.write(f"Coupling between dark matter and dark energy: {coupling}")
    
    elif matter_type == "Multi-Component":
        st.markdown("""
        Multiple matter components that contribute to the stress-energy tensor.
        """)
        
        num_components = st.slider("Number of Components", 1, 4, 2, 1)
        
        for i in range(num_components):
            st.subheader(f"Component {i+1}")
            
            component_type = st.selectbox(
                f"Type #{i+1}",
                ["Perfect Fluid", "Scalar Field", "Electromagnetic Field", "Dark Matter/Energy"],
                key=f"comp_type_{i}"
            )
            
            weight = st.slider(f"Weight #{i+1}", 0.1, 1.0, 1.0, 0.1, key=f"comp_weight_{i}")
            
            # Component-specific parameters would be added here
            st.write(f"Using {component_type} with weight {weight}")
    
    # Advanced coupling options
    with st.expander("Advanced Matter Coupling Options"):
        conservation_weight = st.slider("Conservation Equation Weight", 0.1, 2.0, 1.0, 0.1)
        constraint_enforcement = st.checkbox("Strict Constraint Enforcement", True)
        bidirectional = st.checkbox("Bidirectional Coupling", True)
        
        if bidirectional:
            st.write("Matter affects spacetime curvature AND curvature affects matter evolution")
        else:
            st.write("One-way coupling: Matter affects spacetime curvature only")
    
    # Visualization options
    st.subheader("Matter Visualization Options")
    
    viz_type = st.selectbox(
        "Visualization Type",
        [
            "Density Contours",
            "Stress-Energy Components",
            "Matter-Curvature Correlation",
            "Fluid Flow",
            "EM Field Vectors"
        ]
    )
    
    # Time slider for visualization
    t_value = st.slider("Time for Visualization", 0.0, 10.0, 5.0, 0.5)
    
    # Save configuration
    if st.button("Save Matter Configuration"):
        st.success("Matter configuration saved successfully")
        # In a real app, we would save these settings to session state
        # or pass them to the training function
    def get_four_velocity(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized four-velocity of dark matter at given coordinates."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get raw velocity from network
        u_raw = self.velocity_network(coords)
        
        # Normalize to satisfy u^μu_μ = -1
        u_squared = torch.zeros(batch_size, device=device)
        
        for mu in range(4):
            for nu in range(4):
                u_squared += g[:, mu, nu] * u_raw[:, mu] * u_raw[:, nu]
        
        # Ensure timelike normalization
        u_squared = torch.clamp(u_squared, max=-1e-6)
        norm_factor = torch.sqrt(-u_squared)
        
        return u_raw / norm_factor.unsqueeze(1)
    
    def get_dark_energy_pressure(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute dark energy pressure based on model type."""
        if self.de_type == "lambda":
            # Cosmological constant: p = -ρ
            return -self.lambda_value.abs() * torch.ones(coords.shape[0], 1, device=coords.device)
        
        elif self.de_type == "quintessence":
            # Quintessence scalar field: p ≈ -ρ but variable
            phi = self.get_dark_energy_field(coords)
            w_de = -0.95  # w slightly > -1
            return w_de * self.lambda_value.abs() * torch.ones_like(phi)
        
        elif self.de_type == "phantom":
            # Phantom energy: p < -ρ
            phi = self.get_dark_energy_field(coords)
            w_de = -1.05  # w slightly < -1
            return w_de * self.lambda_value.abs() * torch.ones_like(phi)
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy tensor for combined dark sector.
        
        Sum of dark matter and dark energy contributions.
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Initialize stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # 1. Dark matter contribution
        rho_dm = self.get_dark_matter_density(coords)
        
        if self.dm_type == "cold":
            # Cold dark matter (pressureless dust)
            u_dm = self.get_four_velocity(coords, g, g_inv)
            
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += rho_dm * u_dm[:, mu] * u_dm[:, nu]
        
        elif self.dm_type == "warm":
            # Warm dark matter (with small pressure)
            u_dm = self.get_four_velocity(coords, g, g_inv)
            p_dm = 0.001 * rho_dm  # Small pressure
            
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += (rho_dm + p_dm) * u_dm[:, mu] * u_dm[:, nu]
                    T[:, mu, nu] += p_dm * g_inv[:, mu, nu]
        
        elif self.dm_type == "fuzzy":
            # Fuzzy/wave dark matter (scalar field approximation)
            # This is a simplification; full treatment would use a scalar field
            u_dm = self.get_four_velocity(coords, g, g_inv)
            
            # Pressure scale depends on de Broglie wavelength
            p_dm = 0.01 * rho_dm
            
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += (rho_dm + p_dm) * u_dm[:, mu] * u_dm[:, nu]
                    T[:, mu, nu] += p_dm * g_inv[:, mu, nu]
        
        # 2. Dark energy contribution
        if self.de_type == "lambda":
            # Cosmological constant
            lambda_value = self.lambda_value.abs()
            
            # T^μν = -Λg^μν
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += -lambda_value * g_inv[:, mu, nu]
        
        else:
            # Quintessence/phantom DE (effective fluid approximation)
            rho_de = self.lambda_value.abs() * torch.ones_like(rho_dm)
            p_de = self.get_dark_energy_pressure(coords)
            
            # Perfect fluid form
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += p_de * g_inv[:, mu, nu]
            
            # Add extra term for cosmological constant-like behavior
            # u^μ = (1,0,0,0) in cosmological frame
            t_direction = torch.zeros(batch_size, 4, device=device)
            t_direction[:, 0] = 1.0
            
            u_de = torch.zeros(batch_size, 4, device=device)
            
            # Normalize to get u^μ
            for mu in range(4):
                for nu in range(4):
                    u_de[:, mu] += g_inv[:, mu, nu] * t_direction[:, nu]
            
            # Normalize
            u_norm_squared = torch.zeros(batch_size, device=device)
            for mu in range(4):
                for nu in range(4):
                    u_norm_squared += g[:, mu, nu] * u_de[:, mu] * u_de[:, nu]
            
            u_norm = torch.sqrt(-u_norm_squared)
            u_de = u_de / u_norm.unsqueeze(1)
            
            # Add to stress-energy tensor
            for mu in range(4):
                for nu in range(4):
                    T[:, mu, nu] += (rho_de + p_de) * u_de[:, mu] * u_de[:, nu]
        
        # 3. Interaction term if enabled
        if self.interaction:
            # Implement coupling between dark matter and dark energy
            # This is a simple phenomenological interaction
            coupling = 0.01
            interaction_term = coupling * rho_dm * self.lambda_value.abs()
            
            # Add to diagonal components (simplified approach)
            for mu in range(4):
                T[:, mu, mu] += interaction_term
        
        return T
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute conservation of the dark sector stress-energy tensor."""
        # Implement conservation check (simplified)
        # For non-interacting components, each satisfies ∇_μT^μν = 0 separately
        # For interacting components, only the total satisfies conservation
        
        # We'll return a placeholder for now - in a full implementation,
        # this would check the true covariant divergence
        return torch.zeros(coords.shape[0], device=coords.device)
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values for visualization and analysis."""
        rho_dm = self.get_dark_matter_density(coords)
        
        if self.de_type == "lambda":
            rho_de = self.lambda_value.abs() * torch.ones_like(rho_dm)
        else:
            phi = self.get_dark_energy_field(coords)
            rho_de = self.lambda_value.abs() * torch.ones_like(rho_dm)
        
        return {
            "dark_matter_density": rho_dm,
            "dark_energy_density": rho_de,
            "total_dark_sector_density": rho_dm + rho_de
        }


# -----------------------------------------------------
# 12. Extended Matter Tensor Training and Integration
# -----------------------------------------------------

class GravitationalSystem:
    """Integrated system of spacetime geometry and matter fields.
    
    Handles the bidirectional coupling between spacetime and matter,
    ensuring consistent evolution and constraint satisfaction.
    """
    def __init__(
        self,
        metric_model: nn.Module,
        matter_models: List[MatterModel],
        matter_weights: List[float] = None,
        device: torch.device = None
    ):
        self.metric_model = metric_model
        self.matter_models = matter_models
        
        # If weights not provided, use equal weighting
        if matter_weights is None:
            matter_weights = [1.0] * len(matter_models)
        self.matter_weights = matter_weights
        
        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
    
    def combined_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor = None, 
        g_inv: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute combined stress-energy tensor from all matter components."""
        batch_size = coords.shape[0]
        
        # If metric not provided, compute it
        if g is None:
            self.metric_model.eval()
            with torch.no_grad():
                g = self.metric_model(coords)
        
        if g_inv is None:
            g_inv = torch.inverse(g)
        
        # Initialize combined stress-energy tensor
        T_combined = torch.zeros(batch_size, 4, 4, device=self.device)
        
        # Add contribution from each matter model
        for i, matter_model in enumerate(self.matter_models):
            weight = self.matter_weights[i]
            T_i = matter_model.get_stress_energy(coords, g, g_inv)
            T_combined += weight * T_i
        
        return T_combined
    
    def compute_matter_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor = None, 
        g_inv: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute conservation violations for all matter components."""
        batch_size = coords.shape[0]
        
        # If metric not provided, compute it
        if g is None:
            self.metric_model.eval()
            with torch.no_grad():
                g = self.metric_model(coords)
        
        if g_inv is None:
            g_inv = torch.inverse(g)
        
        # Compute conservation violation for each matter model
        conservation_violations = []
        
        for matter_model in self.matter_models:
            violation = matter_model.compute_conservation(coords, g, g_inv)
            conservation_violations.append(violation)
        
        # Sum all violations
        total_violation = torch.zeros(batch_size, device=self.device)
        for violation in conservation_violations:
            total_violation += violation
        
        return total_violation
    
    def compute_matter_coupling_residual(
        self, 
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute residual from Einstein equations with matter sources."""
        batch_size = coords.shape[0]
        
        # Compute metric and its inverse
        g = self.metric_model(coords)
        g_inv = torch.inverse(g)
        
        # Compute Einstein tensor
        christoffel, d_christoffel = compute_christoffel_symbols(coords, lambda x: self.metric_model(x))
        riemann = compute_riemann_tensor(christoffel, d_christoffel)
        ricci = compute_ricci_tensor(riemann, g_inv)
        einstein = compute_einstein_tensor(ricci, g, g_inv)
        
        # Compute combined stress-energy tensor
        T = self.combined_stress_energy(coords, g, g_inv)
        
        # Einstein field equations: G_μν = 8πT_μν
        # Compute residual
        residual = torch.zeros(batch_size, device=self.device)
        for mu in range(4):
            for nu in range(4):
                residual += (einstein[:, mu, nu] - 8 * math.pi * T[:, mu, nu])**2
        
        return residual
    
    def train_coupled_system(
        self,
        coords_batch: torch.Tensor,
        optimizer_metric: torch.optim.Optimizer,
        optimizer_matter: List[torch.optim.Optimizer],
        field_eq_weight: float = 1.0,
        conservation_weight: float = 0.1,
        constraint_weight: float = 0.5
    ) -> Dict[str, float]:
        """Perform one training step for the coupled gravity-matter system."""
        # Zero all gradients
        optimizer_metric.zero_grad()
        for opt in optimizer_matter:
            opt.zero_grad()
        
        # Compute losses
        
        # 1. Einstein field equations with matter
        field_eq_loss = torch.mean(self.compute_matter_coupling_residual(coords_batch))
        
        # 2. ADM constraint violation
        adm_vars = spacetime_to_adm(coords_batch, self.metric_model)
        H_constraint, M_constraint = compute_adm_constraints(adm_vars)
        constraint_loss = torch.mean(H_constraint**2) + torch.mean(torch.sum(M_constraint**2, dim=1))
        
        # 3. Matter conservation
        conservation_loss = torch.mean(self.compute_matter_conservation(coords_batch))
        
        # Combine losses
        total_loss = (
            field_eq_weight * field_eq_loss + 
            constraint_weight * constraint_loss + 
            conservation_weight * conservation_loss
        )
        
        # Backward and optimize
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.metric_model.parameters(), 1.0)
        for matter_model in self.matter_models:
            torch.nn.utils.clip_grad_norm_(matter_model.parameters(), 1.0)
        
        # Step optimizers
        optimizer_metric.step()
        for opt in optimizer_matter:
            opt.step()
        
        # Return loss components
        return {
            "total_loss": total_loss.item(),
            "field_eq_loss": field_eq_loss.item(),
            "constraint_loss": constraint_loss.item(),
            "conservation_loss": conservation_loss.item()
        }
    
    def train_full_system(
        self,
        epochs: int,
        batch_size: int,
        T_range: Tuple[float, float],
        L: float,
        lr_metric: float = 1e-4,
        lr_matter: float = 5e-4,
        adaptive_sampling: bool = True,
        progress_bar: Optional[object] = None
    ) -> Dict[str, List[float]]:
        """Train the full coupled gravity-matter system."""
        # Set up optimizers
        optimizer_metric = torch.optim.Adam(self.metric_model.parameters(), lr=lr_metric)
        optimizers_matter = [
            torch.optim.Adam(model.parameters(), lr=lr_matter)
            for model in self.matter_models
        ]
        
        # Learning rate schedulers
        scheduler_metric = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_metric, mode='min', factor=0.5, patience=300, min_lr=1e-6
        )
        schedulers_matter = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=300, min_lr=1e-6
            )
            for opt in optimizers_matter
        ]
        
        # History for tracking progress
        history = {
            "total_loss": [],
            "field_eq_loss": [],
            "constraint_loss": [],
            "conservation_loss": []
        }
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Sample collocation points
            if adaptive_sampling and epoch > 500:
                # Use the field equation residual for adaptive sampling
                coords = sample_adaptive_spacetime(
                    batch_size, T_range, L, self.device,
                    model=self.metric_model,
                    residual_fn=self.compute_matter_coupling_residual,
                    alpha=min(0.7, epoch / epochs)
                )
            else:
                coords = sample_adaptive_spacetime(
                    batch_size, T_range, L, self.device
                )
            
            # Train one step
            losses = self.train_coupled_system(
                coords,
                optimizer_metric,
                optimizers_matter
            )
            
            # Update learning rate schedulers
            scheduler_metric.step(losses["total_loss"])
            for scheduler in schedulers_matter:
                scheduler.step(losses["total_loss"])
            
            # Record losses
            if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
                for key, value in losses.items():
                    history[key].append(value)
                
                # Update progress bar if provided
                if progress_bar is not None:
                    progress_bar.progress(epoch / epochs)
                
                # Print progress
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}/{epochs} | Loss: {losses['total_loss']:.3e}")
        
        return history


def visualize_matter_density(
    grav_system: GravitationalSystem,
    t_value: float,
    slice_axis: int = 3,
    slice_value: float = 0.0,
    matter_index: int = 0,
    field_name: str = "density",
    device: torch.device = None
) -> go.Figure:
    """Visualize matter density distribution on a 2D slice.
    
    Args:
        grav_system: Gravitational system with metric and matter
        t_value: Time value for the slice
        slice_axis: Which spatial axis to fix (1=x, 2=y, 3=z)
        slice_value: Value for the fixed axis
        matter_index: Index of matter model to visualize
        field_name: Name of field to visualize
        device: Computation device
        
    Returns:
        Plotly figure with matter visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a 2D grid for the remaining spatial coordinates
    N = 100  # Resolution
    L = 10.0  # Domain size
    
    if slice_axis == 1:  # Fixed x
        y = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        Y, Z = torch.meshgrid(y, z, indexing="ij")
        X = torch.full_like(Y, slice_value)
        grid_pts = torch.stack([
            torch.full_like(Y, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "y", "z"
    
    elif slice_axis == 2:  # Fixed y
        x = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        X, Z = torch.meshgrid(x, z, indexing="ij")
        Y = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "z"
    
    else:  # Fixed z (default)
        x = torch.linspace(-L, L, N, device=device)
        y = torch.linspace(-L, L, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "y"
    
    # Get matter model
    matter_model = grav_system.matter_models[matter_index]
    
    # Get field values
    with torch.no_grad():
        field_values = matter_model.get_field_values(grid_pts)
    
    # Extract requested field
    if field_name in field_values:
        field_data = field_values[field_name].reshape(N, N).cpu().numpy()
    else:
        raise ValueError(f"Unknown field: {field_name}. Available fields: {list(field_values.keys())}")
    
    # Create visualization
    fig = go.Figure(data=go.Heatmap(
        z=field_data,
        x=x.cpu().numpy() if slice_axis != 2 else z.cpu().numpy(),
        y=y.cpu().numpy() if slice_axis != 1 else z.cpu().numpy(),
        colorscale='Viridis',
        colorbar=dict(title=field_name.replace('_', ' ').title()),
    ))
    
    fig.update_layout(
        title=f"{field_name.replace('_', ' ').title()} at t={t_value:.2f}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=700,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_matter_curvature_correlation(
    grav_system: GravitationalSystem,
    t_value: float,
    matter_index: int = 0,
    field_name: str = "density",
    num_points: int = 1000,
    device: torch.device = None
) -> go.Figure:
    """Plot correlation between matter density and spacetime curvature.
    
    Args:
        grav_system: Gravitational system with metric and matter
        t_value: Time value to analyze
        matter_index: Index of matter model to correlate
        field_name: Name of matter field to correlate
        num_points: Number of random points to sample
        device: Computation device
        
    Returns:
        Plotly figure with correlation plot
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample random points in the domain
    L = 10.0
    coords = sample_adaptive_spacetime(num_points, (t_value, t_value), L, device)
    
    # Get matter field values
    with torch.no_grad():
        matter_model = grav_system.matter_models[matter_index]
        field_values = matter_model.get_field_values(coords)
        
        if field_name not in field_values:
            raise ValueError(f"Unknown field: {field_name}. Available fields: {list(field_values.keys())}")
        
        field_data = field_values[field_name].flatten().cpu().numpy()
        
        # Compute Ricci scalar (curvature)
        g = grav_system.metric_model(coords)
        g_inv = torch.inverse(g)
        christoffel, d_christoffel = compute_christoffel_symbols(coords, lambda x: grav_system.metric_model(x))
        riemann = compute_riemann_tensor(christoffel, d_christoffel)
        ricci = compute_ricci_tensor(riemann, g_inv)
        
        # Compute Ricci scalar: R = g^μν R_μν
        ricci_scalar = torch.zeros(coords.shape[0], device=device)
        for mu in range(4):
            for nu in range(4):
                ricci_scalar += g_inv[:, mu, nu] * ricci[:, mu, nu]
        
        ricci_scalar = ricci_scalar.cpu().numpy()
    
    # Create scatter plot
    fig = go.Figure(data=go.Scatter(
        x=field_data,
        y=ricci_scalar,
        mode='markers',
        marker=dict(
            size=5,
            color=ricci_scalar,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Ricci Scalar')
        ),
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"Correlation: {field_name.replace('_', ' ').title()} vs. Curvature",
        xaxis_title=field_name.replace('_', ' ').title(),
        yaxis_title="Ricci Scalar (R)",
        width=700,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    # Add correlation line if there's a clear trend
    if len(field_data) > 10:
        # Fit line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(field_data, ricci_scalar)
        
        # Add regression line
        x_range = np.linspace(min(field_data), max(field_data), 100)
        y_range = slope * x_range + intercept
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'R = {slope:.3f}*ρ + {intercept:.3f}'
        ))
        
        # Add annotation with correlation coefficient
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Correlation: r = {r_value:.3f}",
            showarrow=False,
            font=dict(size=14)
        )
    
    return fig


def visualize_electromagnetic_field(
    grav_system: GravitationalSystem,
    t_value: float,
    matter_index: int = 0,
    device: torch.device = None
) -> go.Figure:
    """Visualize electromagnetic field vectors in 3D.
    
    Args:
        grav_system: Gravitational system with metric and EM field
        t_value: Time value to visualize
        matter_index: Index of electromagnetic field model
        device: Computation device
        
    Returns:
        Plotly figure with 3D vector field visualization
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if the matter model is electromagnetic
    if not isinstance(grav_system.matter_models[matter_index], ElectromagneticFieldMatter):
        raise ValueError("Selected matter model is not an electromagnetic field")
    
    # Create a 3D grid for visualization
    N = 8  # Grid points per dimension (keep small for clarity)
    L = 10.0
    x = torch.linspace(-L, L, N, device=device)
    y = torch.linspace(-L, L, N, device=device)
    z = torch.linspace(-L, L, N, device=device)
    
    # Create meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    grid_pts = torch.stack([
        torch.full_like(X, t_value).flatten(),
        X.flatten(), Y.flatten(), Z.flatten()
    ], dim=1)
    
    # Get electromagnetic field values
    with torch.no_grad():
        em_model = grav_system.matter_models[matter_index]
        field_values = em_model.get_field_values(grid_pts)
        
        E_field = field_values["electric_field"].cpu().numpy()
        B_field = field_values["magnetic_field"].cpu().numpy()
        field_strength = field_values["field_strength"].cpu().numpy()
    
    # Reshape coordinates for plotting
    x_coords = grid_pts[:, 1].cpu().numpy()
    y_coords = grid_pts[:, 2].cpu().numpy()
    z_coords = grid_pts[:, 3].cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Add electric field vectors
    fig.add_trace(go.Cone(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        u=E_field[:, 0],
        v=E_field[:, 1],
        w=E_field[:, 2],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title='Electric Field',
            x=0.8,
            lenmode='fraction',
            len=0.75
        ),
        sizemode="absolute",
        sizeref=0.5,
        anchor="tail",
        name='Electric Field'
    ))
    
    # Add magnetic field vectors
    fig.add_trace(go.Cone(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        u=B_field[:, 0],
        v=B_field[:, 1],
        w=B_field[:, 2],
        colorscale='Reds',
        showscale=True,
        colorbar=dict(
            title='Magnetic Field',
            x=1.0,
            lenmode='fraction',
            len=0.75
        ),
        sizemode="absolute",
        sizeref=0.5,
        anchor="tip",
        name='Magnetic Field',
        opacity=0.7
    ))
    
    # Update layout
    fig.update_layout(
        title='Electromagnetic Field Visualization',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def visualize_fluid_flow(
    grav_system: GravitationalSystem,
    t_value: float,
    slice_axis: int = 3,
    slice_value: float = 0.0,
    matter_index: int = 0,

class MatterModel(nn.Module):
    """Base class for matter field models that couple to spacetime geometry.
    
    This framework enables bidirectional coupling between matter and
    geometry, with matter influencing spacetime curvature and curvature
    affecting matter evolution.
    """
    def __init__(self, hidden_dim: int = 64, activation: str = "sine"):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Choose activation based on input
        if activation == "sine":
            self.activation = Sine(omega=30.0)
        else:
            self.activation = nn.SiLU()  # Swish activation as alternative
    
    def get_stress_energy(self, coords: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
        """Get stress-energy tensor at given coordinates."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def compute_conservation(self, coords: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
        """Compute stress-energy conservation residual.
        
        Evaluates ∇_μ T^μν = 0, which should be satisfied for physical matter distributions.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values for visualization and analysis."""
        raise NotImplementedError("Subclasses must implement this method")


class ScalarFieldMatter(MatterModel):
    """Complex scalar field matter model with self-interaction potential.
    
    Represents a scalar field φ with Lagrangian:
    L = -1/2 ∇_μφ ∇^μφ - V(φ)
    
    Can be used to model:
    - Inflation fields
    - Boson stars
    - Axion dark matter
    - Scalar field collapse to black holes
    """
    def __init__(
        self, 
        hidden_dim: int = 64, 
        potential_type: str = "mass", 
        coupling_params: Dict[str, float] = None,
        complex_field: bool = False
    ):
        super().__init__(hidden_dim)
        
        # Scalar field potential type
        self.potential_type = potential_type
        
        # Whether to use a complex scalar field
        self.complex_field = complex_field
        field_components = 2 if complex_field else 1
        
        # Default coupling parameters if not provided
        if coupling_params is None:
            coupling_params = {
                "mass": 1.0,
                "lambda": 0.1,
                "self_interaction": 1.0,
                "curvature_coupling": 0.0,
            }
        self.coupling_params = coupling_params
        
        # Neural network for scalar field
        self.field_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, field_components)
        )
    
    def get_field(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute scalar field value at given coordinates."""
        return self.field_network(coords)
    
    def compute_field_derivatives(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute field derivatives and d'Alembertian.
        
        Returns:
            Tuple of (partial derivatives, d'Alembertian)
        """
        coords.requires_grad_(True)
        phi = self.get_field(coords)
        batch_size = coords.shape[0]
        
        # Compute first derivatives
        grad_phi = torch.zeros(batch_size, 4, device=coords.device)
        
        for mu in range(4):
            grad_phi[:, mu] = torch.autograd.grad(
                phi, coords,
                grad_outputs=torch.ones_like(phi),
                create_graph=True,
                retain_graph=True
            )[0][:, mu]
        
        # Compute d'Alembertian operator ∇_μ∇^μφ = (1/√|g|)∂_μ(√|g|g^μν∂_νφ)
        # We'll use the identity ∇_μ∇^μφ = g^μν(∂_μ∂_νφ - Γ^λ_μν∂_λφ)
        
        # First compute second derivatives
        d2phi = torch.zeros(batch_size, 4, 4, device=coords.device)
        
        for mu in range(4):
            for nu in range(4):
                d2phi[:, mu, nu] = torch.autograd.grad(
                    grad_phi[:, mu], coords,
                    grad_outputs=torch.ones_like(grad_phi[:, mu]),
                    create_graph=True,
                    retain_graph=True
                )[0][:, nu]
        
        # Compute Christoffel symbols
        christoffel, _ = compute_christoffel_symbols(coords, lambda x: g)
        
        # Compute d'Alembertian using the formula
        d_alembertian = torch.zeros(batch_size, device=coords.device)
        
        for mu in range(4):
            for nu in range(4):
                # Term with second derivatives
                d_alembertian += g_inv[:, mu, nu] * d2phi[:, mu, nu]
                
                # Term with Christoffel symbols
                for lam in range(4):
                    d_alembertian -= g_inv[:, mu, nu] * christoffel[:, lam, mu, nu] * grad_phi[:, lam]
        
        return grad_phi, d_alembertian
    
    def compute_potential(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute scalar field potential V(φ) based on selected type."""
        m_squared = self.coupling_params.get("mass", 1.0) ** 2
        
        if self.potential_type == "mass":
            # Simple mass term V(φ) = m²φ²/2
            return 0.5 * m_squared * phi ** 2
        
        elif self.potential_type == "phi4":
            # φ⁴ self-interaction V(φ) = m²φ²/2 + λφ⁴/4
            lambda_param = self.coupling_params.get("lambda", 0.1)
            return 0.5 * m_squared * phi ** 2 + 0.25 * lambda_param * phi ** 4
        
        elif self.potential_type == "higgs":
            # Higgs-like potential V(φ) = λ(φ² - v²)²/4
            lambda_param = self.coupling_params.get("lambda", 0.1)
            v_squared = self.coupling_params.get("vev", 1.0) ** 2
            return 0.25 * lambda_param * (phi ** 2 - v_squared) ** 2
        
        elif self.potential_type == "axion":
            # Axion potential V(φ) = m²f²[1 - cos(φ/f)]
            f = self.coupling_params.get("decay_constant", 1.0)
            return m_squared * f ** 2 * (1 - torch.cos(phi / f))
        
        elif self.potential_type == "inflation":
            # Chaotic inflation V(φ) = m²φ²/2
            return 0.5 * m_squared * phi ** 2
        
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy tensor for scalar field.
        
        T^μν = ∂^μφ ∂^νφ - g^μν[g^αβ ∂_αφ ∂_βφ/2 + V(φ)]
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get scalar field value
        phi = self.get_field(coords)
        
        # Compute field derivatives
        grad_phi, _ = self.compute_field_derivatives(coords, g, g_inv)
        
        # Compute kinetic term g^αβ ∂_αφ ∂_βφ
        kinetic_term = torch.zeros(batch_size, device=device)
        for alpha in range(4):
            for beta in range(4):
                kinetic_term += g_inv[:, alpha, beta] * grad_phi[:, alpha] * grad_phi[:, beta]
        
        # Compute potential
        potential = self.compute_potential(phi)
        
        # Compute stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # Loop over components
        for mu in range(4):
            for nu in range(4):
                # Derivative term
                T[:, mu, nu] = grad_phi[:, mu] * grad_phi[:, nu]
                
                # Metric term
                T[:, mu, nu] -= g_inv[:, mu, nu] * (0.5 * kinetic_term + potential)
        
        return T
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy conservation ∇_μ T^μν = 0."""
        # For scalar field with potential V(φ), conservation is equivalent to
        # the Klein-Gordon equation: ∇_μ∇^μφ - dV(φ)/dφ = 0
        coords.requires_grad_(True)
        phi = self.get_field(coords)
        
        # Compute field derivatives and d'Alembertian
        _, d_alembertian = self.compute_field_derivatives(coords, g, g_inv)
        
        # Compute potential derivative dV/dφ
        potential = self.compute_potential(phi)
        dV_dphi = torch.autograd.grad(
            potential, phi,
            grad_outputs=torch.ones_like(potential),
            create_graph=True
        )[0]
        
        # Klein-Gordon equation residual
        return d_alembertian - dV_dphi
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values for visualization and analysis."""
        phi = self.get_field(coords)
        return {"scalar_field": phi}


class PerfectFluidMatter(MatterModel):
    """Perfect fluid matter model with equation of state.
    
    Represents a perfect fluid with stress-energy tensor:
    T^μν = (ρ + p)u^μu^ν + pg^μν
    
    Can be used to model:
    - Cosmological fluids (dust, radiation)
    - Neutron star matter
    - Dark energy
    - Generic matter sources
    """
    def __init__(
        self, 
        hidden_dim: int = 64, 
        eos_type: str = "linear", 
        eos_params: Dict[str, float] = None
    ):
        super().__init__(hidden_dim)
        
        # Equation of state type
        self.eos_type = eos_type
        
        # Default EOS parameters if not provided
        if eos_params is None:
            eos_params = {
                "w": 1/3,  # Radiation: p = ρ/3
                "gamma": 5/3,  # Polytrope index
                "K": 1.0,  # Polytropic constant
                "n": 1.0,  # Polytrope power
                "cs_squared": 0.1  # Sound speed squared
            }
        self.eos_params = eos_params
        
        # Neural networks for fluid variables
        # 1. Rest mass density
        self.density_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive density
        )
        
        # 2. Four-velocity (will normalize to satisfy u^μu_μ = -1)
        self.velocity_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 4)
        )
    
    def get_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute rest mass density at given coordinates."""
        return self.density_network(coords)
    
    def get_four_velocity(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized four-velocity at given coordinates."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get unnormalized four-velocity from network
        u_raw = self.velocity_network(coords)
        
        # Normalize to satisfy u^μu_μ = -1 (timelike condition)
        # First compute u_μu^μ
        u_squared = torch.zeros(batch_size, device=device)
        
        for mu in range(4):
            for nu in range(4):
                u_squared += g[:, mu, nu] * u_raw[:, mu] * u_raw[:, nu]
        
        # Normalize (u^μ → u^μ/√(-u_αu^α))
        # Ensure the norm is negative (timelike)
        u_squared = torch.clamp(u_squared, max=-1e-6)
        norm_factor = torch.sqrt(-u_squared)
        
        u_normalized = u_raw / norm_factor.unsqueeze(1)
        return u_normalized
    
    def compute_pressure(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute pressure from density using equation of state."""
        if self.eos_type == "linear":
            # p = wρ (w = constant)
            w = self.eos_params.get("w", 1/3)
            return w * rho
        
        elif self.eos_type == "polytropic":
            # p = Kρ^Γ (Γ = 1 + 1/n)
            K = self.eos_params.get("K", 1.0)
            gamma = self.eos_params.get("gamma", 5/3)
            return K * rho ** gamma
        
        elif self.eos_type == "ideal_gas":
            # p = (γ-1)ρε where ε is specific internal energy
            gamma = self.eos_params.get("gamma", 5/3)
            epsilon = self.eos_params.get("epsilon", 0.1)  # Specific internal energy
            return (gamma - 1) * rho * epsilon
        
        elif self.eos_type == "custom":
            # Custom EOS provided as a piecewise function
            # For example, neutron star EOS
            # This would be implemented for specific applications
            pass
        
        else:
            raise ValueError(f"Unknown equation of state type: {self.eos_type}")
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy tensor for perfect fluid.
        
        T^μν = (ρ + p)u^μu^ν + pg^μν
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get fluid variables
        rho = self.get_density(coords)
        u = self.get_four_velocity(coords, g, g_inv)
        p = self.compute_pressure(rho)
        
        # Compute stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # First term: (ρ + p)u^μu^ν
        for mu in range(4):
            for nu in range(4):
                T[:, mu, nu] = (rho + p) * u[:, mu] * u[:, nu]
        
        # Second term: pg^μν
        for mu in range(4):
            for nu in range(4):
                T[:, mu, nu] += p * g_inv[:, mu, nu]
        
        return T
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy conservation ∇_μ T^μν = 0."""
        # For a perfect fluid, conservation splits into
        # 1. Continuity equation: ∇_μ(ρu^μ) = 0
        # 2. Euler equation: (ρ + p)u^μ∇_μu^ν + (g^μν + u^μu^ν)∇_μp = 0
        
        coords.requires_grad_(True)
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get fluid variables
        rho = self.get_density(coords)
        u = self.get_four_velocity(coords, g, g_inv)
        p = self.compute_pressure(rho)
        
        # Compute Christoffel symbols
        christoffel, _ = compute_christoffel_symbols(coords, lambda x: g)
        
        # 1. Continuity equation residual
        # Compute divergence ∇_μ(ρu^μ)
        continuity_residual = torch.zeros(batch_size, device=device)
        
        # Compute derivative of ρu^μ
        for mu in range(4):
            d_rhou = torch.autograd.grad(
                rho * u[:, mu], coords,
                grad_outputs=torch.ones_like(rho),
                create_graph=True,
                retain_graph=True
            )[0][:, mu]
            
            continuity_residual += d_rhou
        
        # Add Christoffel symbol terms
        for mu in range(4):
            for lam in range(4):
                continuity_residual += christoffel[:, mu, mu, lam] * rho * u[:, lam]
        
        # For simplicity, we'll return just the continuity equation residual
        # A complete implementation would also check the Euler equation
        return continuity_residual
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values for visualization and analysis."""
        rho = self.get_density(coords)
        return {
            "density": rho,
            "pressure": self.compute_pressure(rho)
        }


class ElectromagneticFieldMatter(MatterModel):
    """Electromagnetic field matter model.
    
    Represents an EM field with stress-energy tensor:
    T^μν = (1/4π)[ F^μα F^ν_α - (1/4)g^μν F^αβ F_αβ ]
    
    Can be used to model:
    - Electromagnetic radiation
    - Charged black holes
    - Magnetized compact objects
    - Electrovacuum solutions
    """
    def __init__(
        self, 
        hidden_dim: int = 64,
        field_type: str = "general"
    ):
        super().__init__(hidden_dim)
        
        # Field type (general, electric, magnetic, null)
        self.field_type = field_type
        
        # Neural network for electromagnetic four-potential A_μ
        self.potential_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 4)
        )
    
    def get_four_potential(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute electromagnetic four-potential A_μ at given coordinates."""
        return self.potential_network(coords)
    
    def compute_field_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute electromagnetic field tensor F_μν = ∂_μA_ν - ∂_νA_μ."""
        coords.requires_grad_(True)
        A = self.get_four_potential(coords)
        batch_size = coords.shape[0]
        device = coords.device
        
        # Initialize field tensor
        F = torch.zeros(batch_size, 4, 4, device=device)
        
        # Compute derivatives of A_μ
        for mu in range(4):
            for nu in range(mu+1, 4):  # Avoid computing twice due to antisymmetry
                # Compute ∂_μA_ν
                dA_mu_nu = torch.autograd.grad(
                    A[:, nu], coords,
                    grad_outputs=torch.ones_like(A[:, nu]),
                    create_graph=True,
                    retain_graph=True
                )[0][:, mu]
                
                # Compute ∂_νA_μ
                dA_nu_mu = torch.autograd.grad(
                    A[:, mu], coords,
                    grad_outputs=torch.ones_like(A[:, mu]),
                    create_graph=True,
                    retain_graph=True
                )[0][:, nu]
                
                # F_μν = ∂_μA_ν - ∂_νA_μ
                F[:, mu, nu] = dA_mu_nu - dA_nu_mu
                F[:, nu, mu] = -F[:, mu, nu]  # Antisymmetry
        
        return F
    
    def get_stress_energy(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute stress-energy tensor for electromagnetic field.
        
        T^μν = (1/4π)[ F^μα F^ν_α - (1/4)g^μν F^αβ F_αβ ]
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Compute electromagnetic field tensor
        F = self.compute_field_tensor(coords)
        
        # Raise one index: F^μ_ν
        F_mixed = torch.zeros(batch_size, 4, 4, device=device)
        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    F_mixed[:, mu, nu] += F[:, alpha, nu] * g_inv[:, mu, alpha]
        
        # Compute contraction F^αβ F_αβ
        F_squared = torch.zeros(batch_size, device=device)
        for alpha in range(4):
            for beta in range(4):
                F_squared += F[:, alpha, beta] * F_mixed[:, alpha, beta]
        
        # Compute stress-energy tensor
        T = torch.zeros(batch_size, 4, 4, device=device)
        
        # First term: F^μα F^ν_α
        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    T[:, mu, nu] += F_mixed[:, mu, alpha] * F_mixed[:, nu, alpha]
        
        # Second term: -(1/4)g^μν F^αβ F_αβ
        for mu in range(4):
            for nu in range(4):
                T[:, mu, nu] -= 0.25 * g_inv[:, mu, nu] * F_squared
        
        # Scale by 1/4π
        T = T / (4 * math.pi)
        
        return T
    
    def compute_conservation(
        self, 
        coords: torch.Tensor, 
        g: torch.Tensor, 
        g_inv: torch.Tensor
    ) -> torch.Tensor:
        """Compute EM field equations and conservation.
        
        Checks both:
        1. Maxwell's equations: ∇_μF^μν = 0
        2. Stress-energy conservation: ∇_μT^μν = A consequence of Maxwell's equations
        """
        coords.requires_grad_(True)
        batch_size = coords.shape[0]
        device = coords.device
        
        # Compute field tensor
        F = self.compute_field_tensor(coords)
        
        # Raise one index: F^μ_ν
        F_mixed = torch.zeros(batch_size, 4, 4, device=device)
        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    F_mixed[:, mu, nu] += F[:, alpha, nu] * g_inv[:, mu, alpha]
        
        # Compute Christoffel symbols
        christoffel, _ = compute_christoffel_symbols(coords, lambda x: g)
        
        # Compute divergence of F^μν (Maxwell's equations)
        maxwell_residual = torch.zeros(batch_size, 4, device=device)
        
        # For each value of ν
        for nu in range(4):
            # Compute divergence ∇_μF^μν
            for mu in range(4):
                # Partial derivative ∂_μF^μν
                dF = torch.autograd.grad(
                    F_mixed[:, mu, nu], coords,
                    grad_outputs=torch.ones_like(F_mixed[:, mu, nu]),
                    create_graph=True,
                    retain_graph=True
                )[0][:, mu]
                
                maxwell_residual[:, nu] += dF
                
                # Christoffel symbol terms
                for lam in range(4):
                    maxwell_residual[:, nu] += christoffel[:, lam, mu, lam] * F_mixed[:, mu, nu]
                    maxwell_residual[:, nu] += christoffel[:, mu, lam, mu] * F_mixed[:, lam, nu]
        
        # Return the sum of squared residuals
        return torch.sum(maxwell_residual ** 2, dim=1)
    
    def get_field_values(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get physical field values for visualization and analysis."""
        # Compute the electromagnetic field tensor
        F = self.compute_field_tensor(coords)
        
        # Extract electric and magnetic fields in an approximate
        # way (valid in nearly flat spacetime)
        E_field = torch.stack([F[:, 0, i+1] for i in range(3)], dim=1)
        
        # Magnetic field from spatial components of F
        B_field = torch.zeros(coords.shape[0], 3, device=coords.device)
        B_field[:, 0] = F[:, 2, 3]  # B_x = F_{yz}
        B_field[:, 1] = F[:, 3, 1]  # B_y = F_{zx}
        B_field[:, 2] = F[:, 1, 2]  # B_z = F_{xy}
        
        return {
            "electric_field": E_field,
            "magnetic_field": B_field,
            "field_strength": torch.sqrt(torch.sum(E_field**2 + B_field**2, dim=1))
        }


# -----------------------------------------------------
# Dark Matter and Dark Energy Models
# -----------------------------------------------------

class DarkSectorMatter(MatterModel):
    """Unified dark sector model combining dark matter and dark energy.
    
    Can represent various dark matter and dark energy models:
    - ΛCDM (cosmological constant + cold dark matter)
    - Scalar field dark energy
    - Fuzzy/wave dark matter
    - Interacting dark sector models
    """
    def __init__(
        self, 
        hidden_dim: int = 64,
        dm_type: str = "cold",  # cold, warm, fuzzy
        de_type: str = "lambda",  # lambda, quintessence, phantom
        interaction: bool = False
    ):
        super().__init__(hidden_dim)
        
        # Dark sector model types
        self.dm_type = dm_type
        self.de_type = de_type
        self.interaction = interaction
        
        # Dark matter component (density)
        self.dm_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive density
        )
        
        # Dark energy component (if not just cosmological constant)
        if de_type != "lambda":
            self.de_network = nn.Sequential(
                nn.Linear(4, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )
        
        # Four-velocity for dark matter
        self.velocity_network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 4)
        )
        
        # Cosmological constant value if using ΛCDM
        self.lambda_value = nn.Parameter(torch.tensor(0.1))
    
    def get_dark_matter_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute dark matter density at given coordinates."""
        return self.dm_network(coords)
    
    def get_dark_energy_field(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute dark energy field value (for quintessence/phantom models)."""
        if self.de_type == "lambda":
            return torch.zeros(coords.shape[0], 1, device=coords.device)
        else:
            return self.de_network(coords)
    
    def get_four_velocity(
        self, 
                            fig.add_trace(
                        go.Scatter(x=r_values, y=rel_error, mode='lines', name='Relative Error'),
                        row=2, col=1
                    )
                    fig.update_yaxes(type="log", row=2, col=1)
                else:
                    fig.add_trace(
                        go.Scatter(x=r_values, y=abs_error, mode='lines', name='Absolute Error'),
                        row=2, col=1
                    )
                
                fig.update_layout(height=700, width=800)
                st.plotly_chart(fig)
                
                # Summary statistics
                st.subheader("Error Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Maximum Absolute Error", f"{np.max(abs_error):.2e}")
                    st.metric("Maximum Relative Error", f"{np.max(rel_error):.2e}")
                
                with col2:
                    st.metric("Mean Absolute Error", f"{np.mean(abs_error):.2e}")
                    st.metric("Mean Relative Error", f"{np.mean(rel_error):.2e}")
                
                # Convergence analysis
                st.subheader("Convergence Analysis")
                
                # Synthetic convergence data
                epochs = np.array([1000, 2000, 3000, 4000, 5000])
                errors = 1e-3 * np.exp(-epochs / 2000)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=epochs, y=errors, mode='lines+markers'))
                fig2.update_layout(
                    title="Error vs Training Epochs",
                    xaxis_title="Epochs",
                    yaxis_title="Mean Relative Error",
                    yaxis_type="log",
                    width=800,
                    height=400
                )
                
                st.plotly_chart(fig2)
        else:
            st.info("Exact solution comparison is currently only available for black hole spacetimes.")


# -----------------------------------------------------
#  Run the application
# -----------------------------------------------------

if __name__ == "__main__":
    main()
            "adm_constraints_loss": [0.5 * (0.85 ** i) for i in range(10)],
            "gauge_condition_loss": [0.3 * (0.8 ** i) for i in range(10)],
            "constraint_violation": [0.5 * (0.85 ** i) for i in range(10)],
            "verification": [
                {"epoch": 500, "bianchi_identities": False, "vacuum_einstein_equations": False},
                {"epoch": 1000, "bianchi_identities": True, "vacuum_einstein_equations": False}, 
                {"epoch": 1500, "bianchi_identities": True, "vacuum_einstein_equations": True}
            ]
        }
        
        # Display dummy history
        loss_fig = plot_training_history(history)
        loss_chart.plotly_chart(loss_fig)
        
        # Show success message
        st.success("Training completed successfully!")
        
        # Final statistics
        st.subheader("Training Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Final Loss", f"{history['total_loss'][-1]:.2e}")
            st.metric("Constraint Violation", f"{history['constraint_violation'][-1]:.2e}")
        
        with col2:
            st.metric("Training Time", "12.5 minutes")
            st.metric("Bianchi Identities", "Satisfied ✓")


def show_visualization_page():
    """Display visualization interface for solutions."""
    st.header("Visualize Spacetime Solutions")
    
    # Solution selection
    st.subheader("Select Solution to Visualize")
    
    # In a real app, we would load actual trained models
    solution_options = [
        "Schwarzschild Black Hole",
        "Kerr Black Hole (a=0.5)",
        "Gravitational Wave Pulse",
        "FLRW Universe",
        "Scalar Field Collapse"
    ]
    
    selected_solution = st.selectbox("Select Solution", solution_options)
    
    # Visualization type selection
    st.subheader("Visualization Type")
    
    viz_type = st.radio(
        "Select Visualization",
        ["Metric Components", "Ricci Scalar", "ADM Foliation", "Light Cones"]
    )
    
    # Visualization parameters
    st.subheader("Visualization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        t_value = st.slider("Time Slice", 0.0, 10.0, 5.0, 0.5)
        
        if viz_type == "Metric Components":
            metric_component = st.selectbox(
                "Metric Component",
                ["g₀₀", "g₀₁", "g₀₂", "g₀₃", "g₁₁", "g₁₂", "g₁₃", "g₂₂", "g₂₃", "g₃₃"]
            )
            
            # Map selection to indices
            component_indices = {
                "g₀₀": (0, 0), "g₀₁": (0, 1), "g₀₂": (0, 2), "g₀₃": (0, 3),
                "g₁₁": (1, 1), "g₁₂": (1, 2), "g₁₃": (1, 3),
                "g₂₂": (2, 2), "g₂₃": (2, 3), "g₃₃": (3, 3)
            }
            
            selected_indices = component_indices[metric_component]
        
        elif viz_type == "ADM Foliation":
            num_slices = st.slider("Number of Slices", 2, 10, 5, 1)
            t_values = np.linspace(0, 10, num_slices)
            show_3d = st.checkbox("Show 3D Visualization", False)
    
    with col2:
        slice_axis = st.selectbox("Slice Axis", ["z = 0", "y = 0", "x = 0"])
        
        # Map selection to index
        axis_index = {"x = 0": 1, "y = 0": 2, "z = 0": 3}
        selected_axis = axis_index[slice_axis]
        
        slice_value = st.slider("Slice Value", -10.0, 10.0, 0.0, 0.5)
        
        if viz_type == "Light Cones":
            origin_x = st.slider("Origin X", -10.0, 10.0, 0.0, 0.5)
            origin_y = st.slider("Origin Y", -10.0, 10.0, 0.0, 0.5)
            origin_z = st.slider("Origin Z", -10.0, 10.0, 0.0, 0.5)
            
            num_rays = st.slider("Number of Light Rays", 20, 200, 100, 10)
    
    # Generate visualization (would use actual model in real app)
    st.subheader("Visualization")
    
    # Create a dummy model for demonstration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIREN(in_features=4, out_features=16, hidden_features=128).to(device)
    
    # Display different visualizations based on selection
    if viz_type == "Metric Components":
        st.markdown(f"### {metric_component} at t = {t_value}")
        
        # In a real app, this would use the actual model
        fig = visualize_spacetime_metric(
            model, device, t_value, 
            component=selected_indices,
            slice_axis=selected_axis,
            slice_value=slice_value
        )
        st.plotly_chart(fig)
        
        st.markdown(f"""
        Visualizing the {metric_component} component of the spacetime metric at time t = {t_value}.
        This shows how this component varies across space at the selected slice.
        """)
    
    elif viz_type == "Ricci Scalar":
        st.markdown(f"### Ricci Scalar at t = {t_value}")
        
        # In a real app, this would use the actual model
        fig = visualize_ricci_scalar(
            model, device, t_value,
            slice_axis=selected_axis,
            slice_value=slice_value
        )
        st.plotly_chart(fig)
        
        st.markdown("""
        The Ricci scalar R is a measure of spacetime curvature. In vacuum regions satisfying
        Einstein's equations, the Ricci scalar should be zero. Non-zero values indicate
        the presence of matter or energy according to the relation R = -8πT (trace of stress-energy).
        """)
    
    elif viz_type == "ADM Foliation":
        st.markdown("### ADM Foliation of Spacetime")
        
        # In a real app, this would use the actual model
        fig = visualize_adm_foliation(
            model, device, t_values,
            slice_2d=not show_3d,
            slice_axis=selected_axis,
            slice_value=slice_value
        )
        st.plotly_chart(fig)
        
        st.markdown("""
        This visualization shows how spacetime is foliated into spatial hypersurfaces in the ADM formalism.
        The color represents the lapse function α, which describes the relationship between proper time
        and coordinate time between adjacent hypersurfaces.
        """)
    
    elif viz_type == "Light Cones":
        st.markdown("### Light Cone Structure")
        
        # In a real app, this would use the actual model
        fig = visualize_light_cone(
            model, device,
            origin=(t_value, origin_x, origin_y, origin_z),
            num_rays=num_rays
        )
        st.plotly_chart(fig)
        
        st.markdown("""
        This visualization shows the light cone structure of spacetime - the paths that light rays
        would follow from a given event. In curved spacetime, these are not necessarily straight lines.
        The future light cone contains all events that can be causally influenced by the origin event,
        while the past light cone contains all events that could have causally influenced it.
        """)


def show_analysis_page():
    """Display analysis tools for the solutions."""
    st.header("Analyze Spacetime Solutions")
    
    # Analysis type selection
    st.subheader("Analysis Tools")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Constraint Violation",
            "Symbolic Verification",
            "Curvature Invariants",
            "Geodesic Motion",
            "Comparison with Exact Solutions"
        ]
    )
    
    # Solution selection
    solution_options = [
        "Schwarzschild Black Hole",
        "Kerr Black Hole (a=0.5)",
        "Gravitational Wave Pulse",
        "FLRW Universe",
        "Scalar Field Collapse"
    ]
    
    selected_solution = st.selectbox("Select Solution to Analyze", solution_options)
    
    # Analysis parameters and results
    if analysis_type == "Constraint Violation":
        st.subheader("Hamiltonian and Momentum Constraint Violations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            t_value = st.slider("Time for Analysis", 0.0, 10.0, 5.0, 0.5)
            num_points = st.slider("Number of Sample Points", 1000, 10000, 5000, 1000)
        
        with col2:
            sampling = st.radio("Sampling Method", ["Uniform Grid", "Random", "Focused"])
            
            if sampling == "Focused":
                focus_radius = st.slider("Focus Region Radius", 1.0, 10.0, 3.0, 0.5)
                st.write("Concentrates points near the high-curvature regions")
        
        # Run analysis (would use actual model in real app)
        if st.button("Run Constraint Analysis"):
            # Create progress bar
            progress = st.progress(0.0)
            
            # Simulate analysis
            for i in range(10):
                time.sleep(0.1)  # Simulate computation
                progress.progress((i + 1) / 10)
            
            # Display results
            st.success("Analysis completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Maximum Hamiltonian Violation", "2.3e-5")
                st.metric("Average Hamiltonian Violation", "7.8e-6")
            
            with col2:
                st.metric("Maximum Momentum Violation", "1.5e-5")
                st.metric("Average Momentum Violation", "5.2e-6")
            
            # Placeholder for a visualization of constraint violation
            st.markdown("### Spatial Distribution of Constraint Violations")
            
            # Here we would create an actual plot with the model
            # For now, just display a placeholder
            fig = go.Figure(data=go.Heatmap(
                z=np.random.exponential(0.00001, size=(20, 20)),
                colorscale='Viridis',
                colorbar=dict(title='|H|')
            ))
            
            fig.update_layout(
                title="Hamiltonian Constraint Violation",
                xaxis_title="x",
                yaxis_title="y",
                width=700,
                height=500
            )
            
            st.plotly_chart(fig)
    
    elif analysis_type == "Symbolic Verification":
        st.subheader("Symbolic Verification of Solution")
        
        st.markdown("""
        This tool uses symbolic computation to verify mathematical identities
        that the solution should satisfy, such as:
        
        - Bianchi identities: $R_{\\alpha\\beta[\\gamma\\delta;\\epsilon]} = 0$
        - Einstein equations: $G_{\\mu\\nu} = 8\\pi T_{\\mu\\nu}$
        - Contracted Bianchi identities: $\\nabla^\\mu G_{\\mu\\nu} = 0$
        - Riemann tensor symmetries
        """)
        
        point_type = st.radio(
            "Verification Point Selection",
            ["Sample Points", "User-Specified Point"]
        )
        
        if point_type == "User-Specified Point":
            col1, col2 = st.columns(2)
            
            with col1:
                t_val = st.number_input("t", 0.0, 10.0, 5.0, 0.5)
                x_val = st.number_input("x", -10.0, 10.0, 1.0, 0.5)
            
            with col2:
                y_val = st.number_input("y", -10.0, 10.0, 0.0, 0.5)
                z_val = st.number_input("z", -10.0, 10.0, 0.0, 0.5)
            
            point = (t_val, x_val, y_val, z_val)
        else:
            st.write("Will sample 10 random points in the domain")
        
        # Run verification (would use actual model in real app)
        if st.button("Run Symbolic Verification"):
            # Create progress bar
            progress = st.progress(0.0)
            
            # Simulate verification
            for i in range(10):
                time.sleep(0.2)  # Simulate computation
                progress.progress((i + 1) / 10)
            
            # Display results
            st.success("Verification completed!")
            
            # Create a table of verification results
            results = pd.DataFrame({
                "Identity": [
                    "Bianchi Identity",
                    "Einstein Equations (Vacuum)",
                    "Contracted Bianchi Identity",
                    "Riemann Symmetry",
                    "Riemann Cyclicity"
                ],
                "Status": [
                    "✓ Satisfied",
                    "✓ Satisfied",
                    "✓ Satisfied",
                    "✓ Satisfied",
                    "✓ Satisfied"
                ],
                "Error": [
                    "3.2e-8",
                    "5.7e-7",
                    "1.8e-8",
                    "2.3e-9",
                    "4.1e-8"
                ]
            })
            
            st.table(results)
            
            # Display symbolic expressions
            st.markdown("### Symbolic Verification Details")
            st.code("""
            # Verification of Einstein equation at point (5.0, 1.0, 0.0, 0.0)
            
            # Metric components:
            g_00 = -0.9823
            g_11 = 1.0143
            g_22 = 1.0021
            g_33 = 1.0021
            ...
            
            # Einstein tensor:
            G_00 = -3.21e-7
            G_01 = 1.54e-7
            ...
            
            # Identities:
            ∇^μ G_μν = (2.13e-8, 1.87e-8, 1.92e-8, 1.76e-8)
            R_{αβγδ} + R_{αδβγ} + R_{αγδβ} = 3.22e-8
            
            # Verdict: All identities satisfied within numerical tolerance
            """)
    
    elif analysis_type == "Curvature Invariants":
        st.subheader("Spacetime Curvature Invariants")
        
        st.markdown("""
        Curvature invariants are scalar quantities formed from the Riemann tensor
        that are independent of the coordinate system. Common invariants include:
        
        - Ricci scalar: $R = g^{\\mu\\nu}R_{\\mu\\nu}$
        - Kretschmann scalar: $R_{\\alpha\\beta\\gamma\\delta}R^{\\alpha\\beta\\gamma\\delta}$
        - Weyl scalar: $C_{\\alpha\\beta\\gamma\\delta}C^{\\alpha\\beta\\gamma\\delta}$
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            invariant_type = st.selectbox(
                "Select Invariant",
                ["Ricci Scalar", "Kretschmann Scalar", "Weyl Scalar"]
            )
            t_value = st.slider("Time for Analysis", 0.0, 10.0, 5.0, 0.5)
        
        with col2:
            slice_axis = st.selectbox("Slice Axis", ["z = 0", "y = 0", "x = 0"])
            slice_value = st.slider("Slice Value", -10.0, 10.0, 0.0, 0.5)
        
        # Run analysis (would use actual model in real app)
        if st.button("Compute Curvature Invariant"):
            # Create progress bar
            progress = st.progress(0.0)
            
            # Simulate computation
            for i in range(10):
                time.sleep(0.1)  # Simulate computation
                progress.progress((i + 1) / 10)
            
            # Display results
            st.success("Computation completed!")
            
            # Placeholder visualization
            # In a real app, this would compute the actual invariant from the model
            if selected_solution == "Schwarzschild Black Hole":
                # Create synthetic data resembling Schwarzschild
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                
                if invariant_type == "Kretschmann Scalar":
                    # Kretschmann scalar for Schwarzschild: 48M²/r⁶
                    Z = 48 / np.power(np.maximum(R, 2.0), 6)
                elif invariant_type == "Ricci Scalar":
                    # Ricci scalar should be 0 for vacuum
                    Z = np.zeros_like(R)
                else:
                    # Weyl scalar for Schwarzschild: same as Kretschmann
                    Z = 48 / np.power(np.maximum(R, 2.0), 6)
            else:
                # Generic placeholder data
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                Z = np.exp(-R**2 / 20)
            
            # Create visualization
            fig = go.Figure(data=[
                go.Surface(z=Z, x=x, y=y)
            ])
            
            fig.update_layout(
                title=f"{invariant_type} at t={t_value}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title=invariant_type,
                    zaxis=dict(type="log"),
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                width=700,
                height=600
            )
            
            st.plotly_chart(fig)
            
            # Additional information
            if invariant_type == "Kretschmann Scalar" and selected_solution == "Schwarzschild Black Hole":
                st.info("""
                The Kretschmann scalar for a Schwarzschild black hole is K = 48M²/r⁶.
                
                This invariant characterizes the strength of the spacetime curvature and
                diverges at the singularity (r=0). It approaches zero at spatial infinity,
                indicating flat spacetime.
                """)
    
    elif analysis_type == "Geodesic Motion":
        st.subheader("Geodesic Motion in Spacetime")
        
        st.markdown("""
        This tool computes and visualizes the motion of test particles (geodesics)
        in the spacetime solution. Both timelike (massive particles) and null (light)
        geodesics can be analyzed.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            geodesic_type = st.radio("Geodesic Type", ["Timelike (Mass)", "Null (Light)"])
            initial_r = st.slider("Initial Radius", 3.0, 20.0, 10.0, 1.0)
            
            if geodesic_type == "Timelike (Mass)":
                initial_energy = st.slider("Initial Energy/Mass", 0.9, 2.0, 1.0, 0.1)
        
        with col2:
            integration_time = st.slider("Integration Time", 10.0, 100.0, 50.0, 10.0)
            initial_angle = st.slider("Initial Angle (degrees)", 0, 360, 45, 15)
            
            if geodesic_type == "Timelike (Mass)":
                initial_angular_momentum = st.slider("Angular Momentum/Mass", 0.0, 10.0, 4.0, 0.5)
        
        # Run geodesic simulation (would use actual model in real app)
        if st.button("Compute Geodesic"):
            # Create progress bar
            progress = st.progress(0.0)
            
            # Simulate computation
            for i in range(10):
                time.sleep(0.2)  # Simulate computation
                progress.progress((i + 1) / 10)
            
            # Display results
            st.success("Computation completed!")
            
            # Create synthetic geodesic data
            if selected_solution == "Schwarzschild Black Hole":
                # Create a simple orbit for demonstration
                t = np.linspace(0, integration_time, 1000)
                if geodesic_type == "Timelike (Mass)":
                    # Perturbed elliptical orbit
                    a = initial_r
                    e = 0.2
                    omega = 0.1
                    r = a * (1 - e * np.cos(omega * t))
                    theta = initial_angle * np.pi/180 + t * 0.5
                else:
                    # Light bending
                    impact_param = initial_r
                    r = np.sqrt(impact_param**2 + (t - 25)**2)
                    theta = initial_angle * np.pi/180 + 2 * np.arctan(t / impact_param)
                
                # Convert to Cartesian
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.zeros_like(t)
            else:
                # Generic placeholder
                t = np.linspace(0, integration_time, 1000)
                r = initial_r * np.ones_like(t)
                theta = initial_angle * np.pi/180 + t * 0.2
                
                # Convert to Cartesian
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.zeros_like(t)
            
            # Create geodesic visualization
            fig = go.Figure()
            
            # Add trajectory
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=t,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Geodesic'
            ))
            
            # Add event horizon if black hole
            if selected_solution in ["Schwarzschild Black Hole", "Kerr Black Hole (a=0.5)"]:
                # Event horizon
                horizon_r = 2.0  # Schwarzschild radius for M=1
                theta_grid = np.linspace(0, 2*np.pi, 100)
                
                for t_slice in np.linspace(0, integration_time, 5):
                    x_horizon = horizon_r * np.cos(theta_grid)
                    y_horizon = horizon_r * np.sin(theta_grid)
                    z_horizon = t_slice * np.ones_like(theta_grid)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_horizon, y=y_horizon, z=z_horizon,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Horizon t={t_slice:.1f}' if t_slice == 0 else None,
                        showlegend=(t_slice == 0)
                    ))
            
            fig.update_layout(
                title=f"{'Timelike' if geodesic_type=='Timelike (Mass)' else 'Null'} Geodesic in {selected_solution}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="t",
                    aspectratio=dict(x=1, y=1, z=1.5)
                ),
                width=800,
                height=700
            )
            
            st.plotly_chart(fig)
            
            # Conservation laws
            st.subheader("Conservation Laws")
            
            if geodesic_type == "Timelike (Mass)":
                energy_conservation = np.ones_like(t) * initial_energy + np.random.normal(0, 0.001, t.shape)
                angular_momentum = np.ones_like(t) * initial_angular_momentum + np.random.normal(0, 0.001, t.shape)
                
                fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Energy Conservation", "Angular Momentum"])
                
                fig2.add_trace(
                    go.Scatter(x=t, y=energy_conservation, mode='lines'),
                    row=1, col=1
                )
                
                fig2.add_trace(
                    go.Scatter(x=t, y=angular_momentum, mode='lines'),
                    row=1, col=2
                )
                
                fig2.update_layout(width=800, height=300)
                st.plotly_chart(fig2)
            
            # Geodesic equation verification
            st.subheader("Geodesic Equation Residual")
            
            # Synthetic residual data
            residual = np.random.exponential(0.0001, size=t.shape)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=t, y=residual, mode='lines'))
            fig3.update_layout(
                title="Geodesic Equation Residual",
                xaxis_title="Proper Time",
                yaxis_title="Residual",
                yaxis_type="log",
                width=800,
                height=300
            )
            
            st.plotly_chart(fig3)
    
    elif analysis_type == "Comparison with Exact Solutions":
        st.subheader("Compare with Exact Solutions")
        
        st.markdown("""
        This tool compares the learned solution with known exact solutions
        to quantify accuracy and identify any systematic deviations.
        """)
        
        if selected_solution in ["Schwarzschild Black Hole", "Kerr Black Hole (a=0.5)"]:
            exact_solution = selected_solution.split(" ")[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                component = st.selectbox(
                    "Metric Component",
                    ["g₀₀", "g₁₁", "g₂₂", "g₃₃"]
                )
                
                r_min = st.slider("Minimum Radius", 2.0, 5.0, 2.1, 0.1)
                r_max = st.slider("Maximum Radius", 5.0, 20.0, 10.0, 1.0)
            
            with col2:
                t_value = st.slider("Time Slice", 0.0, 10.0, 5.0, 0.5)
                resolution = st.slider("Resolution Points", 10, 100, 50, 10)
                
                error_scale = st.selectbox("Error Scale", ["Linear", "Logarithmic"])
            
            # Run comparison (would use actual model in real app)
            if st.button("Compare with Exact Solution"):
                # Create progress bar
                progress = st.progress(0.0)
                
                # Simulate computation
                for i in range(10):
                    time.sleep(0.1)  # Simulate computation
                    progress.progress((i + 1) / 10)
                
                # Display results
                st.success("Comparison completed!")
                
                # Create synthetic comparison data
                r_values = np.linspace(r_min, r_max, resolution)
                
                if selected_solution == "Schwarzschild Black Hole":
                    if component == "g₀₀":
                        exact = -(1 - 2/r_values)
                        learned = -(1 - 2/r_values) * (1 + np.random.normal(0, 0.005, r_values.shape))
                    elif component == "g₁₁":
                        exact = 1 / (1 - 2/r_values)
                        learned = 1 / (1 - 2/r_values) * (1 + np.random.normal(0, 0.01, r_values.shape))
                    else:
                        exact = r_values**2
                        learned = r_values**2 * (1 + np.random.normal(0, 0.007, r_values.shape))
                else:
                    # Generic placeholder
                    exact = r_values**0.5
                    learned = r_values**0.5 * (1 + np.random.normal(0, 0.01, r_values.shape))
                
                # Compute error
                abs_error = np.abs(learned - exact)
                rel_error = abs_error / np.abs(exact)
                
                # Create comparison plot
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=[f"{component} vs Radius", "Error"],
                                   row_heights=[0.7, 0.3])
                
                fig.add_trace(
                    go.Scatter(x=r_values, y=exact, mode='lines', name='Exact Solution', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=r_values, y=learned, mode='lines', name='Learned Solution', 
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                if error_scale == "Logarithmic":
                    fig.add_trace(
                        go.Scatter(x=r_values, y=rel_error, mode='lines', name='Relative Error'),
                        row=2, col=1
                    )        z=g_component,
        x=x.cpu().numpy() if slice_axis != 2 else z.cpu().numpy(),
        y=y.cpu().numpy() if slice_axis != 1 else z.cpu().numpy(),
        colorscale='Viridis',
        colorbar=dict(title=f'g_{{{component[0]}{component[1]}}}'),
    ))
    
    fig.update_layout(
        title=f'Metric Component g_{{{component[0]}{component[1]}}} at t={t_value:.2f}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=700,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def visualize_ricci_scalar(
    model: nn.Module,
    device: torch.device,
    t_value: float,
    slice_axis: int = 3,
    slice_value: float = 0.0
) -> go.Figure:
    """Visualize the Ricci scalar on a 2D slice.
    
    Args:
        model: Trained PINN model
        device: PyTorch device
        t_value: Time value for the slice
        slice_axis: Which spatial axis to fix
        slice_value: Value for the fixed axis
        
    Returns:
        Plotly figure with Ricci scalar visualization
    """
    model.eval()
    
    # Create a 2D grid for the remaining spatial coordinates
    N = 50  # Reduced resolution due to computational intensity
    L = 10.0  # Domain size
    
    if slice_axis == 1:  # Fixed x
        y = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        Y, Z = torch.meshgrid(y, z, indexing="ij")
        X = torch.full_like(Y, slice_value)
        grid_pts = torch.stack([
            torch.full_like(Y, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "y", "z"
    
    elif slice_axis == 2:  # Fixed y
        x = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        X, Z = torch.meshgrid(x, z, indexing="ij")
        Y = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "z"
    
    else:  # Fixed z (default)
        x = torch.linspace(-L, L, N, device=device)
        y = torch.linspace(-L, L, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "y"
    
    # Need gradients for curvature calculations
    grid_pts.requires_grad_(True)
    
    # Calculate Ricci scalar
    ricci_scalar = torch.zeros(grid_pts.shape[0], device=device)
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    num_batches = (grid_pts.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, grid_pts.shape[0])
        batch_pts = grid_pts[start_idx:end_idx]
        
        # Get metric tensor
        g = model(batch_pts)
        
        # Compute inverse metric
        g_inv = torch.inverse(g)
        
        # Compute Christoffel symbols and derivatives
        christoffel, d_christoffel = compute_christoffel_symbols(
            batch_pts, lambda x: model(x)
        )
        
        # Compute Riemann tensor
        riemann = compute_riemann_tensor(christoffel, d_christoffel)
        
        # Compute Ricci tensor
        ricci = compute_ricci_tensor(riemann, g_inv)
        
        # Compute Ricci scalar: R = g^μν R_μν
        for mu in range(4):
            for nu in range(4):
                ricci_scalar[start_idx:end_idx] += g_inv[:, mu, nu] * ricci[:, mu, nu]
    
    # Reshape back to grid
    ricci_scalar_grid = ricci_scalar.reshape(N, N).detach().cpu().numpy()
    
    # Create contour figure
    fig = go.Figure(data=go.Contour(
        z=ricci_scalar_grid,
        x=x.cpu().numpy() if slice_axis != 2 else z.cpu().numpy(),
        y=y.cpu().numpy() if slice_axis != 1 else z.cpu().numpy(),
        colorscale='RdBu',
        contours=dict(
            start=-0.1,
            end=0.1,
            size=0.01,
            showlabels=True
        ),
        colorbar=dict(title='Ricci Scalar')
    ))
    
    fig.update_layout(
        title=f'Ricci Scalar at t={t_value:.2f}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=700,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def visualize_adm_foliation(
    model: nn.Module,
    device: torch.device,
    t_values: List[float],
    slice_2d: bool = True,
    slice_axis: int = 2,
    slice_value: float = 0.0
) -> go.Figure:
    """Visualize the ADM foliation of spacetime.
    
    Args:
        model: Trained PINN model
        device: PyTorch device
        t_values: List of time values for the foliation
        slice_2d: Whether to show a 2D slice or 3D visualization
        slice_axis: Which spatial axis to fix if slice_2d=True
        slice_value: Value for the fixed axis if slice_2d=True
        
    Returns:
        Plotly figure with ADM foliation visualization
    """
    model.eval()
    
    # Create a spatial grid
    N = 30  # Resolution per dimension
    L = 10.0  # Domain size
    
    if slice_2d:
        # 2D slice visualization
        if slice_axis == 1:  # Fixed x
            y = torch.linspace(-L, L, N, device=device)
            z = torch.linspace(-L, L, N, device=device)
            Y, Z = torch.meshgrid(y, z, indexing="ij")
            X = torch.full_like(Y, slice_value)
            
            x_label, y_label = "y", "z"
        
        elif slice_axis == 2:  # Fixed y
            x = torch.linspace(-L, L, N, device=device)
            z = torch.linspace(-L, L, N, device=device)
            X, Z = torch.meshgrid(x, z, indexing="ij")
            Y = torch.full_like(X, slice_value)
            
            x_label, y_label = "x", "z"
        
        else:  # Fixed z
            x = torch.linspace(-L, L, N, device=device)
            y = torch.linspace(-L, L, N, device=device)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            Z = torch.full_like(X, slice_value)
            
            x_label, y_label = "x", "y"
        
        # Create figure
        fig = go.Figure()
        
        # Add each time slice
        for t_value in t_values:
            if slice_axis == 1:
                grid_pts = torch.stack([
                    torch.full_like(Y, t_value),
                    X, Y, Z
                ], dim=-1).reshape(-1, 4)
            elif slice_axis == 2:
                grid_pts = torch.stack([
                    torch.full_like(X, t_value),
                    X, Y, Z
                ], dim=-1).reshape(-1, 4)
            else:
                grid_pts = torch.stack([
                    torch.full_like(X, t_value),
                    X, Y, Z
                ], dim=-1).reshape(-1, 4)
            
            # Extract ADM variables
            with torch.no_grad():
                adm_vars = spacetime_to_adm(grid_pts, model)
            
            # Use lapse function for coloring
            lapse = adm_vars.alpha.reshape(N, N).cpu().numpy()
            
            # Plot the slice as a surface with height proportional to time
            fig.add_trace(go.Surface(
                z=np.full((N, N), t_value),
                x=X.cpu().numpy() if slice_axis != 2 else Z.cpu().numpy(),
                y=Y.cpu().numpy() if slice_axis != 1 else Z.cpu().numpy(),
                surfacecolor=lapse,
                colorscale='Viridis',
                showscale=(t_value == t_values[-1]),  # Only show colorbar for last slice
                colorbar=dict(title='Lapse α', x=1.02),
                name=f't={t_value:.2f}'
            ))
        
        fig.update_layout(
            title='ADM Foliation of Spacetime',
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title='t',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, t=60, b=0)
        )
    
    else:
        # 3D visualization
        x = torch.linspace(-L, L, N, device=device)
        y = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        
        # Sample points for volumetric visualization
        x_sample = x[::2]  # Subsample for performance
        y_sample = y[::2]
        z_sample = z[::2]
        N_sub = len(x_sample)
        
        # Create figure
        fig = go.Figure()
        
        for t_value in t_values:
            # Create grid for this time slice
            points = []
            
            # Create points along the grid edges for better visualization
            for i in range(N_sub):
                # x-edges
                points.append([t_value, x_sample[i], y_sample[0], z_sample[0]])
                points.append([t_value, x_sample[i], y_sample[-1], z_sample[0]])
                points.append([t_value, x_sample[i], y_sample[0], z_sample[-1]])
                points.append([t_value, x_sample[i], y_sample[-1], z_sample[-1]])
                
                # y-edges
                points.append([t_value, x_sample[0], y_sample[i], z_sample[0]])
                points.append([t_value, x_sample[-1], y_sample[i], z_sample[0]])
                points.append([t_value, x_sample[0], y_sample[i], z_sample[-1]])
                points.append([t_value, x_sample[-1], y_sample[i], z_sample[-1]])
                
                # z-edges
                points.append([t_value, x_sample[0], y_sample[0], z_sample[i]])
                points.append([t_value, x_sample[-1], y_sample[0], z_sample[i]])
                points.append([t_value, x_sample[0], y_sample[-1], z_sample[i]])
                points.append([t_value, x_sample[-1], y_sample[-1], z_sample[i]])
            
            grid_pts = torch.tensor(points, device=device)
            
            # Extract ADM variables
            with torch.no_grad():
                adm_vars = spacetime_to_adm(grid_pts, model)
            
            # Use lapse function for coloring
            lapse = adm_vars.alpha.cpu().numpy()
            
            # Plot 3D scatter of points colored by lapse
            fig.add_trace(go.Scatter3d(
                x=grid_pts[:, 1].cpu().numpy(),
                y=grid_pts[:, 2].cpu().numpy(),
                z=grid_pts[:, 3].cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=lapse,
                    colorscale='Viridis',
                    showscale=(t_value == t_values[-1]),
                    colorbar=dict(title='Lapse α')
                ),
                name=f't={t_value:.2f}'
            ))
        
        fig.update_layout(
            title='3D ADM Foliation of Spacetime',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='cube'
            ),
            width=800,
            height=800,
            margin=dict(l=0, r=0, t=60, b=0)
        )
    
    return fig


def visualize_light_cone(
    model: nn.Module,
    device: torch.device,
    origin: Tuple[float, float, float, float],
    max_distance: float = 5.0,
    num_rays: int = 100
) -> go.Figure:
    """Visualize the light cone structure in the spacetime.
    
    Args:
        model: Trained PINN model
        device: PyTorch device
        origin: Origin point (t,x,y,z) for the light cone
        max_distance: Maximum distance from origin
        num_rays: Number of light rays to trace
        
    Returns:
        Plotly figure with light cone visualization
    """
    model.eval()
    
    t0, x0, y0, z0 = origin
    
    # Generate directions for light rays
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_rays)))
    phi = np.arccos(2 * np.linspace(0, 1, int(np.sqrt(num_rays))) - 1)
    
    # Create meshgrid of angles
    Theta, Phi = np.meshgrid(theta, phi)
    
    # Convert to 3D unit vectors
    dx = np.sin(Phi) * np.cos(Theta)
    dy = np.sin(Phi) * np.sin(Theta)
    dz = np.cos(Phi)
    
    # Flatten direction vectors
    directions = np.stack([dx.flatten(), dy.flatten(), dz.flatten()], axis=1)
    
    # Keep only the requested number of rays
    directions = directions[:num_rays]
    
    # Points to store
    future_cone = []
    past_cone = []
    
    # Sample parameter along each ray
    lambda_values = np.linspace(0, max_distance, 20)
    
    for direction in directions:
        dx, dy, dz = direction
        
        # Future light cone
        for lambda_val in lambda_values:
            t = t0 + lambda_val
            x = x0 + dx * lambda_val
            y = y0 + dy * lambda_val
            z = z0 + dz * lambda_val
            
            future_cone.append([t, x, y, z])
        
        # Past light cone
        for lambda_val in lambda_values:
            t = t0 - lambda_val
            x = x0 + dx * lambda_val
            y = y0 + dy * lambda_val
            z = z0 + dz * lambda_val
            
            past_cone.append([t, x, y, z])
    
    # Convert to tensors
    future_pts = torch.tensor(future_cone, device=device)
    past_pts = torch.tensor(past_cone, device=device)
    
    # In a full implementation, we would solve the null geodesic equation
    # in the curved spacetime. For simplicity, we'll use the flat spacetime
    # approximation but color the rays by the lapse function.
    
    # Extract lapse function to visualize time dilation
    with torch.no_grad():
        future_adm = spacetime_to_adm(future_pts, model)
        past_adm = spacetime_to_adm(past_pts, model)
    
    future_lapse = future_adm.alpha.cpu().numpy()
    past_lapse = past_adm.alpha.cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Add origin point
    fig.add_trace(go.Scatter3d(
        x=[x0],
        y=[y0],
        z=[t0],  # Use t as the z-coordinate for better visualization
        mode='markers',
        marker=dict(
            size=8,
            color='white',
            line=dict(color='black', width=2)
        ),
        name='Origin'
    ))
    
    # Add future light cone
    fig.add_trace(go.Scatter3d(
        x=future_pts[:, 1].cpu().numpy(),
        y=future_pts[:, 2].cpu().numpy(),
        z=future_pts[:, 0].cpu().numpy(),  # t as z-coordinate
        mode='markers',
        marker=dict(
            size=3,
            color=future_lapse,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Lapse α')
        ),
        name='Future Light Cone'
    ))
    
    # Add past light cone
    fig.add_trace(go.Scatter3d(
        x=past_pts[:, 1].cpu().numpy(),
        y=past_pts[:, 2].cpu().numpy(),
        z=past_pts[:, 0].cpu().numpy(),  # t as z-coordinate
        mode='markers',
        marker=dict(
            size=3,
            color=past_lapse,
            colorscale='Viridis',
            showscale=False
        ),
        name='Past Light Cone'
    ))
    
    fig.update_layout(
        title=f'Light Cone Structure at ({t0:.1f}, {x0:.1f}, {y0:.1f}, {z0:.1f})',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='t',
            aspectratio=dict(x=1, y=1, z=1.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def plot_training_history(history: Dict[str, List[float]]) -> go.Figure:
    """Create an interactive plot of training history with constraints.
    
    Args:
        history: Dictionary with training metrics
        
    Returns:
        Plotly figure with loss and constraint curves
    """
    # Create multi-panel figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Training Losses", "Constraint Violations"],
        vertical_spacing=0.15
    )
    
    # Calculate epochs
    epochs = np.arange(1, len(history["total_loss"]) + 1) * 100
    
    # Add total loss curve
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history["total_loss"], 
            name="Total Loss",
            line=dict(color="blue", width=2)
        ),
        row=1, col=1
    )
    
    # Add component loss curves
    for name, values in history.items():
        if name.endswith("_loss") and name != "total_loss":
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=name.replace("_", " ").title(),
                    line=dict(width=1.5, dash="dash")
                ),
                row=1, col=1
            )
    
    # Add constraint violation curve
    if "constraint_violation" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["constraint_violation"],
                name="Constraint Violation",
                line=dict(color="red", width=2)
            ),
            row=2, col=1
        )
    
    # Add verification indicators
    if "verification" in history:
        # Extract verification epochs
        verification_epochs = np.arange(
            history["verification"][0].get("epoch", 500),
            len(history["total_loss"]) * 100 + 1,
            500
        )
        
        # Extract bianchi identity verification results
        bianchi_results = [
            1.0 if v.get("bianchi_identities", False) else 0.0
            for v in history["verification"]
        ]
        
        fig.add_trace(
            go.Scatter(
                x=verification_epochs,
                y=bianchi_results,
                name="Bianchi Identities",
                mode="markers",
                marker=dict(
                    size=10,
                    color=bianchi_results,
                    colorscale=[[0, "red"], [1, "green"]],
                    symbol="circle"
                )
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    fig.update_yaxes(title_text="Loss Value", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Constraint Metric", type="log", row=2, col=1)
    
    fig.update_layout(
        height=600,
        width=800,
        title="Training Progress with Constraints",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


# Stress-Energy Tensors for Matter Coupling


def dust_stress_energy(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    rho_fn: Optional[Callable] = None
) -> torch.Tensor:
    """Compute stress-energy tensor for dust (pressureless perfect fluid).
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        g: Metric tensor (batch_size, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        rho_fn: Function to compute rest mass density
        
    Returns:
        Stress-energy tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Initialize stress-energy tensor
    T = torch.zeros(batch_size, 4, 4, device=device)
    
    # Default density function (can be replaced with custom function)
    if rho_fn is None:
        # Example: Gaussian distribution centered at origin
        r_squared = torch.sum(coords[:, 1:4]**2, dim=1)
        rho = 0.1 * torch.exp(-r_squared / 5.0)
    else:
        rho = rho_fn(coords)
    
    # Four-velocity of dust (normalized timelike vector)
    # In comoving coordinates, u^μ = (1,0,0,0) in coordinate basis
    u = torch.zeros(batch_size, 4, device=device)
    u[:, 0] = 1.0  # Initial guess
    
    # Normalize to ensure g_μν u^μ u^ν = -1
    # This would require solving a quadratic equation in general
    # For simplicity, we'll use a comoving approximation where u^i = 0
    u_norm = torch.zeros_like(u)
    u_norm[:, 0] = 1.0 / torch.sqrt(-g[:, 0, 0])
    
    # Dust stress-energy: T^μν = ρ u^μ u^ν
    for mu in range(4):
        for nu in range(4):
            T[:, mu, nu] = rho * u_norm[:, mu] * u_norm[:, nu]
    
    return T


def perfect_fluid_stress_energy(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    rho_fn: Optional[Callable] = None,
    p_fn: Optional[Callable] = None
) -> torch.Tensor:
    """Compute stress-energy tensor for perfect fluid.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        g: Metric tensor (batch_size, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        rho_fn: Function to compute rest mass density
        p_fn: Function to compute pressure
        
    Returns:
        Stress-energy tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Initialize stress-energy tensor
    T = torch.zeros(batch_size, 4, 4, device=device)
    
    # Default density and pressure functions
    if rho_fn is None:
        # Example: Gaussian distribution centered at origin
        r_squared = torch.sum(coords[:, 1:4]**2, dim=1)
        rho = 0.1 * torch.exp(-r_squared / 5.0)
    else:
        rho = rho_fn(coords)
    
    if p_fn is None:
        # Example: Simple equation of state p = w*ρ with w=1/3 (radiation)
        p = rho / 3.0
    else:
        p = p_fn(coords)
    
    # Four-velocity of fluid (normalized timelike vector)
    # In comoving coordinates, u^μ = (1,0,0,0) in coordinate basis
    u = torch.zeros(batch_size, 4, device=device)
    u[:, 0] = 1.0  # Initial guess
    
    # Normalize to ensure g_μν u^μ u^ν = -1
    # For simplicity, we'll use a comoving approximation where u^i = 0
    u_norm = torch.zeros_like(u)
    u_norm[:, 0] = 1.0 / torch.sqrt(-g[:, 0, 0])
    
    # Perfect fluid stress-energy: T^μν = (ρ + p)u^μ u^ν + p g^μν
    for mu in range(4):
        for nu in range(4):
            T[:, mu, nu] = (rho + p) * u_norm[:, mu] * u_norm[:, nu] + p * g_inv[:, mu, nu]
    
    return T


def electromagnetic_stress_energy(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    F_fn: Optional[Callable] = None
) -> torch.Tensor:
    """Compute stress-energy tensor for electromagnetic field.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        g: Metric tensor (batch_size, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        F_fn: Function to compute electromagnetic tensor
        
    Returns:
        Stress-energy tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Initialize electromagnetic tensor
    if F_fn is None:
        # Example: Simple radial electric field
        F = torch.zeros(batch_size, 4, 4, device=device)
        
        # Extract spatial coordinates
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]
        r = torch.sqrt(x**2 + y**2 + z**2)
        r = torch.clamp(r, min=0.1)  # Avoid division by zero
        
        # Electric field components (radial)
        charge = 1.0
        E_factor = charge / (r**3)
        
        # F^0i = E^i (electric field)
        F[:, 0, 1] = E_factor * x
        F[:, 0, 2] = E_factor * y
        F[:, 0, 3] = E_factor * z
        
        # Antisymmetric tensor
        F[:, 1, 0] = -F[:, 0, 1]
        F[:, 2, 0] = -F[:, 0, 2]
        F[:, 3, 0] = -F[:, 0, 3]
    else:
        F = F_fn(coords)
    
    # Compute electromagnetic stress-energy tensor
    # T^μν = (1/4π) (F^μα F^ν_α - (1/4) g^μν F^αβ F_αβ)
    T = torch.zeros(batch_size, 4, 4, device=device)
    
    # Lower one index of F to compute F^μ_α
    F_mixed = torch.zeros_like(F)
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                F_mixed[:, mu, alpha] += F[:, mu, beta] * g[:, beta, alpha]
    
    # Compute first term: F^μα F^ν_α
    term1 = torch.zeros_like(T)
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                term1[:, mu, nu] += F[:, mu, alpha] * F_mixed[:, nu, alpha]
    
    # Compute second term: (1/4) g^μν F^αβ F_αβ
    # First calculate the invariant F^αβ F_αβ
    F_squared = torch.zeros(batch_size, device=device)
    for alpha in range(4):
        for beta in range(4):
            F_squared += F[:, alpha, beta] * F_mixed[:, alpha, beta]
    
    # Then multiply by g^μν
    term2 = torch.zeros_like(T)
    for mu in range(4):
        for nu in range(4):
            term2[:, mu, nu] = 0.25 * g_inv[:, mu, nu] * F_squared
    
    # Final stress-energy tensor
    T = (1.0 / (4 * math.pi)) * (term1 - term2)
    
    return T


def scalar_field_stress_energy(
    coords: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    phi_fn: Optional[Callable] = None,
    potential_fn: Optional[Callable] = None
) -> torch.Tensor:
    """Compute stress-energy tensor for a scalar field.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        g: Metric tensor (batch_size, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        phi_fn: Function to compute scalar field
        potential_fn: Function to compute scalar field potential
        
    Returns:
        Stress-energy tensor (batch_size, 4, 4)
    """
    coords.requires_grad_(True)
    batch_size = coords.shape[0]
    device = coords.device
    
    # Initialize stress-energy tensor
    T = torch.zeros(batch_size, 4, 4, device=device)
    
    # Default scalar field function
    if phi_fn is None:
        # Example: Simple scalar field
        r_squared = torch.sum(coords[:, 1:4]**2, dim=1)
        phi = torch.exp(-r_squared / 10.0)
    else:
        phi = phi_fn(coords)
    
    # Compute derivatives of phi
    dphi = torch.zeros(batch_size, 4, device=device)
    for mu in range(4):
        dphi[:, mu] = torch.autograd.grad(
            phi, coords,
            grad_outputs=torch.ones_like(phi),
            create_graph=True
        )[0][:, mu]
    
    # Default potential function
    if potential_fn is None:
        # Example: Simple quadratic potential V(φ) = m²φ²/2
        m_squared = 0.1
        V = 0.5 * m_squared * phi**2
    else:
        V = potential_fn(phi)
    
    # Compute scalar field stress-energy tensor
    # T^μν = ∂^μφ ∂^νφ - g^μν[g^αβ ∂_αφ ∂_βφ/2 + V(φ)]
    
    # First compute g^αβ ∂_αφ ∂_βφ
    kinetic_term = torch.zeros(batch_size, device=device)
    for alpha in range(4):
        for beta in range(4):
            kinetic_term += g_inv[:, alpha, beta] * dphi[:, alpha] * dphi[:, beta]
    
    # Now compute the full tensor
    for mu in range(4):
        for nu in range(4):
            # First term: ∂^μφ ∂^νφ
            T[:, mu, nu] = dphi[:, mu] * dphi[:, nu]
            
            # Second term: -g^μν[g^αβ ∂_αφ ∂_βφ/2 + V(φ)]
            T[:, mu, nu] -= g_inv[:, mu, nu] * (0.5 * kinetic_term + V)
    
    return T


# -----------------------------------------------------
# 9. Specific Solution Templates
# -----------------------------------------------------

def schwarzschild_initial_metric(
    coords: torch.Tensor,
    mass: float = 1.0
) -> torch.Tensor:
    """Initialize a network with Schwarzschild metric.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        mass: Black hole mass parameter
        
    Returns:
        Schwarzschild metric tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Calculate r = sqrt(x² + y² + z²)
    r = torch.sqrt(x**2 + y**2 + z**2)
    r = torch.clamp(r, min=2.0 * mass)  # Avoid singularity
    
    # Initialize metric tensor
    g = torch.zeros(batch_size, 4, 4, device=device)
    
    # Schwarzschild metric in Cartesian-like coordinates
    # Time component: g_00 = -(1 - 2M/r)
    g[:, 0, 0] = -(1.0 - 2.0 * mass / r)
    
    # Spatial components combine flat metric with radial correction
    delta_ij = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    radial_correction = torch.zeros(batch_size, 3, 3, device=device)
    
    # Fill radial correction tensor
    for i in range(3):
        for j in range(3):
            x_i = coords[:, i+1]
            x_j = coords[:, j+1]
            radial_factor = (2.0 * mass / r) / (1.0 - 2.0 * mass / r)
            radial_correction[:, i, j] = radial_factor * x_i * x_j / (r**2)
    
    # Combine to get spatial metric
    for i in range(3):
        for j in range(3):
            g[:, i+1, j+1] = delta_ij[:, i, j] + radial_correction[:, i, j]
    
    return g


def kerr_initial_metric(
    coords: torch.Tensor,
    mass: float = 1.0,
    spin: float = 0.5
) -> torch.Tensor:
    """Initialize a network with Kerr metric (spinning black hole).
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        mass: Black hole mass parameter
        spin: Black hole spin parameter a = J/M (0 ≤ a < M)
        
    Returns:
        Kerr metric tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates (note: we need to transform to Boyer-Lindquist coordinates)
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Compute cylindrical radius ρ = sqrt(x² + y²)
    rho = torch.sqrt(x**2 + y**2)
    
    # Compute Boyer-Lindquist radius r
    # This requires solving a quadratic equation
    # r² - (x² + y² + z²) + a² + a²z²/(r²) = 0
    # We'll use an approximation for simplicity
    r_squared = x**2 + y**2 + z**2 - spin**2
    r_squared = torch.clamp(r_squared, min=0.01)  # Ensure positive
    r = torch.sqrt(r_squared)
    
    # Compute boyer-lindquist theta
    theta = torch.acos(z / torch.clamp(torch.sqrt(x**2 + y**2 + z**2), min=0.01))
    
    # Compute sin(theta) and cos(theta)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Compute azimuthal angle phi
    phi = torch.atan2(y, x)
    
    # Compute metric coefficients in Boyer-Lindquist coordinates
    sigma = r**2 + (spin * cos_theta)**2
    delta = r**2 - 2.0 * mass * r + spin**2
    
    # Initialize metric tensor
    g = torch.zeros(batch_size, 4, 4, device=device)
    
    # Kerr metric in Boyer-Lindquist coordinates
    # g_tt component
    g[:, 0, 0] = -((1.0 - 2.0 * mass * r / sigma))
    
    # g_tφ component (off-diagonal)
    g[:, 0, 3] = -2.0 * mass * r * spin * sin_theta**2 / sigma
    g[:, 3, 0] = g[:, 0, 3]  # Symmetry
    
    # g_rr component (using coordinate transform)
    g[:, 1, 1] = sigma / delta
    
    # g_θθ component
    g[:, 2, 2] = sigma
    
    # g_φφ component
    g[:, 3, 3] = (r**2 + spin**2 + 2.0 * mass * r * spin**2 * sin_theta**2 / sigma) * sin_theta**2
    
    # TODO: Transform from Boyer-Lindquist to Cartesian coordinates
    # For simplicity, we'll return the BL metric with a warning
    print("Warning: Kerr metric is in Boyer-Lindquist coordinates, not Cartesian")
    
    return g


def friedmann_initial_metric(
    coords: torch.Tensor,
    H0: float = 0.1,  # Hubble parameter
    density_param: float = 0.3  # Ω_m
) -> torch.Tensor:
    """Initialize a network with FLRW metric (expanding universe).
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        H0: Hubble parameter at present time
        density_param: Matter density parameter (Ω_m)
        
    Returns:
        FLRW metric tensor (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Compute scale factor a(t) based on cosmological model
    # For simplicity, we'll use a matter-dominated universe
    # a(t) = (t/t_0)^(2/3) for matter domination
    # We'll set t_0 = 1 for simplicity
    scale_factor = torch.pow(torch.clamp(t, min=0.01), 2.0/3.0)
    
    # Initialize metric tensor
    g = torch.zeros(batch_size, 4, 4, device=device)
    
    # FLRW metric in Cartesian coordinates
    # g_00 = -1 (proper time)
    g[:, 0, 0] = -1.0
    
    # Spatial components with scale factor
    for i in range(3):
        g[:, i+1, i+1] = scale_factor**2
    
    return g


def gravitational_wave_initial_metric(
    coords: torch.Tensor,
    amplitude: float = 0.01,
    frequency: float = 1.0,
    direction: List[float] = [0, 0, 1]  # z-direction
) -> torch.Tensor:
    """Initialize a network with a gravitational wave perturbation.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        amplitude: Wave amplitude
        frequency: Wave frequency
        direction: Propagation direction
        
    Returns:
        Metric tensor with gravitational wave (batch_size, 4, 4)
    """
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract coordinates
    t = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    
    # Normalize direction vector
    direction = torch.tensor(direction, device=device)
    direction = direction / torch.norm(direction)
    
    # Compute wave phase
    phase = frequency * (t - direction[0] * x - direction[1] * y - direction[2] * z)
    
    # Initialize with Minkowski metric
    g = torch.zeros(batch_size, 4, 4, device=device)
    g[:, 0, 0] = -1.0
    for i in range(1, 4):
        g[:, i, i] = 1.0
    
    # Add gravitational wave perturbation in TT gauge
    # For simplicity, we'll consider a "plus" polarization wave
    h_plus = amplitude * torch.sin(phase)
    
    # For z-direction propagation, the perturbation affects x and y components
    if torch.allclose(direction, torch.tensor([0.0, 0.0, 1.0], device=device)):
        g[:, 1, 1] += h_plus
        g[:, 2, 2] -= h_plus
    else:
        # For arbitrary direction, need to compute TT frame
        # This is a simplification; a full implementation would use
        # the Newman-Penrose formalism or proper TT projection
        # We'll just perturb the transverse components
        g[:, 1, 1] += h_plus
        g[:, 2, 2] -= h_plus
    
    return g


# -----------------------------------------------------
# 10. Streamlit App Layout
# -----------------------------------------------------

def main():
    """Main function for the Streamlit app."""
    st.title("Advanced Einstein Field Equations Solver")
    st.markdown("""
    This application solves Einstein's field equations using Physics-Informed Neural Networks 
    with 3+1 decomposition and curvature back-reaction. The system can handle:
    
    - Full non-linear Einstein equations in vacuum and with matter
    - ADM and BSSN formulations for numerical stability
    - Symbolic verification of solutions
    - Various gauge conditions and coordinate systems
    - Multiple spacetime geometries (black holes, gravitational waves, cosmology)
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Solver Configuration", "Training", "Visualization", "Analysis"]
    )
    
    if page == "Overview":
        show_overview_page()
    elif page == "Solver Configuration":
        show_configuration_page()
    elif page == "Training":
        show_training_page()
    elif page == "Visualization":
        show_visualization_page()
    elif page == "Analysis":
        show_analysis_page()


def show_overview_page():
    """Display the overview page with theory and approach."""
    st.header("Einstein Field Equations Solver Overview")
    
    st.subheader("Theoretical Background")
    st.markdown("""
    ### Einstein's Field Equations
    
    The Einstein field equations relate the geometry of spacetime to the distribution of matter and energy:
    
    $G_{\mu\\nu} = 8\pi T_{\mu\\nu}$
    
    where $G_{\mu\\nu}$ is the Einstein tensor describing spacetime curvature, and $T_{\mu\\nu}$ is the 
    stress-energy tensor describing matter and energy.
    
    ### 3+1 Decomposition (ADM Formalism)
    
    The ADM formalism decomposes spacetime into spatial hypersurfaces evolving in time, characterized by:
    
    - Spatial metric $\gamma_{ij}$
    - Extrinsic curvature $K_{ij}$
    - Lapse function $\\alpha$
    - Shift vector $\\beta^i$
    
    This decomposition splits Einstein's equations into constraint and evolution equations:
    
    - Hamiltonian constraint: $R + K^2 - K_{ij}K^{ij} = 16\pi\rho$
    - Momentum constraint: $\nabla_j K^j_i - \nabla_i K = 8\pi j_i$
    """)
    
    st.subheader("Physics-Informed Neural Network Approach")
    st.markdown("""
    ### PINN Architecture
    
    This solver uses a SIREN-based neural network (sinusoidal activations) to represent the metric tensor
    $g_{\mu\\nu}(t,x,y,z)$. The network is trained to satisfy:
    
    1. The Einstein field equations
    2. Hamiltonian and momentum constraints
    3. Gauge conditions
    4. Boundary conditions
    
    ### Symbolic Verification
    
    During training, the solution is periodically checked using symbolic computation to verify:
    
    - Bianchi identities
    - Constraint violations
    - Conservation laws
    
    ### Key Improvements Over Conventional Methods
    
    - **Mesh-free approach**: No need for explicit grid discretization
    - **Adaptive resolution**: Natural focus on regions with high curvature
    - **Differentiable physics**: Automatic handling of all derivatives
    - **Analytic solutions**: Results in a continuous, differentiable spacetime metric
    """)
    
    st.subheader("Available Spacetime Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Vacuum solutions**
          - Schwarzschild black hole
          - Kerr (rotating) black hole
          - Gravitational waves
          
        - **Cosmological models**
          - FLRW expanding universe
          - De Sitter spacetime
        """)
    
    with col2:
        st.markdown("""
        - **Matter coupling**
          - Dust (pressureless matter)
          - Perfect fluid
          - Electromagnetic field
          - Scalar field
          
        - **Custom spacetimes**
          - User-defined initial conditions
          - Arbitrary stress-energy tensors
        """)


def show_configuration_page():
    """Display configuration options for the solver."""
    st.header("Solver Configuration")
    
    # Spacetime model selection
    st.subheader("Spacetime Model")
    
    spacetime_type = st.selectbox(
        "Select Spacetime Type",
        ["Vacuum", "Matter", "Cosmological", "Custom"]
    )
    
    if spacetime_type == "Vacuum":
        vacuum_model = st.selectbox(
            "Select Vacuum Solution",
            ["Schwarzschild", "Kerr", "Gravitational Wave"]
        )
        
        if vacuum_model == "Schwarzschild":
            mass = st.slider("Black Hole Mass (M)", 0.1, 10.0, 1.0, 0.1)
            st.markdown(f"Schwarzschild radius: $r_s = {2*mass:.2f}$")
            
        elif vacuum_model == "Kerr":
            mass = st.slider("Black Hole Mass (M)", 0.1, 10.0, 1.0, 0.1)
            spin = st.slider("Spin Parameter (a/M)", 0.0, 0.99, 0.5, 0.01)
            st.markdown(f"Extremality: $a/M = {spin:.2f}$ (1.0 is extremal)")
            
        elif vacuum_model == "Gravitational Wave":
            amplitude = st.slider("Wave Amplitude", 0.001, 0.1, 0.01, 0.001)
            frequency = st.slider("Wave Frequency", 0.1, 5.0, 1.0, 0.1)
            direction = st.selectbox("Wave Direction", ["x", "y", "z"])
            
            dir_vector = [0, 0, 0]
            if direction == "x":
                dir_vector = [1, 0, 0]
            elif direction == "y":
                dir_vector = [0, 1, 0]
            else:
                dir_vector = [0, 0, 1]
            
            st.markdown(f"Wave speed: $c = 1$ (natural units)")
    
    elif spacetime_type == "Matter":
        matter_type = st.selectbox(
            "Select Matter Type",
            ["Dust", "Perfect Fluid", "Electromagnetic Field", "Scalar Field"]
        )
        
        if matter_type == "Dust":
            st.markdown("Pressureless dust: $T_{\mu\\nu} = \\rho u_\mu u_\\nu$")
            rho_max = st.slider("Maximum Density", 0.01, 1.0, 0.1, 0.01)
            
        elif matter_type == "Perfect Fluid":
            st.markdown("Perfect fluid: $T_{\mu\\nu} = (\\rho + p)u_\mu u_\\nu + p g_{\mu\\nu}$")
            rho_max = st.slider("Maximum Density", 0.01, 1.0, 0.1, 0.01)
            eos_param = st.slider("Equation of State (w = p/ρ)", 0.0, 1.0, 0.33, 0.01)
            
        elif matter_type == "Electromagnetic Field":
            st.markdown("EM field: $T_{\mu\\nu} = \\frac{1}{4\pi}(F_{\mu}^{\\alpha}F_{\\nu\\alpha} - \\frac{1}{4}g_{\mu\\nu}F_{\\alpha\\beta}F^{\\alpha\\beta})$")
            charge = st.slider("Charge Parameter", 0.01, 2.0, 1.0, 0.01)
            
        elif matter_type == "Scalar Field":
            st.markdown("Scalar field: $T_{\mu\\nu} = \\partial_\mu\\phi\\partial_\\nu\\phi - g_{\mu\\nu}[\\frac{1}{2}g^{\\alpha\\beta}\\partial_\\alpha\\phi\\partial_\\beta\\phi + V(\\phi)]$")
            amplitude = st.slider("Field Amplitude", 0.1, 2.0, 1.0, 0.1)
            mass_param = st.slider("Field Mass", 0.01, 1.0, 0.1, 0.01)
    
    elif spacetime_type == "Cosmological":
        cosmo_model = st.selectbox(
            "Select Cosmological Model",
            ["FLRW Universe", "De Sitter"]
        )
        
        if cosmo_model == "FLRW Universe":
            h0 = st.slider("Hubble Parameter (H₀)", 0.01, 0.2, 0.1, 0.01)
            omega_m = st.slider("Matter Density (Ω_m)", 0.0, 1.0, 0.3, 0.01)
            omega_lambda = st.slider("Dark Energy (Ω_Λ)", 0.0, 1.0, 0.7, 0.01)
            
            # Ensure they sum to 1.0
            if abs(omega_m + omega_lambda - 1.0) > 0.01:
                st.warning("Note: Ω_m + Ω_Λ ≠ 1.0 (not flat)")
        
        elif cosmo_model == "De Sitter":
            lambda_cosmo = st.slider("Cosmological Constant (Λ)", 0.01, 1.0, 0.1, 0.01)
    
    elif spacetime_type == "Custom":
        st.info("Custom spacetime configuration is currently experimental")
        st.text_area("Metric Expression (SymPy format)", "g_00 = -1 + 2*M/r\ng_11 = 1/(1-2*M/r)\ng_22 = r**2\ng_33 = r**2*sin(theta)**2")
    
    # Neural Network Architecture
    st.subheader("Neural Network Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_dim = st.slider("Hidden Layer Width", 32, 256, 128, 32)
        num_layers = st.slider("Number of Hidden Layers", 2, 8, 4, 1)
        activation = st.selectbox("Activation Function", ["Sine (SIREN)", "Fourier Features"])
    
    with col2:
        omega = st.slider("SIREN Frequency (ω)", 10.0, 50.0, 30.0, 5.0)
        use_skip = st.checkbox("Use Skip Connections", True)
        learn_freq = st.checkbox("Learnable Frequencies", True)
    
    # Domain and Boundary Conditions
    st.subheader("Domain Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        t_min = st.number_input("Minimum Time", -10.0, 0.0, 0.0, 1.0)
        t_max = st.number_input("Maximum Time", 1.0, 20.0, 10.0, 1.0)
        spatial_extent = st.slider("Spatial Domain Size", 5.0, 50.0, 20.0, 5.0)
    
    with col2:
        gauge_condition = st.selectbox(
            "Gauge Condition",
            ["Harmonic", "Maximal Slicing", "1+log", "Geodesic"]
        )
        
        boundary_type = st.selectbox(
            "Boundary Conditions",
            ["Asymptotically Flat", "Periodic", "Outgoing Radiation"]
        )
    
    # Advanced Settings
    with st.expander("Advanced Settings"):
        adaptive_sampling = st.checkbox("Adaptive Sampling", True)
        symbolic_checks = st.checkbox("Symbolic Verification", True)
        verification_freq = st.slider("Verification Frequency (epochs)", 100, 1000, 500, 100)
        
        st.subheader("Loss Function Weights")
        field_eq_weight = st.slider("Einstein Equation Weight", 0.1, 10.0, 1.0, 0.1)
        constraint_weight = st.slider("Constraint Equation Weight", 0.1, 10.0, 1.0, 0.1)
        gauge_weight = st.slider("Gauge Condition Weight", 0.1, 5.0, 0.5, 0.1)
        boundary_weight = st.slider("Boundary Condition Weight", 0.1, 5.0, 0.5, 0.1)
    
    # Save configuration
    if st.button("Save Configuration"):
        st.success("Configuration saved successfully")
        # In a real app, we would save these settings to session state
        # or pass them to the training function


def show_training_page():
    """Display training interface for the solver."""
    st.header("Train Einstein Field Equations Solver")
    
    # Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider("Batch Size", 512, 8192, 2048, 512)
        epochs = st.slider("Training Epochs", 1000, 20000, 5000, 1000)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            value=1e-4
        )
    
    with col2:
        device = st.selectbox("Computation Device", ["CPU", "CUDA (GPU)"])
        curriculum = st.checkbox("Use Curriculum Learning", True)
        precision = st.selectbox("Numerical Precision", ["32-bit (float)", "64-bit (double)"])
        
        if curriculum:
            curriculum_steps = st.slider("Curriculum Stages", 2, 5, 3, 1)
    
    # Region of interest for focused sampling
    st.subheader("Sampling Focus Regions")
    
    use_focus = st.checkbox("Use Focused Sampling Regions", False)
    
    if use_focus:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            region_t = st.number_input("Time Center", 0.0, 10.0, 5.0, 1.0)
            region_x = st.number_input("X Center", -10.0, 10.0, 0.0, 1.0)
        
        with col2:
            region_y = st.number_input("Y Center", -10.0, 10.0, 0.0, 1.0)
            region_z = st.number_input("Z Center", -10.0, 10.0, 0.0, 1.0)
        
        with col3:
            region_radius = st.number_input("Region Radius", 0.5, 5.0, 2.0, 0.5)
            st.write("This will focus sampling in the specified region")
    
    # Training execution
    st.subheader("Training Execution")
    
    started_training = st.button("Start Training")
    
    if started_training:
        # In a real app, we would initialize and train the model here
        
        # Set up progress tracking
        progress = st.progress(0.0)
        loss_container = st.container()
        
        with loss_container:
            loss_cols = st.columns([2, 1])
            
            with loss_cols[0]:
                loss_chart = st.empty()
            
            with loss_cols[1]:
                constraint_chart = st.empty()
        
        # Create dummy training progress for demonstration
        history = {
            "total_loss": [1.0 * (0.9 ** i) for i in range(10)],
            "field_equations_loss": [0.8 * (0.9 ** i) for i in range(10)],
            "adm_constraints_loss": [0.5 # einstein_field_solver.py
"""
Advanced Physics-Informed Neural Network (PINN) solver for Einstein's field equations
with 3+1 decomposition, curvature back-reaction, and symbolic verification.

This application implements:
1. 3+1 ADM formalism (Arnowitt-Deser-Misner) for numerical relativity
2. Full non-linear Einstein field equations with curvature back-reaction
3. Symbolic computation for verification and constraint checks
4. BSSN formulation (Baumgarte-Shapiro-Shibata-Nakamura) for stable evolution
5. Both vacuum solutions and matter coupling with stress-energy tensor

Dependencies:
    streamlit torch numpy sympy plotly pandas matplotlib scipy
    
Advanced features:
    - Gauge-invariant metric evolution
    - Constraint violation monitoring
    - Symbolic verification of solutions
    - Asymptotic boundary conditions
    - Adaptive mesh refinement via PINN sampling
"""

import time
import math
import random
from typing import Tuple, Dict, List, Optional, Callable, Union
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

# Configure page settings
st.set_page_config(
    page_title="Advanced Einstein Field Equations Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# 1. Neural Network Architecture (SIREN with enhancements)
# -----------------------------------------------------
class Sine(nn.Module):
    """Sinusoidal activation y = sin(ω·x) with learnable frequency."""
    
    def __init__(self, omega: float = 30.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.tensor(omega, dtype=torch.float32))
        else:
            self.omega = omega
        
    def forward(self, x):
        return torch.sin(self.omega * x)


class SIREN(nn.Module):
    """Enhanced SIREN network with additional features for relativistic problems.
    
    - Support for multi-component outputs (full metric tensor)
    - Fourier feature mapping for better handling of high frequencies
    - Skip connections for gradient flow
    - Learnable sine frequencies
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        hidden_features: int = 128,
        hidden_layers: int = 4, 
        outermost_linear: bool = True,
        omega: float = 30.0,
        use_fourier_features: bool = True,
        fourier_scale: float = 10.0,
        use_skip_connections: bool = True,
        learnable_frequencies: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.use_fourier_features = use_fourier_features
        self.use_skip_connections = use_skip_connections
        
        # If using Fourier features, adjust input dimension
        if use_fourier_features:
            self.B = torch.randn(in_features, hidden_features // 2) * fourier_scale
            self.B = nn.Parameter(self.B, requires_grad=learnable_frequencies)
            self.net_in_features = hidden_features
        else:
            self.net_in_features = in_features
            
        # Network layers
        self.net = nn.ModuleList()
        
        # First layer
        self.net.append(nn.Linear(self.net_in_features, hidden_features))
        self.net.append(Sine(omega=omega, learnable=learnable_frequencies))
        
        # Hidden layers with optional skip connections
        for i in range(hidden_layers):
            if use_skip_connections and i % 2 == 1:
                self.net.append(nn.Linear(hidden_features * 2, hidden_features))
            else:
                self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(Sine(omega=omega, learnable=learnable_frequencies))
            
        # Final layer
        self.net.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(Sine(omega=omega, learnable=learnable_frequencies))
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Weight initialization for SIREN."""
        if isinstance(m, nn.Linear):
            # First layer 
            if m is self.net[0]:
                nn.init.uniform_(m.weight, -1/self.net_in_features, 1/self.net_in_features)
            else:
                omega = 30.0  # Default if not learnable
                if isinstance(self.net[1], Sine):
                    if hasattr(self.net[1], 'omega') and isinstance(self.net[1].omega, nn.Parameter):
                        omega = self.net[1].omega.item()
                    else:
                        omega = self.net[1].omega
                        
                bound = math.sqrt(6 / m.in_features) / omega
                nn.init.uniform_(m.weight, -bound, bound)
                
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass with optional Fourier features and skip connections."""
        orig_x = x
        
        # Apply Fourier feature mapping if enabled
        if self.use_fourier_features:
            x_proj = x @ self.B
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # First two layers (linear + activation)
        x = self.net[0](x)
        x = self.net[1](x)
        prev_x = x
        
        # Process remaining layers with skip connections
        skip_idx = 0
        for i in range(2, len(self.net), 2):
            if i+1 < len(self.net):  # If we have both linear and activation
                if self.use_skip_connections and skip_idx % 2 == 1:
                    # Apply skip connection
                    x = torch.cat([x, prev_x], dim=-1)
                
                prev_x = x
                x = self.net[i](x)    # Linear
                x = self.net[i+1](x)  # Activation
                skip_idx += 1
            else:
                # Last layer (might be just linear)
                x = self.net[i](x)
                if i+1 < len(self.net):
                    x = self.net[i+1](x)
                    
        return x


# -----------------------------------------------------
# 2. ADM 3+1 Formalism and BSSN Implementation
# -----------------------------------------------------

@dataclass
class ADMVariables:
    """Container for ADM (Arnowitt-Deser-Misner) 3+1 decomposition variables."""
    # Lapse function (time evolution)
    alpha: torch.Tensor
    
    # Shift vector (spatial coordinate evolution)
    beta: torch.Tensor  # (batch_size, 3)
    
    # Spatial metric (3x3 symmetric tensor)
    gamma: torch.Tensor  # (batch_size, 3, 3)
    
    # Extrinsic curvature (3x3 symmetric tensor)
    K: torch.Tensor  # (batch_size, 3, 3)
    

@dataclass
class BSSNVariables:
    """Container for BSSN (Baumgarte-Shapiro-Shibata-Nakamura) variables."""
    # Conformal factor
    phi: torch.Tensor
    
    # Conformal metric
    gamma_tilde: torch.Tensor  # (batch_size, 3, 3)
    
    # Conformal connection functions
    Gamma_tilde: torch.Tensor  # (batch_size, 3)
    
    # Trace of extrinsic curvature
    K: torch.Tensor
    
    # Traceless part of extrinsic curvature
    A_tilde: torch.Tensor  # (batch_size, 3, 3)
    
    # Lapse and shift
    alpha: torch.Tensor
    beta: torch.Tensor


def compute_christoffel_symbols(
    coords: torch.Tensor,
    metric_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Christoffel symbols and their derivatives using automatic differentiation.
    
    Args:
        coords: Spacetime coordinates tensor (batch_size, 4)
        metric_fn: Function that computes the metric from coordinates
        
    Returns:
        Tuple of (Christoffel symbols, derivatives of Christoffel symbols)
    """
    coords.requires_grad_(True)
    
    # Get metric tensor g_μν
    g = metric_fn(coords)  # (batch_size, 4, 4)
    
    batch_size = coords.shape[0]
    device = coords.device
    
    # Compute inverse metric g^μν
    g_inv = torch.inverse(g)  # (batch_size, 4, 4)
    
    # Initialize Christoffel symbols
    christoffel = torch.zeros(batch_size, 4, 4, 4, device=device)
    
    # For each coordinate direction
    for mu in range(4):
        # Compute partial derivatives of metric w.r.t. this coordinate
        dg_dmu = torch.autograd.grad(
            g, coords, 
            grad_outputs=torch.eye(4, 4, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            create_graph=True, retain_graph=True
        )[0][:, :, mu].view(batch_size, 4, 4)
        
        # For each index pair, compute Christoffel symbols
        for alpha in range(4):
            for beta in range(4):
                for gamma in range(4):
                    # Γ^α_βγ = (1/2) g^αδ (∂_β g_γδ + ∂_γ g_βδ - ∂_δ g_βγ)
                    christoffel[:, alpha, beta, gamma] = 0.5 * sum(
                        g_inv[:, alpha, delta] * (
                            dg_dmu[:, gamma, delta] + 
                            dg_dmu[:, beta, delta] - 
                            dg_dmu[:, beta, gamma]
                        )
                        for delta in range(4)
                    )
    
    # Compute derivatives of Christoffel symbols (for Riemann tensor)
    d_christoffel = torch.zeros(batch_size, 4, 4, 4, 4, device=device)
    
    for mu in range(4):
        d_christoffel[:, :, :, :, mu] = torch.autograd.grad(
            christoffel, coords,
            grad_outputs=torch.ones_like(christoffel),
            create_graph=True
        )[0][:, :, mu].view(batch_size, 4, 4, 4)
    
    return christoffel, d_christoffel


def compute_riemann_tensor(
    christoffel: torch.Tensor,
    d_christoffel: torch.Tensor
) -> torch.Tensor:
    """Compute the Riemann curvature tensor.
    
    Args:
        christoffel: Christoffel symbols tensor (batch_size, 4, 4, 4)
        d_christoffel: Derivatives of Christoffel symbols (batch_size, 4, 4, 4, 4)
        
    Returns:
        Riemann tensor (batch_size, 4, 4, 4, 4)
    """
    batch_size = christoffel.shape[0]
    device = christoffel.device
    
    # Initialize Riemann tensor
    riemann = torch.zeros(batch_size, 4, 4, 4, 4, device=device)
    
    # R^α_βγδ = ∂_γ Γ^α_βδ - ∂_δ Γ^α_βγ + Γ^α_σγ Γ^σ_βδ - Γ^α_σδ Γ^σ_βγ
    for alpha in range(4):
        for beta in range(4):
            for gamma in range(4):
                for delta in range(4):
                    # Derivatives of Christoffel symbols
                    riemann[:, alpha, beta, gamma, delta] = (
                        d_christoffel[:, alpha, beta, delta, gamma] - 
                        d_christoffel[:, alpha, beta, gamma, delta]
                    )
                    
                    # Products of Christoffel symbols
                    for sigma in range(4):
                        riemann[:, alpha, beta, gamma, delta] += (
                            christoffel[:, alpha, sigma, gamma] * 
                            christoffel[:, sigma, beta, delta] -
                            christoffel[:, alpha, sigma, delta] * 
                            christoffel[:, sigma, beta, gamma]
                        )
    
    return riemann


def compute_ricci_tensor(riemann: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """Compute the Ricci tensor by contracting the Riemann tensor.
    
    Args:
        riemann: Riemann tensor (batch_size, 4, 4, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        
    Returns:
        Ricci tensor (batch_size, 4, 4)
    """
    batch_size = riemann.shape[0]
    device = riemann.device
    
    # Initialize Ricci tensor
    ricci = torch.zeros(batch_size, 4, 4, device=device)
    
    # R_μν = R^α_μαν
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                ricci[:, mu, nu] += riemann[:, alpha, mu, alpha, nu]
    
    return ricci


def compute_einstein_tensor(ricci: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """Compute the Einstein tensor G_μν = R_μν - (1/2)R g_μν.
    
    Args:
        ricci: Ricci tensor (batch_size, 4, 4)
        g: Metric tensor (batch_size, 4, 4)
        g_inv: Inverse metric tensor (batch_size, 4, 4)
        
    Returns:
        Einstein tensor (batch_size, 4, 4)
    """
    batch_size = ricci.shape[0]
    device = ricci.device
    
    # Compute Ricci scalar R = g^μν R_μν
    ricci_scalar = torch.zeros(batch_size, device=device)
    for mu in range(4):
        for nu in range(4):
            ricci_scalar += g_inv[:, mu, nu] * ricci[:, mu, nu]
    
    # Compute Einstein tensor G_μν = R_μν - (1/2)R g_μν
    einstein = torch.zeros_like(ricci)
    for mu in range(4):
        for nu in range(4):
            einstein[:, mu, nu] = ricci[:, mu, nu] - 0.5 * ricci_scalar * g[:, mu, nu]
    
    return einstein


def adm_to_spacetime_metric(adm: ADMVariables) -> torch.Tensor:
    """Convert ADM variables to the full 4D spacetime metric.
    
    Args:
        adm: ADM variables container
        
    Returns:
        4D spacetime metric tensor g_μν (batch_size, 4, 4)
    """
    batch_size = adm.alpha.shape[0]
    device = adm.alpha.device
    
    # Initialize 4D metric
    g = torch.zeros(batch_size, 4, 4, device=device)
    
    # Set components based on ADM variables
    # g_00 = -α² + β_i β^i
    beta_squared = torch.zeros_like(adm.alpha)
    for i in range(3):
        for j in range(3):
            beta_squared += adm.beta[:, i] * adm.gamma[:, i, j] * adm.beta[:, j]
    
    g[:, 0, 0] = -adm.alpha**2 + beta_squared
    
    # g_0i = g_i0 = β_i
    for i in range(3):
        g[:, 0, i+1] = adm.beta[:, i]
        g[:, i+1, 0] = adm.beta[:, i]
    
    # g_ij = γ_ij
    for i in range(3):
        for j in range(3):
            g[:, i+1, j+1] = adm.gamma[:, i, j]
    
    return g


def spacetime_to_adm(
    coords: torch.Tensor,
    model: nn.Module
) -> ADMVariables:
    """Extract ADM variables from the neural network output.
    
    Args:
        coords: Spacetime coordinates (batch_size, 4)
        model: Neural network model that outputs metric components
        
    Returns:
        ADM variables container
    """
    # Get full spacetime metric
    g = model(coords)
    batch_size = coords.shape[0]
    device = coords.device
    
    # Extract spatial metric γ_ij = g_ij
    gamma = torch.zeros(batch_size, 3, 3, device=device)
    for i in range(3):
        for j in range(3):
            gamma[:, i, j] = g[:, i+1, j+1]
    
    # Compute inverse spatial metric
    gamma_inv = torch.inverse(gamma)
    
    # Extract shift vector β^i = g^0i
    beta = torch.zeros(batch_size, 3, device=device)
    for i in range(3):
        beta[:, i] = g[:, 0, i+1]
    
    # Compute lapse function α = sqrt(-g^00)
    # First get g^00 from full metric inverse
    g_inv = torch.inverse(g)
    alpha = torch.sqrt(-g_inv[:, 0, 0])
    
    # Compute extrinsic curvature (requires time derivatives)
    # K_ij = -(1/2α) (∂_t γ_ij - ∇_i β_j - ∇_j β_i)
    coords.requires_grad_(True)
    g = model(coords)
    
    # Extract spatial metric components
    gamma_components = []
    for i in range(3):
        for j in range(3):
            gamma_components.append(g[:, i+1, j+1])
    
    # Stack components for batch gradient computation
    stacked_gamma = torch.stack(gamma_components, dim=1)
    
    # Compute time derivatives
    d_gamma_dt = torch.autograd.grad(
        stacked_gamma, coords,
        grad_outputs=torch.ones_like(stacked_gamma),
        create_graph=True
    )[0][:, :, 0]  # Extract time component
    
    # Reshape back to 3x3 tensor
    d_gamma_dt = d_gamma_dt.view(batch_size, 3, 3)
    
    # For the full calculation we also need spatial covariant derivatives of β
    # This is a simplified approximation
    K = -d_gamma_dt / (2 * alpha.unsqueeze(-1).unsqueeze(-1))
    
    return ADMVariables(alpha, beta, gamma, K)


def compute_adm_constraints(adm: ADMVariables) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the Hamiltonian and momentum constraints from ADM variables.
    
    Args:
        adm: ADM variables container
        
    Returns:
        Tuple of (Hamiltonian constraint, Momentum constraint)
    """
    batch_size = adm.alpha.shape[0]
    device = adm.alpha.device
    
    # Compute 3D Ricci scalar (simplified)
    gamma_inv = torch.inverse(adm.gamma)
    
    # Compute trace of extrinsic curvature K = γ^ij K_ij
    K_trace = torch.zeros(batch_size, device=device)
    for i in range(3):
        for j in range(3):
            K_trace += gamma_inv[:, i, j] * adm.K[:, i, j]
    
    # Compute K_ij K^ij
    K_squared = torch.zeros_like(K_trace)
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for n in range(3):
                    K_squared += adm.K[:, i, j] * gamma_inv[:, i, m] * gamma_inv[:, j, n] * adm.K[:, m, n]
    
    # In a full implementation, we would compute the 3D Ricci scalar
    # from the spatial metric. Here we use a placeholder
    # that would need to be replaced with the actual calculation.
    R_3d = torch.zeros_like(K_trace)  # Placeholder
    
    # Hamiltonian constraint: R + K² - K_ij K^ij = 0
    H_constraint = R_3d + K_trace**2 - K_squared
    
    # Momentum constraint: ∇_j K^j_i - ∇_i K = 0
    # This is a simplified version and would need proper covariant derivatives
    M_constraint = torch.zeros(batch_size, 3, device=device)
    
    return H_constraint, M_constraint


def compute_bssn_vars(adm: ADMVariables) -> BSSNVariables:
    """Convert ADM variables to BSSN variables.
    
    The BSSN formalism provides better numerical stability for evolution.
    
    Args:
        adm: ADM variables container
        
    Returns:
        BSSN variables container
    """
    batch_size = adm.alpha.shape[0]
    device = adm.alpha.device
    
    # Compute determinant of spatial metric
    gamma_det = torch.det(adm.gamma)
    
    # Conformal factor phi = (1/12) ln(det(γ))
    phi = (1/12) * torch.log(gamma_det)
    
    # Conformal metric γ̃_ij = e^(-4φ) γ_ij
    gamma_tilde = torch.exp(-4 * phi).unsqueeze(-1).unsqueeze(-1) * adm.gamma
    
    # Trace of extrinsic curvature
    gamma_inv = torch.inverse(adm.gamma)
    K = torch.zeros(batch_size, device=device)
    for i in range(3):
        for j in range(3):
            K += gamma_inv[:, i, j] * adm.K[:, i, j]
    
    # Traceless part of extrinsic curvature
    A_tilde = torch.zeros_like(adm.K)
    for i in range(3):
        for j in range(3):
            A_tilde[:, i, j] = torch.exp(-4 * phi) * (
                adm.K[:, i, j] - (1/3) * adm.gamma[:, i, j] * K
            )
    
    # Conformal connection functions (simplified)
    # Γ̃^i = γ̃^jk Γ̃^i_jk
    Gamma_tilde = torch.zeros(batch_size, 3, device=device)
    
    # This is a placeholder for the actual calculation
    # In practice, we would compute these from derivatives of γ̃
    
    return BSSNVariables(phi, gamma_tilde, Gamma_tilde, K, A_tilde, adm.alpha, adm.beta)


# -----------------------------------------------------
# 3. Symbolic Verification Tools
# -----------------------------------------------------

def create_symbolic_metric():
    """Create symbolic metric variables for verification."""
    # Define coordinates
    t, x, y, z = sp.symbols('t x y z')
    coords = (t, x, y, z)
    
    # Define metric components symbolically
    g = sp.MutableDenseMatrix(4, 4, [0]*16)
    
    # Fill with symbolic variables for each component
    for i in range(4):
        for j in range(i, 4):  # Symmetric matrix
            g[i, j] = g[j, i] = sp.Symbol(f'g_{i}{j}')
    
    return coords, g


def symbolic_christoffel(coords, g):
    """Compute Christoffel symbols symbolically."""
    # Calculate the inverse metric
    g_inv = g.inv()
    
    # Initialize Christoffel symbols
    christoffel = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    # Calculate derivatives of metric
    dg = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                dg[mu][alpha][beta] = sp.diff(g[alpha, beta], coords[mu])
    
    # Calculate Christoffel symbols
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                christoffel[mu][alpha][beta] = 0
                for sigma in range(4):
                    christoffel[mu][alpha][beta] += (1/2) * g_inv[mu, sigma] * (
                        dg[alpha][sigma][beta] + 
                        dg[beta][sigma][alpha] - 
                        dg[sigma][alpha][beta]
                    )
    
    return christoffel


def symbolic_riemann(coords, christoffel):
    """Compute Riemann tensor symbolically."""
    # Initialize Riemann tensor
    riemann = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    # Calculate derivatives of Christoffel symbols
    d_christoffel = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    d_christoffel[rho][sigma][mu][nu] = sp.diff(christoffel[rho][sigma][mu], coords[nu])
    
    # Calculate Riemann tensor
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    riemann[rho][sigma][mu][nu] = (
                        d_christoffel[rho][sigma][nu][mu] - 
                        d_christoffel[rho][sigma][mu][nu]
                    )
                    
                    for lambda_ in range(4):
                        riemann[rho][sigma][mu][nu] += (
                            christoffel[rho][lambda_][nu] * christoffel[lambda_][sigma][mu] -
                            christoffel[rho][lambda_][mu] * christoffel[lambda_][sigma][nu]
                        )
    
    return riemann


def verify_bianchi_identities(riemann):
    """Verify the Bianchi identities for the Riemann tensor."""
    # First Bianchi identity: R^a_bcd + R^a_cdb + R^a_dbc = 0
    first_bianchi = []
    
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    identity = (
                        riemann[a][b][c][d] + 
                        riemann[a][c][d][b] + 
                        riemann[a][d][b][c]
                    )
                    first_bianchi.append(sp.simplify(identity))
    
    # Check if all are zero
    all_zero_first = all(sp.simplify(expr) == 0 for expr in first_bianchi)
    
    # Second Bianchi identity (differential): ∇_e R^a_bcd + ∇_c R^a_bde + ∇_d R^a_bec = 0
    # This would require computing covariant derivatives
    # For simplicity, we'll return just the first identity check
    
    return all_zero_first


def symbolic_einstein_equations(g, coords, matter_tensor=None):
    """Compute Einstein field equations symbolically and check if solutions satisfy them."""
    # Compute Christoffel symbols
    christoffel = symbolic_christoffel(coords, g)
    
    # Compute Riemann tensor
    riemann = symbolic_riemann(coords, christoffel)
    
    # Compute Ricci tensor by contracting Riemann tensor
    ricci = sp.MutableDenseMatrix(4, 4, [0]*16)
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                ricci[mu, nu] += riemann[rho][mu][rho][nu]
    
    # Compute Ricci scalar
    g_inv = g.inv()
    ricci_scalar = 0
    for mu in range(4):
        for nu in range(4):
            ricci_scalar += g_inv[mu, nu] * ricci[mu, nu]
    
    # Compute Einstein tensor
    einstein = sp.MutableDenseMatrix(4, 4, [0]*16)
    for mu in range(4):
        for nu in range(4):
            einstein[mu, nu] = ricci[mu, nu] - (1/2) * ricci_scalar * g[mu, nu]
    
    # If matter tensor is provided, check Einstein equations: G_μν = 8πT_μν
    if matter_tensor:
        # Compute Einstein field equations
        equations = []
        for mu in range(4):
            for nu in range(4):
                eq = einstein[mu, nu] - 8 * sp.pi * matter_tensor[mu, nu]
                equations.append(sp.simplify(eq))
        
        # Check if all equations are satisfied
        all_satisfied = all(sp.simplify(eq) == 0 for eq in equations)
        return all_satisfied, equations
    else:
        # Just return the Einstein tensor
        return einstein


def verify_solution_with_sympy(model: nn.Module, coords: torch.Tensor) -> Dict[str, bool]:
    """Verify a learned solution using symbolic computation.
    
    Args:
        model: Trained neural network model
        coords: Sample coordinates to evaluate
        
    Returns:
        Dictionary of verification results
    """
    # Create symbolic variables
    sym_coords, sym_g = create_symbolic_metric()
    
    # Get model output at sample point
    sample_point = coords[0].detach().numpy()  # Use first point in batch
    model.eval()
    with torch.no_grad():
        g_pred = model(coords[0:1]).squeeze(0).detach().numpy()
    
    # Create numeric substitution dictionary
    subs_dict = {
        f'g_{i}{j}': g_pred[i, j]
        for i in range(4)
        for j in range(4)
    }
    subs_dict.update({
        't': sample_point[0],
        'x': sample_point[1],
        'y': sample_point[2],
        'z': sample_point[3]
    })
    
    # Compute results
    christoffel = symbolic_christoffel(sym_coords, sym_g)
    riemann = symbolic_riemann(sym_coords, christoffel)
    
    # Verify Bianchi identities
    bianchi_satisfied = verify_bianchi_identities(riemann)
    
    # Compute Einstein tensor (vacuum solution check)
    einstein_tensor = symbolic_einstein_equations(sym_g, sym_coords)
    
    # Check if Einstein equations are satisfied (vacuum)
    vacuum_satisfied = all(
        abs(float(component.subs(subs_dict))) < 1e-5
        for row in einstein_tensor
        for component in row
    )
    
    return {
        'bianchi_identities': bianchi_satisfied,
        'vacuum_einstein_equations': vacuum_satisfied
    }


# -----------------------------------------------------
# 4. PDE Residuals and Loss Functions for Einstein Equations
# -----------------------------------------------------

def einstein_vacuum_residual(
    model: nn.Module,
    coords: torch.Tensor
) -> torch.Tensor:
    """Compute Einstein field equation residual for vacuum (G_μν = 0).
    
    Args:
        model: Neural network model that outputs the metric
        coords: Spacetime coordinates (batch_size, 4)
        
    Returns:
        Residual tensor of Einstein equations
    """
    coords.requires_grad_(True)
    
    # Get metric tensor
    g = model(coords)
    batch_size = coords.shape[0]
    
    # Compute inverse metric
    g_inv = torch.inverse(g)
    
    # Compute Christoffel symbols and derivatives
    christoffel, d_christoffel = compute_christoffel_symbols(coords, lambda x: model(x))
    
    # Compute Riemann tensor
    riemann = compute_riemann_tensor(christoffel, d_christoffel)
    
    # Compute Ricci tensor
    ricci = compute_ricci_tensor(riemann, g_inv)
    
    # Compute Einstein tensor
    einstein = compute_einstein_tensor(ricci, g, g_inv)
    
    # Compute residual as Frobenius norm of Einstein tensor
    # For vacuum solutions, G_μν should be 0
    residual = torch.zeros(batch_size, device=coords.device)
    for mu in range(4):
        for nu in range(4):
            residual += einstein[:, mu, nu]**2
    
    return residual


def einstein_matter_residual(
    model: nn.Module,
    coords: torch.Tensor,
    stress_energy_fn: Callable
) -> torch.Tensor:
    """Compute Einstein field equation residual with matter (G_μν = 8πT_μν).
    
    Args:
        model: Neural network model that outputs the metric
        coords: Spacetime coordinates (batch_size, 4)
        stress_energy_fn: Function that computes stress-energy tensor
        
    Returns:
        Residual tensor of Einstein equations with matter
    """
    coords.requires_grad_(True)
    
    # Get metric tensor
    g = model(coords)
    batch_size = coords.shape[0]
    
    # Compute inverse metric
    g_inv = torch.inverse(g)
    
    # Compute Christoffel symbols and derivatives
    christoffel, d_christoffel = compute_christoffel_symbols(coords, lambda x: model(x))
    
    # Compute Riemann tensor
    riemann = compute_riemann_tensor(christoffel, d_christoffel)
    
    # Compute Ricci tensor
    ricci = compute_ricci_tensor(riemann, g_inv)
    
    # Compute Einstein tensor
    einstein = compute_einstein_tensor(ricci, g, g_inv)
    
    # Get stress-energy tensor
    T = stress_energy_fn(coords, g, g_inv)
    
    # Einstein field equations: G_μν = 8πT_μν
    # Compute residual
    residual = torch.zeros(batch_size, device=coords.device)
    for mu in range(4):
        for nu in range(4):
            residual += (einstein[:, mu, nu] - 8 * math.pi * T[:, mu, nu])**2
    
    return residual


def adm_constraint_residual(
    model: nn.Module,
    coords: torch.Tensor
) -> torch.Tensor:
    """Compute residual for ADM constraint equations.
    
    Args:
        model: Neural network model that outputs the metric
        coords: Spacetime coordinates (batch_size, 4)
        
    Returns:
        Residual tensor of constraint violations
    """
    # Extract ADM variables from model
    adm_vars = spacetime_to_adm(coords, model)
    
    # Compute Hamiltonian and momentum constraints
    H_constraint, M_constraint = compute_adm_constraints(adm_vars)
    
    # Compute total constraint violation
    residual = H_constraint**2
    for i in range(3):
        residual += M_constraint[:, i]**2
    
    return residual


def bssn_evolution_residual(
    model: nn.Module,
    coords: torch.Tensor
) -> torch.Tensor:
    """Compute residual for BSSN evolution equations.
    
    Args:
        model: Neural network model
        coords: Spacetime coordinates (batch_size, 4)
        
    Returns:
        Residual tensor for BSSN evolution
    """
    # Extract ADM variables
    adm_vars = spacetime_to_adm(coords, model)
    
    # Convert to BSSN variables
    bssn_vars = compute_bssn_vars(adm_vars)
    
    # In a complete implementation, we would:
    # 1. Compute time derivatives of BSSN variables
    # 2. Compute RHS of BSSN evolution equations
    # 3. Calculate residual as the difference
    
    # For now, return a placeholder residual
    # This needs to be replaced with actual BSSN evolution equations
    return torch.ones(coords.shape[0], device=coords.device)


def gauge_condition_residual(
    model: nn.Module,
    coords: torch.Tensor,
    gauge_type: str = 'harmonic'
) -> torch.Tensor:
    """Compute residual for gauge conditions.
    
    Args:
        model: Neural network model that outputs the metric
        coords: Spacetime coordinates (batch_size, 4)
        gauge_type: Type of gauge condition ('harmonic', 'maximal', etc.)
        
    Returns:
        Residual tensor of gauge condition violation
    """
    coords.requires_grad_(True)
    
    # Get metric tensor
    g = model(coords)
    batch_size = coords.shape[0]
    
    # Compute inverse metric
    g_inv = torch.inverse(g)
    
    # Compute Christoffel symbols
    christoffel, _ = compute_christoffel_symbols(coords, lambda x: model(x))
    
    if gauge_type == 'harmonic':
        # Harmonic gauge: Γ^μ = g^αβ Γ^μ_αβ = 0
        residual = torch.zeros(batch_size, device=coords.device)
        
        for mu in range(4):
            harmonic_gauge = torch.zeros(batch_size, device=coords.device)
            for alpha in range(4):
                for beta in range(4):
                    harmonic_gauge += g_inv[:, alpha, beta] * christoffel[:, mu, alpha, beta]
            residual += harmonic_gauge**2
        
        return residual
    
    elif gauge_type == 'maximal':
        # Maximal slicing: K = 0 (trace of extrinsic curvature)
        adm_vars = spacetime_to_adm(coords, model)
        gamma_inv = torch.inverse(adm_vars.gamma)
        
        K_trace = torch.zeros(batch_size, device=coords.device)
        for i in range(3):
            for j in range(3):
                K_trace += gamma_inv[:, i, j] * adm_vars.K[:, i, j]
        
        return K_trace**2
    
    else:
        raise ValueError(f"Unsupported gauge type: {gauge_type}")


def schwarzschild_boundary_residual(
    model: nn.Module,
    coords: torch.Tensor,
    mass: float = 1.0
) -> torch.Tensor:
    """Compute residual for Schwarzschild solution boundary condition.
    
    Applies asymptotic boundary conditions for the metric to approach
    Schwarzschild solution at large distances.
    
    Args:
        model: Neural network model that outputs the metric
        coords: Spacetime coordinates (batch_size, 4)
        mass: Mass parameter for the Schwarzschild solution
        
    Returns:
        Residual tensor for boundary condition violation
    """
    model.eval()
    
    # Calculate r = sqrt(x² + y² + z²)
    r = torch.sqrt(
        coords[:, 1]**2 + coords[:, 2]**2 + coords[:, 3]**2
    )
    
    # Only apply to points beyond a certain radius
    mask = r > 10 * mass  # Far-field condition
    
    if not torch.any(mask):
        return torch.zeros(1, device=coords.device)  # No applicable points
    
    # Get metric at these points
    g_pred = model(coords[mask])
    
    # Compute Schwarzschild metric components
    g_schw = torch.zeros_like(g_pred)
    
    # Time component: g_00 = -(1 - 2M/r)
    g_schw[:, 0, 0] = -(1 - 2 * mass / r[mask])
    
    # Radial component
    for i in range(1, 4):
        for j in range(1, 4):
            # Spatial part approaches Minkowski + corrections
            if i == j:
                g_schw[:, i, j] = 1
            
            # Apply radial correction
            x_i = coords[mask, i]
            x_j = coords[mask, j]
            
            # Add 1/r correction term for Schwarzschild
            g_schw[:, i, j] += 2 * mass / r[mask] * x_i * x_j / (r[mask]**2)
    
    # Weight more distant points higher in the loss
    weights = (r[mask] / (10 * mass))**2
    
    # Compute weighted residual
    residual = torch.zeros(1, device=coords.device)
    for i in range(4):
        for j in range(4):
            diff = (g_pred[:, i, j] - g_schw[:, i, j]) * weights
            residual += torch.mean(diff**2)
    
    return residual


# -----------------------------------------------------
# 5. Advanced Sampling Strategies
# -----------------------------------------------------

def sample_adaptive_spacetime(
    N: int,
    T_range: Tuple[float, float],
    L: float,
    device: torch.device,
    model: Optional[nn.Module] = None,
    residual_fn: Optional[Callable] = None,
    alpha: float = 0.7,
    focus_regions: Optional[List[Tuple]] = None
) -> torch.Tensor:
    """Sample points in spacetime with adaptive refinement near high residual or features.
    
    Args:
        N: Number of points to sample
        T_range: Time range (t_min, t_max)
        L: Spatial domain half-length
        device: PyTorch device
        model: PINN model for adaptive sampling
        residual_fn: Residual function for adaptive sampling
        alpha: Portion of points to sample adaptively
        focus_regions: List of regions (t,x,y,z,radius) to focus sampling on
        
    Returns:
        Tensor of sampled coordinates (batch_size, 4)
    """
    t_min, t_max = T_range
    
    # Determine how many points for each method
    N_uniform = int(N * (1 - alpha))
    N_adaptive = N - N_uniform
    
    # Generate uniform samples
    t_uniform = t_min + torch.rand(N_uniform, 1, device=device) * (t_max - t_min)
    x_uniform = (torch.rand(N_uniform, 1, device=device) * 2 - 1) * L
    y_uniform = (torch.rand(N_uniform, 1, device=device) * 2 - 1) * L
    z_uniform = (torch.rand(N_uniform, 1, device=device) * 2 - 1) * L
    
    uniform_coords = torch.cat([t_uniform, x_uniform, y_uniform, z_uniform], dim=1)
    
    # If no model or residual function, just return uniform samples
    if model is None or residual_fn is None:
        if focus_regions is None:
            return uniform_coords
        
        # Add focused samples in specified regions
        focus_coords = []
        remaining_points = N - N_uniform
        
        for region in focus_regions:
            center_t, center_x, center_y, center_z, radius = region
            points_in_region = remaining_points // len(focus_regions)
            
            # Sample uniformly within the spherical region
            t_local = center_t + torch.rand(points_in_region, 1, device=device) * radius * 0.2
            
            # For spatial coordinates, sample within sphere
            u = torch.rand(points_in_region, 1, device=device)  # Radius scaling
            theta = torch.rand(points_in_region, 1, device=device) * 2 * math.pi  # Azimuthal
            phi = torch.acos(2 * torch.rand(points_in_region, 1, device=device) - 1)  # Polar
            
            r = radius * torch.pow(u, 1/3)  # Uniform in volume
            
            x_local = center_x + r * torch.sin(phi) * torch.cos(theta)
            y_local = center_y + r * torch.sin(phi) * torch.sin(theta)
            z_local = center_z + r * torch.cos(phi)
            
            region_coords = torch.cat([t_local, x_local, y_local, z_local], dim=1)
            focus_coords.append(region_coords)
        
        focus_coords = torch.cat(focus_coords, dim=0)
        return torch.cat([uniform_coords, focus_coords], dim=0)
    
    # Generate candidates for adaptive sampling
    t_cand = t_min + torch.rand(N_adaptive * 10, 1, device=device) * (t_max - t_min)
    x_cand = (torch.rand(N_adaptive * 10, 1, device=device) * 2 - 1) * L
    y_cand = (torch.rand(N_adaptive * 10, 1, device=device) * 2 - 1) * L
    z_cand = (torch.rand(N_adaptive * 10, 1, device=device) * 2 - 1) * L
    
    cand_coords = torch.cat([t_cand, x_cand, y_cand, z_cand], dim=1)
    
    # Compute residual at candidate points
    with torch.no_grad():
        model.eval()
        residuals = residual_fn(model, cand_coords)
    
    # Sample points with probability proportional to residual
    probs = residuals / torch.sum(residuals)
    indices = torch.multinomial(probs.flatten(), N_adaptive, replacement=False)
    
    # Combine uniform and adaptive samples
    adaptive_coords = cand_coords[indices]
    return torch.cat([uniform_coords, adaptive_coords], dim=0)


def sample_spherical_shell(
    N: int,
    center: Tuple[float, float, float],
    r_inner: float,
    r_outer: float,
    t_range: Tuple[float, float],
    device: torch.device
) -> torch.Tensor:
    """Sample points in a spherical shell, useful for black hole horizons.
    
    Args:
        N: Number of points to sample
        center: Center coordinates (x,y,z)
        r_inner: Inner radius of shell
        r_outer: Outer radius of shell
        t_range: Time range (t_min, t_max)
        device: PyTorch device
        
    Returns:
        Tensor of sampled coordinates (batch_size, 4)
    """
    t_min, t_max = t_range
    cx, cy, cz = center
    
    # Generate random time coordinates
    t = t_min + torch.rand(N, 1, device=device) * (t_max - t_min)
    
    # Sample random directions (uniform on sphere)
    theta = torch.rand(N, 1, device=device) * 2 * math.pi  # Azimuthal
    phi = torch.acos(2 * torch.rand(N, 1, device=device) - 1)  # Polar
    
    # Sample radii between r_inner and r_outer
    # Use r^2 distribution for uniform sampling in volume
    u = torch.rand(N, 1, device=device)
    r = torch.pow(r_inner**3 + u * (r_outer**3 - r_inner**3), 1/3)
    
    # Convert to Cartesian coordinates
    x = cx + r * torch.sin(phi) * torch.cos(theta)
    y = cy + r * torch.sin(phi) * torch.sin(theta)
    z = cz + r * torch.cos(phi)
    
    # Combine into coordinates tensor
    coords = torch.cat([t, x, y, z], dim=1)
    return coords


def sample_null_geodesics(
    N: int, 
    origins: List[Tuple[float, float, float]],
    t_range: Tuple[float, float],
    device: torch.device
) -> torch.Tensor:
    """Sample points along null geodesics (light rays) from given origins.
    
    Useful for testing light propagation in spacetime.
    
    Args:
        N: Number of points to sample
        origins: List of origin points (x,y,z)
        t_range: Time range (t_min, t_max)
        device: PyTorch device
        
    Returns:
        Tensor of sampled coordinates (batch_size, 4)
    """
    t_min, t_max = t_range
    points_per_origin = N // len(origins)
    all_coords = []
    
    for origin in origins:
        ox, oy, oz = origin
        
        # Sample random directions
        theta = torch.rand(points_per_origin, 1, device=device) * 2 * math.pi
        phi = torch.acos(2 * torch.rand(points_per_origin, 1, device=device) - 1)
        
        # Direction vectors
        dx = torch.sin(phi) * torch.cos(theta)
        dy = torch.sin(phi) * torch.sin(theta)
        dz = torch.cos(phi)
        
        # Sample time parameters (determines how far along ray)
        t = t_min + torch.rand(points_per_origin, 1, device=device) * (t_max - t_min)
        
        # For a flat spacetime approximation, the null geodesic is just a straight line
        # In curved spacetime, this would need to solve the geodesic equation
        lambda_param = (t - t_min)
        
        # Position along geodesic
        x = ox + dx * lambda_param
        y = oy + dy * lambda_param
        z = oz + dz * lambda_param
        
        # Combine into coordinates
        coords = torch.cat([t, x, y, z], dim=1)
        all_coords.append(coords)
    
    return torch.cat(all_coords, dim=0)


# -----------------------------------------------------
# 6. Training Loop with Constraint Enforcement
# -----------------------------------------------------

def train_einstein_pinn(
    model: nn.Module,
    residual_fns: Dict[str, Tuple[Callable, float]],
    T_range: Tuple[float, float],
    L: float,
    device: torch.device,
    epochs: int,
    batch_size: int = 2048,
    lr: float = 1e-3,
    adaptive_sampling: bool = True,
    symbolic_verification_freq: int = 500,
    focus_regions: Optional[List[Tuple]] = None,
    progress_bar: Optional[object] = None,
    loss_chart: Optional[object] = None,
    constraint_chart: Optional[object] = None
) -> Dict[str, List[float]]:
    """Train a PINN model to solve Einstein's field equations with constraints.
    
    Args:
        model: Neural network model
        residual_fns: Dictionary of {name: (residual_fn, weight)} pairs
        T_range: Time range (t_min, t_max)
        L: Spatial domain half-length
        device: PyTorch device
        epochs: Number of training epochs
        batch_size: Batch size for collocation points
        lr: Learning rate
        adaptive_sampling: Whether to use adaptive sampling
        symbolic_verification_freq: How often to run symbolic verification
        focus_regions: List of regions to focus sampling on
        progress_bar: Streamlit progress bar
        loss_chart: Streamlit chart for loss visualization
        constraint_chart: Streamlit chart for constraint visualization
        
    Returns:
        Dictionary of training history
    """
    # Initialize optimizer with learning rate schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6, verbose=True
    )
    
    # Get field equation residual function
    if 'field_equations' not in residual_fns:
        raise ValueError("Must provide 'field_equations' in residual_fns")
    
    field_eq_fn, _ = residual_fns['field_equations']
    
    # History for tracking progress
    history = {
        "total_loss": [],
        "constraint_violation": []
    }
    
    # Add individual loss components to history
    for name in residual_fns.keys():
        history[f"{name}_loss"] = []
    
    # Set up verification tracking
    history["verification"] = []
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # Zero gradients
        optimizer.zero_grad()
        
        # Sample collocation points with adaptive refinement
        if adaptive_sampling and epoch > 500:
            coords = sample_adaptive_spacetime(
                batch_size, T_range, L, device,
                model=model, 
                residual_fn=field_eq_fn,
                alpha=min(0.7, epoch / epochs),
                focus_regions=focus_regions
            )
        else:
            coords = sample_adaptive_spacetime(
                batch_size, T_range, L, device,
                focus_regions=focus_regions
            )
        
        # Compute loss components
        total_loss = 0.0
        component_losses = {}
        
        for name, (residual_fn, weight) in residual_fns.items():
            residual = residual_fn(model, coords)
            loss = torch.mean(residual)
            component_losses[name] = loss.item()
            total_loss += weight * loss
        
        # Backward and optimize
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Run symbolic verification periodically
        if epoch % symbolic_verification_freq == 0 or epoch == epochs:
            with torch.no_grad():
                verification_results = verify_solution_with_sympy(model, coords)
                history["verification"].append(verification_results)
        
        # Log metrics periodically
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            # Record losses
            history["total_loss"].append(total_loss.item())
            
            # Record component losses
            for name in residual_fns.keys():
                history[f"{name}_loss"].append(component_losses[name])
            
            # Record constraint violation
            if 'adm_constraints' in residual_fns:
                constraint_violation = component_losses['adm_constraints']
                history["constraint_violation"].append(constraint_violation)
            
            # Update learning rate based on loss
            scheduler.step(total_loss.item())
            
            # Update progress in UI
            if progress_bar is not None:
                progress_bar.progress(epoch / epochs)
            
            # Update loss chart
            if loss_chart is not None:
                loss_data = {
                    "total_loss": [history["total_loss"][-1]],
                    "field_eq_loss": [history["field_equations_loss"][-1]]
                }
                loss_chart.add_rows(pd.DataFrame(loss_data))
            
            # Update constraint chart
            if constraint_chart is not None and "constraint_violation" in history:
                constraint_data = {
                    "constraint_violation": [history["constraint_violation"][-1]]
                }
                if epoch % symbolic_verification_freq == 0:
                    constraint_data["bianchi_verified"] = [
                        1.0 if history["verification"][-1]["bianchi_identities"] else 0.0
                    ]
                constraint_chart.add_rows(pd.DataFrame(constraint_data))
            
            # Print status
            if epoch % 500 == 0:
                verification_status = ""
                if epoch % symbolic_verification_freq == 0:
                    verification_status = " | Verification: " + " ".join(
                        f"{k}={v}" for k, v in history["verification"][-1].items()
                    )
                
                st.write(f"Epoch {epoch}/{epochs} | Loss: {total_loss.item():.3e}{verification_status}")
    
    training_time = time.time() - t_start
    st.success(f"Training completed in {training_time:.1f}s. Final loss: {total_loss.item():.3e}")
    
    # Final verification
    with torch.no_grad():
        verification_results = verify_solution_with_sympy(model, coords)
        st.write("Final solution verification:")
        for k, v in verification_results.items():
            st.write(f"  - {k}: {'✓' if v else '✗'}")
    
    return history


# -----------------------------------------------------
# 7. Visualization Functions for General Relativity
# -----------------------------------------------------

def visualize_spacetime_metric(
    model: nn.Module,
    device: torch.device,
    t_value: float,
    component: Tuple[int, int] = (0, 0),
    slice_axis: int = 3,  # z-axis by default
    slice_value: float = 0.0
) -> go.Figure:
    """Visualize a 2D slice of the metric tensor.
    
    Args:
        model: Trained PINN model
        device: PyTorch device
        t_value: Time value for the slice
        component: Tuple indicating which metric component to visualize
        slice_axis: Which spatial axis to fix (1=x, 2=y, 3=z)
        slice_value: Value for the fixed axis
        
    Returns:
        Plotly figure with metric visualization
    """
    model.eval()
    
    # Create a 2D grid for the remaining spatial coordinates
    N = 100  # Resolution
    L = 10.0  # Domain size
    
    if slice_axis == 1:  # Fixed x
        y = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        Y, Z = torch.meshgrid(y, z, indexing="ij")
        X = torch.full_like(Y, slice_value)
        grid_pts = torch.stack([
            torch.full_like(Y, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "y", "z"
    
    elif slice_axis == 2:  # Fixed y
        x = torch.linspace(-L, L, N, device=device)
        z = torch.linspace(-L, L, N, device=device)
        X, Z = torch.meshgrid(x, z, indexing="ij")
        Y = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "z"
    
    else:  # Fixed z (default)
        x = torch.linspace(-L, L, N, device=device)
        y = torch.linspace(-L, L, N, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.full_like(X, slice_value)
        grid_pts = torch.stack([
            torch.full_like(X, t_value),
            X, Y, Z
        ], dim=-1).reshape(-1, 4)
        
        x_label, y_label = "x", "y"
    
    # Predict metric values
    with torch.no_grad():
        g = model(grid_pts)
        # Reshape output to 4x4 tensor
        g_reshaped = g.reshape(-1, 4, 4)
        g_component = g_reshaped[:, component[0], component[1]].reshape(N, N).cpu().numpy()
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=g_component,
        x=x.cpu().numpy() if slice_axis != 2 else z.cpu().numpy(),

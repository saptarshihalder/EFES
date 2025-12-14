import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd

# --- Configuration ---
class HyperParameters:
    def __init__(self):
        self.domain_radius = 3.0
        self.hidden_dim = 64
        self.omega_0 = 30.0  # SIREN frequency
        self.learning_rate = 1e-4
        self.batch_size = 1024
        self.epochs = 800 # Reduced for active learning speed
        
        # Loss weights
        self.lambda_hamiltonian = 1.0  # Physics (Einstein)
        self.lambda_momentum = 1.0     # Momentum constraint
        self.lambda_boundary = 5.0     # Asymptotic flatness
        self.lambda_topology = 10.0    # Forcing shape (warp/wormhole)
        self.lambda_complexity = 0.1   # Regularization
        self.lambda_static = 0.1       # Penalty for time dependence
        self.lambda_signature = 0.1    # Enforce g00 < 0
        self.lambda_determinant = 1.0  # Prevent det(g) near 0
        
        # Physics Parameters (Meta-learnable)
        self.exotic_rho_0 = -0.1       # Density for exotic matter
        self.verbose = False           # Less print noise for active loop

    def update(self, **kwargs):
        """Update params from a dictionary."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                pass # Ignore unknown params for now

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# --- Helper: Derivatives ---
def diff(y, x):
    """Compute dy/dx using torch autograd."""
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grad

# --- 1. Differential Geometry Engines (4D and 3D) ---

class GRGeometry:
    """
    Handles full 4D Differential Geometry calculations.
    """
    @staticmethod
    def get_inverse_metric(g):
        return torch.inverse(g)

    @staticmethod
    def christoffel_symbols(g, coords):
        B, D, _ = g.shape
        g_inv = GRGeometry.get_inverse_metric(g)
        g_flat = g.view(B, -1)
        dg_dcoords = []
        for i in range(D * D):
            dg_dcoords.append(diff(g_flat[:, i].unsqueeze(1), coords))
        dg_dx = torch.stack(dg_dcoords, dim=1).view(B, D, D, D)
        
        gamma = torch.zeros(B, D, D, D, device=g.device)
        for i in range(D):
            for j in range(D):
                for k in range(D):
                    term = 0.0
                    for l in range(D):
                        term += 0.5 * g_inv[:, k, l] * (
                            dg_dx[:, i, l, j] + 
                            dg_dx[:, j, l, i] - 
                            dg_dx[:, i, j, l]
                        )
                    gamma[:, k, i, j] = term
        return gamma

    @staticmethod
    def ricci_tensor(g, coords):
        gamma = GRGeometry.christoffel_symbols(g, coords)
        B, D, _, _ = gamma.shape
        gamma_flat = gamma.view(B, -1)
        dgamma_dcoords = []
        for i in range(D**3):
            dgamma_dcoords.append(diff(gamma_flat[:, i].unsqueeze(1), coords))
        dgamma_dx = torch.stack(dgamma_dcoords, dim=1).view(B, D, D, D, D)
        
        R_uv = torch.zeros(B, D, D, device=g.device)
        for u in range(D):
            for v in range(D):
                t1 = 0; t2 = 0; t3 = 0; t4 = 0
                for a in range(D):
                    t1 += dgamma_dx[:, a, u, a, v]
                    t2 += dgamma_dx[:, a, u, v, a]
                    for b in range(D):
                        t3 += gamma[:, a, b, a] * gamma[:, b, u, v]
                        t4 += gamma[:, a, b, v] * gamma[:, b, u, a]
                R_uv[:, u, v] = t1 - t2 + t3 - t4
        return R_uv

    @staticmethod
    def einstein_tensor(g, coords):
        R_uv = GRGeometry.ricci_tensor(g, coords)
        g_inv = GRGeometry.get_inverse_metric(g)
        R_scalar = torch.einsum('bij,bij->b', g_inv, R_uv).unsqueeze(1).unsqueeze(2)
        G_uv = R_uv - 0.5 * R_scalar * g
        return G_uv


class DifferentialGeometry:
    """Helper for 3D spatial slices (used for topology shaping/constraints)."""
    @staticmethod
    def expansion_scalar(shift, coords_spatial):
        div = 0
        for i in range(3):
            grad = diff(shift[:, i].unsqueeze(1), coords_spatial)
            div += grad[:, i]
        return div.unsqueeze(1)

    @staticmethod
    def ricci_scalar(spatial_metric, coords_spatial):
        R_proxy = torch.zeros(spatial_metric.shape[0], 1, device=spatial_metric.device)
        for i in range(3):
            comp = spatial_metric[:, i, i].unsqueeze(1)
            grads = diff(comp, coords_spatial)
            laplacian = 0
            for k in range(3):
                laplacian += diff(grads[:, k].unsqueeze(1), coords_spatial)[:, k]
            R_proxy += laplacian.unsqueeze(1)
        return R_proxy

# --- 2. Neural Network Architecture ---

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MetricEstimator(nn.Module):
    def __init__(self, config: HyperParameters):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(4, config.hidden_dim, config.omega_0, is_first=True),
            SineLayer(config.hidden_dim, config.hidden_dim, config.omega_0),
            SineLayer(config.hidden_dim, config.hidden_dim, config.omega_0),
            SineLayer(config.hidden_dim, config.hidden_dim, config.omega_0),
            SineLayer(config.hidden_dim, config.hidden_dim, config.omega_0),
            SineLayer(config.hidden_dim, config.hidden_dim, config.omega_0),
            nn.Linear(config.hidden_dim, 10) 
        )
        
    def forward(self, coords):
        output = self.net(coords)
        lapse = F.softplus(output[:, 0]) + 0.1 
        shift = output[:, 1:4]
        L_raw = output[:, 4:] 
        batch_size = coords.shape[0]
        L = torch.zeros(batch_size, 3, 3, device=device)
        L[:, 0, 0] = F.softplus(L_raw[:, 0]) + 0.1
        L[:, 1, 1] = F.softplus(L_raw[:, 1]) + 0.1
        L[:, 2, 2] = F.softplus(L_raw[:, 2]) + 0.1
        L[:, 1, 0] = L_raw[:, 3]
        L[:, 2, 0] = L_raw[:, 4]
        L[:, 2, 1] = L_raw[:, 5]
        spatial_metric = torch.bmm(L, L.transpose(1, 2))
        return lapse, shift, spatial_metric

# --- 3. Physics & Loss Functions ---

def build_4d_metric(lapse, shift, spatial_metric):
    B = lapse.shape[0]
    shift_vec = shift.unsqueeze(2) 
    shift_cov = torch.bmm(spatial_metric, shift_vec).squeeze(2) 
    shift_sq = torch.sum(shift * shift_cov, dim=1)
    g00 = -(lapse**2) + shift_sq
    
    metric_4d = torch.zeros(B, 4, 4, device=lapse.device)
    metric_4d[:, 0, 0] = g00
    metric_4d[:, 0, 1:] = shift_cov
    metric_4d[:, 1:, 0] = shift_cov
    metric_4d[:, 1:, 1:] = spatial_metric
    return metric_4d

def exotic_fluid_T(coords_4d, rho0=-0.1):
    B = coords_4d.shape[0]
    T = torch.zeros(B, 4, 4, device=coords_4d.device)
    T[:, 0, 0] = rho0
    return T

def einstein_residual_loss(metric_4d, coords_4d, T_mu_nu_fn=None):
    G = GRGeometry.einstein_tensor(metric_4d, coords_4d)
    if T_mu_nu_fn is None:
        T = torch.zeros_like(G)
    else:
        T = T_mu_nu_fn(coords_4d)
    residual = G - 8 * np.pi * T
    loss = torch.mean(residual**2)
    return loss, residual

def time_dependence_penalty(metric_4d, coords_4d):
    B, D, _ = metric_4d.shape
    g_flat = metric_4d.view(B, -1)
    penalty = 0.0
    for i in range(D * D):
        grads = diff(g_flat[:, i].unsqueeze(1), coords_4d)
        dt_g = grads[:, 0]
        penalty += torch.mean(dt_g**2)
    return penalty

def momentum_constraint(theta, coords_spatial):
    return torch.mean(theta**2)

def boundary_loss(spatial_metric, coords_spatial):
    """Penalizes deviation from Minkowski metric at large radii (asymptotic flatness)."""
    r = torch.sqrt(torch.sum(coords_spatial**2, dim=1, keepdim=True))
    # At large r, spatial metric should approach identity
    identity = torch.eye(3, device=spatial_metric.device).unsqueeze(0).expand(spatial_metric.shape[0], -1, -1)
    # Weight by 1/r to focus on boundary regions
    weight = torch.exp(-r / 2.0)  # Exponential weight favoring outer regions
    deviation = torch.mean((spatial_metric - identity)**2, dim=(1, 2), keepdim=True)
    loss = torch.mean(weight * deviation)
    return loss

# --- 4. Feature Extraction & Surrogate Model ---

def extract_features(coords, lapse, rho, G_res_norm):
    """Computes embedding of the solution."""
    def radial_moments(v):
        return [float(v.min()), float(v.max()), float(v.mean()), float(np.mean(np.abs(v)))]

    lapse_m = radial_moments(lapse)
    rho_m   = radial_moments(rho)

    feats = np.array([
        *lapse_m,          # 4 numbers
        *rho_m,            # 4 numbers
        G_res_norm,        # 1 number
    ], dtype=np.float32)
    return feats

class SurrogateModel(nn.Module):
    """
    Predicts the 'Interest Score' of a hyperparameter set.
    Input: [rho_0, lambda_topology, radius, omega] (normalized)
    Output: Estimated Interest Score
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class ActiveSearchPolicy:
    def __init__(self, bounds):
        self.bounds = bounds # Dict of ranges {param: (min, max)}
        self.model = SurrogateModel().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.history_X = [] # Hyperparams
        self.history_y = [] # Scores
        self.epsilon = 0.3  # Exploration rate
        
    def normalize(self, params_dict):
        # Convert dict to normalized tensor [0, 1]
        vec = []
        keys = ['exotic_rho_0', 'lambda_topology', 'domain_radius', 'omega_0']
        for k in keys:
            val = params_dict.get(k, 0)
            low, high = self.bounds[k]
            # Clip and normalize
            val = max(min(val, high), low)
            norm_val = (val - low) / (high - low + 1e-6)
            vec.append(norm_val)
        return torch.tensor(vec, dtype=torch.float32).to(device)

    def denormalize(self, vec):
        # Convert tensor back to dict
        params = {}
        keys = ['exotic_rho_0', 'lambda_topology', 'domain_radius', 'omega_0']
        for i, k in enumerate(keys):
            low, high = self.bounds[k]
            val = vec[i].item() * (high - low) + low
            params[k] = float(val)
        return params

    def train(self):
        if len(self.history_X) < 5: return
        
        X = torch.stack(self.history_X)
        y = torch.tensor(self.history_y, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Simple training loop
        for _ in range(50):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            self.optimizer.step()

    def propose_next_experiments(self, n_candidates=500, n_select=1):
        # 1. Random Sampling of Candidates in normalized space
        candidates = torch.rand(n_candidates, 4).to(device)
        
        # 2. Predict Scores using Surrogate
        self.model.eval()
        with torch.no_grad():
            scores = self.model(candidates).flatten()
        self.model.train()
            
        # 3. Acquisition Strategy: Epsilon-Greedy
        best_indices = torch.argsort(scores, descending=True)
        
        proposals = []
        for i in range(n_select):
            if np.random.rand() < self.epsilon:
                # Explore: Pick random candidate
                idx = np.random.randint(0, n_candidates)
                print(f"   [Policy] Exploring random candidate...")
            else:
                # Exploit: Pick best predicted
                idx = best_indices[i]
                print(f"   [Policy] Exploiting best candidate (Pred Score: {scores[idx]:.3f})...")
                
            params = self.denormalize(candidates[idx])
            proposals.append(params)
            
        return proposals

    def update_history(self, params, score):
        self.history_X.append(self.normalize(params))
        self.history_y.append(score)

# --- 5. Simulation Controller & Campaign Runner ---

class GravitySimulator:
    def __init__(self, config):
        self.config = config
        self.device = device
        
    def build_model(self):
        return MetricEstimator(self.config).to(self.device)

    def train_simulation(self, target_topology):
        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        n_samples = self.config.batch_size
        loss_history = []

        for epoch in range(1, self.config.epochs + 1):
            optimizer.zero_grad()

            # Sampling
            spatial = (torch.rand(n_samples, 3, device=self.device) - 0.5) * 2 * self.config.domain_radius
            if target_topology == "warp_bubble":
                spatial[:, 0] *= 0.8; spatial[:, 1] *= 0.3; spatial[:, 2] *= 0.3
            else:
                mask = torch.rand(n_samples, device=self.device) < 0.7
                spatial[mask] *= 0.25

            t = torch.zeros(n_samples, 1, device=self.device)
            coords_4d = torch.cat([t, spatial], dim=1)
            coords_4d.requires_grad_(True)
            coords_spatial = coords_4d[:, 1:4]

            # Forward
            lapse, shift, spatial_metric = model(coords_4d)
            metric_4d = build_4d_metric(lapse, shift, spatial_metric)

            # Matter Source (Uses config rho0)
            if target_topology == "wormhole":
                T_fn = lambda c: exotic_fluid_T(c, rho0=self.config.exotic_rho_0)
            elif target_topology == "warp_bubble":
                T_fn = lambda c: exotic_fluid_T(c, rho0=self.config.exotic_rho_0)
            else:
                T_fn = None

            # Physics Loss
            loss_physics, _ = einstein_residual_loss(metric_4d, coords_4d, T_mu_nu_fn=T_fn)
            
            # Constraints
            g00 = metric_4d[:, 0, 0]
            loss_signature = torch.mean(F.relu(g00 + 1e-3)**2)
            det_g = torch.det(metric_4d)
            loss_det = torch.mean(F.relu(1e-4 - torch.abs(det_g))**2)
            loss_static = time_dependence_penalty(metric_4d, coords_4d)
            theta = DifferentialGeometry.expansion_scalar(shift, coords_spatial)
            loss_momentum = momentum_constraint(theta, coords_spatial)
            loss_boundary = boundary_loss(spatial_metric, coords_spatial)

            # Topology Shaping (Uses config lambda_topology)
            R_spatial = DifferentialGeometry.ricci_scalar(spatial_metric, coords_spatial)
            loss_topology = 0.0
            
            # Weighted by lambda_topology from hyperparams
            if target_topology == "anomaly_hunter":
                metric_identity_dist = torch.mean((spatial_metric - torch.eye(3, device=self.device))**2)
                loss_complexity = torch.mean(R_spatial**2) * 0.1
                loss_topology = loss_complexity - metric_identity_dist * self.config.lambda_complexity
            elif target_topology == "wormhole":
                loss_topology += torch.mean(F.relu(0.1 - lapse)**2) * 50.0
                r = torch.sqrt(torch.sum(coords_spatial**2, dim=1))
                if (r < 1.0).sum() > 0:
                    loss_topology += torch.mean((torch.abs(R_spatial[r < 1.0]) - 20.0)**2) * 0.1
            elif target_topology == "warp_bubble":
                r_s = torch.sqrt(torch.sum(coords_spatial**2, dim=1))
                target_shift_x = torch.exp(-r_s**2)
                loss_topology += F.mse_loss(shift[:, 0], target_shift_x) * 10.0
                loss_topology += torch.mean(shift[:, 1]**2 + shift[:, 2]**2)
                if (coords_spatial[:, 0] > 0.1).sum() > 0:
                    loss_topology += torch.mean(F.relu(theta[coords_spatial[:, 0] > 0.1] + 0.1)**2) * 5.0
                if (coords_spatial[:, 0] < -0.1).sum() > 0:
                    loss_topology += torch.mean(F.relu(0.1 - theta[coords_spatial[:, 0] < -0.1])**2) * 5.0

            # Total Loss
            total_loss = (
                self.config.lambda_hamiltonian * loss_physics +
                self.config.lambda_boundary    * loss_boundary +
                self.config.lambda_topology    * loss_topology + # Controlled by policy
                self.config.lambda_momentum    * loss_momentum +
                self.config.lambda_static      * loss_static +
                self.config.lambda_signature   * loss_signature +
                self.config.lambda_determinant * loss_det
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            loss_history.append(total_loss.item())
            
            if self.config.verbose and epoch % 200 == 0:
                print(f"   Ep {epoch} | Loss: {total_loss.item():.4f}")

        # Final Evaluation
        lapse, shift, spatial_metric = model(coords_4d)
        metric_4d = build_4d_metric(lapse, shift, spatial_metric)
        G_final = GRGeometry.einstein_tensor(metric_4d, coords_4d)
        rho_implied = G_final[:, 0, 0] / (8 * np.pi)

        return (
            model, 
            coords_spatial.detach().cpu().numpy(), 
            lapse.detach().cpu().numpy(), 
            shift.detach().cpu().numpy(), 
            rho_implied.detach().cpu().numpy(),
            loss_history
        )

class ExperimentRunner:
    def __init__(self):
        self.feature_db = [] # List of existing feature vectors

    @staticmethod
    def generate_iso_grid(radius, resolution=10):
        axis = np.linspace(-radius, radius, resolution)
        X, Y, Z = np.meshgrid(axis, axis, axis)
        coords_np = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        t = torch.zeros(coords_np.shape[0], 1)
        spatial = torch.tensor(coords_np, dtype=torch.float32)
        coords_4d = torch.cat([t, spatial], dim=1).to(device)
        coords_4d.requires_grad_(True)
        return coords_4d, coords_np
        
    def calculate_novelty(self, feats):
        if not self.feature_db:
            return 1.0 
        existing = np.stack(self.feature_db, axis=0)
        dists = np.linalg.norm(existing - feats[None, :], axis=1)
        return float(dists.min())

    def classify_topology(self, lapse, shift, rho):
        min_lapse = float(lapse.min())
        max_shift = float(np.linalg.norm(shift, axis=1).max())
        min_rho = float(rho.min())
        tags = []
        if min_lapse < 0.15: tags.append("Horizon-Candidate")
        elif min_lapse > 0.9: tags.append("Minkowski-Like")
        if min_rho < -0.01: tags.append("Exotic-Matter")
        if max_shift > 0.5:
            tags.append("High-Shift")
            if "Exotic-Matter" in tags: tags.append("Warp-Drive-Candidate")
        if not tags: tags.append("Unknown-Topology")
        return ", ".join(tags)

    def run_active_discovery(self, base_config, mode="warp_bubble", n_rounds=6, batch_size=1):
        """
        Runs an Active Learning loop to find interesting physics.
        """
        results = []
        self.feature_db = []
        
        # Define Search Space
        policy = ActiveSearchPolicy(bounds={
            'exotic_rho_0': (-0.5, -0.01),   # Try different fuel densities
            'lambda_topology': (1.0, 50.0),  # Try different forcing strengths
            'domain_radius': (2.0, 6.0),     # Try different bubble sizes
            'omega_0': (10.0, 60.0)          # Try different frequencies
        })
        
        print(f"\n=== STARTING ACTIVE DISCOVERY: {mode.upper()} ({n_rounds} rounds) ===")
        
        eval_coords_4d, eval_coords_np = self.generate_iso_grid(base_config.domain_radius, resolution=12)
        
        for r in range(n_rounds):
            print(f"\n--- ROUND {r+1}/{n_rounds} ---")
            
            # 1. Propose Experiments
            if r == 0:
                # Initial random seeds
                print("   [Init] Running random seeds...")
                specs = [{'hyperparams': {}} for _ in range(batch_size)] 
                # Randomize slightly for init
                for s in specs:
                    s['hyperparams'] = {
                        'exotic_rho_0': np.random.uniform(-0.3, -0.05),
                        'lambda_topology': np.random.uniform(5.0, 20.0)
                    }
            else:
                # Active Selection
                print("   [Policy] Proposing new parameters...")
                proposals = policy.propose_next_experiments(n_candidates=500, n_select=batch_size)
                specs = [{'hyperparams': p} for p in proposals]

            # 2. Run Simulations
            for i, spec in enumerate(specs):
                # Update Config
                run_config = copy.deepcopy(base_config)
                run_config.update(**spec['hyperparams'])
                
                # Run
                seed = int(time.time()) + i
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                print(f"   Running Solve (rho0={run_config.exotic_rho_0:.3f}, lambda={run_config.lambda_topology:.1f})...")
                sim = GravitySimulator(run_config)
                model, _, _, _, _, history = sim.train_simulation(mode)
                
                # Evaluate
                with torch.enable_grad(): 
                    lapse_eval, shift_eval, spat_eval = model(eval_coords_4d)
                    metric_4d_eval = build_4d_metric(lapse_eval, shift_eval, spat_eval)
                    if mode in ["wormhole", "warp_bubble"]:
                        T_fn = lambda c: exotic_fluid_T(c, rho0=run_config.exotic_rho_0)
                    else: T_fn = None
                    _, residual = einstein_residual_loss(metric_4d_eval, eval_coords_4d, T_fn)
                    G_final = GRGeometry.einstein_tensor(metric_4d_eval, eval_coords_4d)
                    rho_implied = G_final[:, 0, 0] / (8 * np.pi)
                
                G_res_norm = torch.mean(residual**2).item()
                
                # Features & Novelty
                lapse_np = lapse_eval.detach().cpu().numpy().flatten()
                shift_np = shift_eval.detach().cpu().numpy()
                rho_np = rho_implied.detach().cpu().numpy().flatten()
                
                feats = extract_features(eval_coords_np, lapse_np, rho_np, G_res_norm)
                novelty_score = self.calculate_novelty(feats)
                self.feature_db.append(feats)
                
                # Calculate Interest Score for Policy Training
                # High Interest = Valid Physics (Low Res) AND Novel
                # Score = Novelty / (1 + Residual_Penalty)
                interest_score = novelty_score / (1.0 + np.log1p(G_res_norm * 10.0))
                
                # Update Policy
                policy.update_history(spec['hyperparams'], interest_score)
                
                classification = self.classify_topology(lapse_np, shift_np, rho_np)
                
                # Log Result
                entry = {
                    "id": f"active_{r}_{i}",
                    "mode": mode,
                    "hyperparams": spec['hyperparams'],
                    "G_res_norm": G_res_norm,
                    "novelty": novelty_score,
                    "interest": interest_score,
                    "classification": classification,
                    "coords": eval_coords_np,
                    "lapse_field": lapse_np,
                    "rho_field": rho_np,
                }
                results.append(entry)
                print(f"   -> Result: Res={G_res_norm:.4f}, Nov={novelty_score:.3f}, Int={interest_score:.3f} | {classification}")

            # 3. Train Policy
            print("   [Policy] Updating surrogate model...")
            policy.train()
            
        return results

def rank_solutions(dataset):
    df = pd.DataFrame(dataset)
    df = df.sort_values('interest', ascending=False)
    print("\n=== TOP DISCOVERED SOLUTIONS ===")
    cols = ['id', 'classification', 'G_res_norm', 'novelty', 'interest']
    print(df[cols].to_string(index=False))
    return df

def visualize_entry(entry):
    coords = entry['coords']
    lapse = entry['lapse_field']
    rho = entry['rho_field']
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    p1 = ax1.scatter(coords[:,0], coords[:,1], coords[:,2], c=lapse, cmap='magma', s=10)
    ax1.set_title(f"Lapse ({entry['mode']})")
    fig.colorbar(p1, ax=ax1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    p2 = ax2.scatter(coords[:,0], coords[:,1], coords[:,2], c=rho, cmap='viridis', s=10)
    ax2.set_title("Implied Rho")
    fig.colorbar(p2, ax=ax2)
    
    params_str = ", ".join([f"{k}:{v:.2f}" for k,v in entry['hyperparams'].items()])
    plt.suptitle(f"Run: {entry['id']} | {params_str}")
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    config = HyperParameters()
    config.epochs = 400 
    
    runner = ExperimentRunner()
    
    # Run Active Discovery for Warp Bubbles
    dataset = runner.run_active_discovery(config, mode="warp_bubble", n_rounds=5, batch_size=1)
    
    ranked_df = rank_solutions(dataset)
    
    # Visualize Best
    if not ranked_df.empty:
        best_id = ranked_df.iloc[0]['id']
        best_entry = next(item for item in dataset if item["id"] == best_id)
        visualize_entry(best_entry)

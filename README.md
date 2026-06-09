# hodge-belief-rs

**Hodge decomposition of belief systems** — applying Hodge theory from differential geometry to analyze how beliefs flow, conflict, and stabilize in multi-agent networks.

[![Crates.io](https://img.shields.io/crates/v/hodge-belief-rs.svg)](https://crates.io/crates/hodge-belief-rs)
[![docs.rs](https://docs.rs/hodge-belief-rs/badge.svg)](https://docs.rs/hodge-belief-rs)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          lib.rs                              │
│  Re-exports: BeliefNetwork, BeliefNode, BeliefEdge,          │
│              HodgeDecomposition, StabilityReport,            │
│              GradientPotential, CurlAnalysis, EchoChamber,   │
│              LyapunovReport                                    │
└──────────────────────────────────────────────────────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ belief_  │  │ simplicial│  │  linalg  │  │ stability│
│ network  │  │          │  │          │  │          │
│ (Graph)  │  │(Complex) │  │(LA ops)  │  │(Analysis)│
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ hodge_   │  │ gradient_│  │ harmonic_│  │  curl_   │
│decomposition│  │  flow   │  │  flow   │  │  flow   │
│(ω=d₀f+h+δβ)│  │(Potential)│  │(Echo Chambers)│  │(Divergence)│
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

The crate is fully self-contained — all linear algebra is implemented from scratch (Gaussian elimination with partial pivoting). No BLAS, LAPACK, or nalgebra dependencies. Every public type derives `Serialize`/`Deserialize` for persistence and IPC.

### Key Design Decisions

- **Discrete Hodge theory on graphs** — Models belief networks as simplicial complexes (0-simplices = agents, 1-simplices = influence edges, 2-simplices = triangles / group interactions).
- **Orthogonal decomposition** — Any belief flow `ω` decomposes uniquely into `ω = ∇f + h + δβ` where the three components are mutually orthogonal in the L₂ inner product.
- **Echo chamber = harmonic flow** — The harmonic component lives in `ker(L₁)`, representing beliefs that circulate in cycles with no source or sink. The dimension of this space is the first Betti number `β₁`.
- **Stability scoring** — A network is stable when gradient (potential-driven) energy dominates. Harmonic and curl dominance signal echo chambers and active conversions.

---

## Quick Start

```rust
use hodge_belief_rs::*;
use hodge_belief_rs::hodge_decomposition::decompose;
use hodge_belief_rs::harmonic_flow::{detect_echo_chambers, harmonic_dimension};
use hodge_belief_rs::curl_flow::{compute_curl, classify_nodes, net_creation_rate};
use hodge_belief_rs::gradient_flow::{compute_gradient, is_curl_free, flow_energy};
use hodge_belief_rs::stability::analyze_stability;

// 1. Build a belief network
let mut net = BeliefNetwork::new();
let n0 = net.add_node(BeliefNode { id: "proposer".into(), beliefs: vec![0.8, 0.2], conviction: 0.9 });
let n1 = net.add_node(BeliefNode { id: "skeptic".into(), beliefs: vec![0.3, 0.7], conviction: 0.6 });
let n2 = net.add_node(BeliefNode { id: "moderator".into(), beliefs: vec![0.5, 0.5], conviction: 0.4 });
net.add_edge(n0, n1, 0.5, -0.2); // strong influence, disagreement
net.add_edge(n1, n2, 0.3, 0.1);
net.add_edge(n2, n0, 0.4, 0.4);

// 2. Compute belief flow
let flow = net.belief_flow();
println!("Flow energy: {:.4}", flow_energy(&flow));

// 3. Hodge decomposition: F = gradient + harmonic + curl
let decomp = decompose(&net, &flow);
let (g_frac, h_frac, c_frac) = decomp.energy_fractions();
println!("Gradient: {:.1}%  Harmonic: {:.1}%  Curl: {:.1}%",
         g_frac * 100.0, h_frac * 100.0, c_frac * 100.0);

// 4. Detect echo chambers (harmonic component)
let chambers = detect_echo_chambers(&net, &decomp.harmonic);
for ec in &chambers {
    println!("Echo chamber: nodes {:?}, strength {:.3}", ec.nodes, ec.strength);
}

// 5. Classify nodes by divergence (curl component)
let curl = compute_curl(&net, &flow);
let types = classify_nodes(&curl.divergence, 0.1);
println!("Net belief creation rate: {:.3}", net_creation_rate(&curl.divergence));

// 6. Stability analysis
let report = analyze_stability(&net, &flow, &decomp);
println!("Stability score: {:.2}", report.stability_score);
println!("Stable nodes: {:?}", report.stable_nodes);
println!("Unstable nodes: {:?}", report.unstable_nodes);
```

---

## API Reference

### Core Types

| Type | Fields | Description |
|------|--------|-------------|
| `BeliefNetwork` | `nodes: Vec<BeliefNode>, edges: Vec<BeliefEdge>` | The full belief graph |
| `BeliefNode` | `id: String, beliefs: Vec<f64>, conviction: f64` | Agent with belief state |
| `BeliefEdge` | `from: usize, to: usize, influence: f64, agreement: f64` | Directed influence |
| `HodgeDecomposition` | `gradient, harmonic, curl: Vec<f64>` | Three orthogonal components |
| `GradientPotential` | `potential, gradient_flow, energy` | Conservative flow from scalar potential |
| `CurlAnalysis` | `divergence, curl_on_triangles, curl_flow, energy` | Divergent flow analysis |
| `EchoChamber` | `nodes, edges, strength` | Detected harmonic cycle cluster |
| `StabilityReport` | `stable_nodes, unstable_nodes, echo_chambers, stability_score, node_stability` | Network stability assessment |
| `LyapunovReport` | `is_stable, energy_trajectory, convergence_rate` | Convergence analysis over time |

### Graph Constructors

| Function | Description |
|----------|-------------|
| `BeliefNetwork::new()` | Empty network |
| `BeliefNetwork::complete_graph(n)` | Fully connected graph (β₁ = 0) |
| `BeliefNetwork::ring_graph(n)` | Single cycle (β₁ = 1) |
| `BeliefNetwork::star_graph(n)` | Central authority (β₁ = 0) |

### Hodge Decomposition

| Function | Module | Description |
|----------|--------|-------------|
| `decompose(network, flow)` | `hodge_decomposition` | Full Hodge decomposition |
| `divergence(network, flow)` | `hodge_decomposition` | Node divergence: `B₁ · flow` |
| `curl_on_triangles(network, flow)` | `hodge_decomposition` | Curl on 2-simplices: `B₂ᵀ · flow` |
| `compute_gradient(network, flow)` | `gradient_flow` | Conservative component |
| `is_curl_free(network, flow)` | `gradient_flow` | Verify `B₂ᵀ · flow ≈ 0` |
| `is_divergence_free(network, flow)` | `gradient_flow` | Verify `B₁ · flow ≈ 0` |
| `gradient_flow_step(network, potential, dt)` | `gradient_flow` | Simulate belief diffusion |
| `compute_curl(network, flow)` | `curl_flow` | Divergent component |
| `classify_nodes(divergence, threshold)` | `curl_flow` | Converter / Apostate / Neutral |
| `net_creation_rate(divergence)` | `curl_flow` | Sum of divergences |
| `detect_echo_chambers(network, harmonic_flow)` | `harmonic_flow` | Find harmonic cycle clusters |
| `harmonic_dimension(network)` | `harmonic_flow` | First Betti number `β₁` |
| `is_harmonic(network, flow)` | `harmonic_flow` | Both divergence-free and curl-free |
| `analyze_stability(network, flow, decomp)` | `stability` | Per-node stability scoring |
| `lyapunov_stability(flow_history)` | `stability` | Check monotonic energy decrease |

### Linear Algebra (Internal)

| Function | Module | Description |
|----------|--------|-------------|
| `solve(a, b)` | `linalg` | Gaussian elimination with partial pivoting |
| `mat_mul(a, b)` | `linalg` | Matrix multiplication |
| `mat_vec_mul(a, x)` | `linalg` | Matrix-vector product |
| `transpose(a)` | `linalg` | Matrix transpose |
| `least_squares_solve(a, b)` | `linalg` | Pseudoinverse solve |
| `incidence_matrix(n, edges)` | `simplicial` | Build `B₁` (nodes × edges) |
| `boundary2_matrix(m, edges, triangles)` | `simplicial` | Build `B₂` (edges × triangles) |
| `hodge_laplacian_1(b1, b2)` | `simplicial` | `L₁ = B₁ᵀB₁ + B₂B₂ᵀ` |
| `betti_number(laplacian)` | `simplicial` | Kernel dimension via rank deficiency |

---

## Integration Notes

### With Agent Orchestration Dashboards

Serialize the full network and decomposition for frontend rendering:

```rust
let net = BeliefNetwork::complete_graph(10);
let flow = net.belief_flow();
let decomp = decompose(&net, &flow);
let report = analyze_stability(&net, &flow, &decomp);

let payload = serde_json::json!({
    "network": net,
    "decomposition": decomp,
    "stability": report,
});
// POST to dashboard API
```

### With Consensus Protocols

Use gradient energy fraction as a consensus metric:

```rust
let decomp = decompose(&network, &flow);
let (g, h, c) = decomp.energy_fractions();

if g > 0.8 {
    println!("Consensus reached — beliefs are potential-driven");
} else if h > 0.5 {
    println!("Warning: echo chambers detected — consensus impossible without bridge agents");
} else if c > 0.3 {
    println!("Active conversion/apostasy — network is in flux");
}
```

### With Recommendation Engines

Detect agents that are sources or sinks of belief:

```rust
let curl = compute_curl(&network, &flow);
let types = classify_nodes(&curl.divergence, 0.1);

for (i, t) in types.iter().enumerate() {
    match t {
        NodeType::Converter => println!("Agent {} is a belief source", i),
        NodeType::Apostate => println!("Agent {} is a belief sink", i),
        NodeType::Neutral => {},
    }
}
```

### With Temporal Analysis

Track belief dynamics over time using Lyapunov stability:

```rust
let mut history: Vec<Vec<f64>> = Vec::new();
for snapshot in time_series {
    let flow = snapshot.belief_flow();
    history.push(flow);
}

let report = lyapunov_stability(&history);
if report.is_stable {
    println!("Beliefs converging at rate {:.3}", report.convergence_rate);
} else {
    println!("Beliefs diverging — network instability");
}
```

### With Topology-Aware Networking

Use Betti numbers to quantify network structure:

```rust
let dim = harmonic_dimension(&network);
println!("Network has {} independent cycles (echo chambers)", dim);

// A tree has dim=0; a ring has dim=1; two rings have dim=2
match dim {
    0 => println!("Tree topology — no echo chambers possible"),
    1 => println!("Single cycle — one potential echo chamber"),
    n if n > 1 => println!("{} cycles — multiple disconnected echo chambers", n),
    _ => {},
}
```

---

## Mathematical Background

### Discrete Hodge Decomposition

For a simplicial complex with boundary operators `B₁` (nodes × edges) and `B₂` (edges × triangles), any edge flow `ω` decomposes uniquely as:

```
ω = d₀f + h + d₁ᵀβ
```

where:
- `d₀f = B₁ᵀ · f` is the **gradient** (exact) component — beliefs flowing along a scalar potential
- `h ∈ ker(L₁)` is the **harmonic** component — beliefs circulating in closed loops
- `d₁ᵀβ = B₂ · β` is the **curl** (coexact) component — beliefs created or destroyed at nodes

The Hodge Laplacian on edges is `L₁ = B₁ᵀB₁ + B₂B₂ᵀ`. The three subspaces are mutually orthogonal.

### Betti Numbers

- `β₀ = dim(ker L₀)` = number of connected components
- `β₁ = dim(ker L₁)` = number of independent cycles (echo chambers)
- `β₂ = dim(ker L₂)` = number of enclosed cavities

### Stability Criterion

A belief network is **stable** when gradient energy dominates. The stability score is the gradient fraction `g_frac`. Per-node stability decreases when connected edges carry significant harmonic or curl energy. Nodes inside echo chambers are capped at stability 0.3.

---

## Testing

```bash
cargo test   # 35+ unit tests covering decomposition, topology, stability, LA
cargo test test_decomposition_sums_to_original_flow
cargo test test_betti_number_ring_is_one
cargo test test_energy_fractions_sum_to_one
cargo test test_stability_identifies_echo_chambers
cargo test test_solve_2x2
cargo test test_gradient_flow_step_beliefs_change
```

---

## License

MIT

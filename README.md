# hodge-belief-rs

**Hodge decomposition of belief systems** — applying Hodge theory from differential geometry to analyze how beliefs flow, conflict, and stabilize in multi-agent networks.

[![Crates.io](https://img.shields.io/crates/v/hodge-belief-rs.svg)](https://crates.io/crates/hodge-belief-rs)

## The Big Idea

Beliefs spread through networks of influencing agents. Some beliefs flow predictably from authorities to followers. Others circulate endlessly in echo chambers. And some appear or vanish at specific nodes — conversions and apostasies.

**Hodge theory** gives us a mathematical lens to decompose any belief flow into three orthogonal components:

| Component | Analogy | Meaning |
|-----------|---------|---------|
| **Gradient** (∇f) | Water flowing downhill | Beliefs explained by a scalar potential — conservative, predictable, stable |
| **Harmonic** (h) | Whirlpools with no drain | Beliefs circulating in closed loops — echo chambers, self-reinforcing cycles |
| **Curl** (δβ) | Springs and sinks | Beliefs created or destroyed at nodes — conversions, apostasies, radicalization |

Given a belief flow **F** on a network:

> **F = ∇f + h + δβ**

This is the **Hodge decomposition**: every flow is uniquely the sum of a gradient component, a harmonic component, and a coexact (curl) component.

## How It Works

### Simplicial Complexes from Social Networks

We model a belief network as a **simplicial complex**:

- **0-simplices** → agents (nodes)
- **1-simplices** → influence relationships (edges)
- **2-simplices** → group interactions (triangles)

From this we build:
- **Boundary operators** B₁ (nodes × edges) and B₂ (edges × triangles)
- **Hodge Laplacian** L₁ = B₁ᵀB₁ + B₂B₂ᵀ on edges

### The Decomposition

For a flow ω on edges:

1. **Gradient**: Solve L₀f = B₁ω for potential f, then ∇f = B₁ᵀf
2. **Curl**: Solve B₂ᵀB₂β = B₂ᵀ(ω − ∇f) for β, then δβ = B₂β
3. **Harmonic**: h = ω − ∇f − δβ (the residual)

### Echo Chambers = Harmonic Flow

The **harmonic component** lives in ker(L₁) — flows that are both divergence-free (no source/sink) and curl-free. These are beliefs that circulate in cycles without any external input or drain. The dimension of this space is the **first Betti number β₁**, which counts independent cycles in the network.

A ring graph has β₁ = 1 (one echo chamber). A complete graph has β₁ = 0 (fully connected, no echo chambers). Two disconnected rings have β₁ = 2.

### Stability Analysis

Beliefs are **stable** when dominated by the gradient component (predictable potential-driven flow). They're **unstable** when harmonic or curl components dominate — indicating echo chambers or active conversions.

We use **Lyapunov stability** to track whether belief energy converges over time.

## Quick Start

```rust
use hodge_belief_rs::*;

// Build a belief network
let mut net = BeliefNetwork::new();
net.add_node(BeliefNode { id: "alice".into(), beliefs: vec![0.9, 0.1], conviction: 0.8 });
net.add_node(BeliefNode { id: "bob".into(),   beliefs: vec![0.3, 0.7], conviction: 0.6 });
net.add_node(BeliefNode { id: "carol".into(), beliefs: vec![0.5, 0.5], conviction: 0.4 });
net.add_edge(0, 1, 0.8, 0.5);  // alice → bob, strong influence, moderate agreement
net.add_edge(1, 2, 0.6, -0.3); // bob → carol, moderate influence, disagreement
net.add_edge(2, 0, 0.4, 0.7);  // carol → alice, weak influence, high agreement

// Compute belief flow
let flow = net.belief_flow();

// Decompose into gradient + harmonic + curl
let decomp = hodge_decomposition::decompose(&net, &flow);
let (grad_frac, harm_frac, curl_frac) = decomp.energy_fractions();
println!("Gradient: {:.1}%  Harmonic: {:.1}%  Curl: {:.1}%",
    grad_frac * 100, harm_frac * 100, curl_frac * 100);

// Detect echo chambers
let chambers = harmonic_flow::detect_echo_chambers(&net, &decomp.harmonic);
for ec in &chambers {
    println!("Echo chamber: nodes {:?}, strength {:.3}", ec.nodes, ec.strength);
}

// Analyze stability
let report = stability::analyze_stability(&net, &flow, &decomp);
println!("Stability score: {:.2}", report.stability_score);
println!("Stable nodes: {:?}", report.stable_nodes);
println!("Unstable nodes: {:?}", report.unstable_nodes);
```

## Modules

| Module | Description |
|--------|-------------|
| `belief_network` | Belief graphs: nodes with belief states, edges with influence/agreement |
| `simplicial` | Simplicial complexes, boundary/coboundary operators, Betti numbers |
| `hodge_decomposition` | Core Hodge decomposition: F = gradient + harmonic + curl |
| `gradient_flow` | Conservative (potential-driven) belief flow |
| `harmonic_flow` | Echo chamber detection via harmonic component analysis |
| `curl_flow` | Divergent belief flow: conversions, apostasy, belief creation/destruction |
| `stability` | Stability analysis and Lyapunov convergence |
| `linalg` | Linear algebra from scratch (Gaussian elimination, matrix operations) |

## Core Types

```rust
struct BeliefNetwork { nodes: Vec<BeliefNode>, edges: Vec<BeliefEdge> }
struct BeliefNode { id: String, beliefs: Vec<f64>, conviction: f64 }
struct BeliefEdge { from: usize, to: usize, influence: f64, agreement: f64 }
struct HodgeDecomposition { gradient: Vec<f64>, harmonic: Vec<f64>, curl: Vec<f64> }
struct StabilityReport { stable_nodes: Vec<usize>, unstable_nodes: Vec<usize>, echo_chambers: Vec<Vec<usize>>, stability_score: f64 }
struct EchoChamber { nodes: Vec<usize>, edges: Vec<usize>, strength: f64 }
```

## Examples

### Ring Network — Echo Chamber Detection

A ring network where everyone influences the next person in line:

```rust
let net = BeliefNetwork::ring_graph(5);
let circular_flow = vec![1.0; 5]; // uniform circular flow
let decomp = hodge_decomposition::decompose(&net, &circular_flow);
// Harmonic energy ≈ 100% — this is a pure echo chamber
```

### Complete Network — Stable Beliefs

A fully connected network where everyone talks to everyone:

```rust
let net = BeliefNetwork::complete_graph(4);
// β₁ = 0 — no echo chambers possible
// Any flow decomposes into gradient + curl only
```

### Star Network — Authority-Driven

A central authority with many followers:

```rust
let net = BeliefNetwork::star_graph(5);
// Belief flow is almost entirely gradient
// Followers receive from the center, no cycles possible
```

## Mathematical Background

### Hodge Theory

In differential geometry, the Hodge decomposition theorem states that any differential form on a compact Riemannian manifold can be decomposed into exact, coexact, and harmonic components. We apply the **discrete** (combinatorial) version to graphs.

### Combinatorial Hodge Decomposition

For a simplicial complex with boundary operators B₁ and B₂:

- **Exact (gradient)**: im(B₁ᵀ) — flows that are gradients of a scalar potential
- **Coexact (curl)**: im(B₂) — flows that are curls of a 2-form
- **Harmonic**: ker(L₁) where L₁ = B₁ᵀB₁ + B₂B₂ᵀ

These three spaces are orthogonal in the L₂ inner product, giving a unique decomposition.

### Betti Numbers

- **β₀** = dim(ker L₀) = number of connected components
- **β₁** = dim(ker L₁) = number of independent cycles (echo chambers)
- **β₂** = dim(ker L₂) = number of enclosed cavities

## Implementation Notes

- **Linear algebra from scratch**: Gaussian elimination with partial pivoting, no external LAPACK/BLAS dependency
- **Serde support**: All core types implement `Serialize`/`Deserialize`
- **Edition 2024**: Uses latest Rust edition features

## License

MIT

use serde::{Deserialize, Serialize};

use crate::belief_network::BeliefNetwork;
use crate::linalg::{mat_mul, mat_vec_mul, solve, transpose};
use crate::simplicial;

/// The scalar belief potential at each node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientPotential {
    /// Potential value at each node.
    pub potential: Vec<f64>,
    /// Gradient flow on edges.
    pub gradient_flow: Vec<f64>,
    /// Total conservative energy.
    pub energy: f64,
}

/// Solve by grounding the first node (setting its potential to 0).
fn solve_grounded(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n <= 1 {
        return Some(vec![0.0; n]);
    }
    let a_red: Vec<Vec<f64>> = a[1..].iter().map(|row| row[1..].to_vec()).collect();
    let b_red: Vec<f64> = b[1..].to_vec();
    let mut x = vec![0.0; n];
    if let Some(x_red) = solve(&a_red, &b_red) {
        for i in 0..x_red.len() {
            x[i + 1] = x_red[i];
        }
        Some(x)
    } else {
        None
    }
}

/// Compute the gradient (conservative) component of belief flow.
///
/// The gradient component satisfies d₀f = B₁ᵀ · f for some potential f on nodes.
/// This is "curl-free": beliefs that can be explained entirely by a scalar field.
pub fn compute_gradient(network: &BeliefNetwork, flow: &[f64]) -> GradientPotential {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let m = edges.len();

    if m == 0 || n == 0 {
        return GradientPotential {
            potential: vec![0.0; n],
            gradient_flow: flow.to_vec(),
            energy: 0.0,
        };
    }

    let b1 = simplicial::incidence_matrix(n, &edges);
    let b1t = transpose(&b1);
    let l0 = mat_mul(&b1, &b1t);
    let b1_flow = mat_vec_mul(&b1, flow);

    let potential = solve_grounded(&l0, &b1_flow).unwrap_or_else(|| vec![0.0; n]);
    let gradient_flow = mat_vec_mul(&b1t, &potential);
    let energy: f64 = gradient_flow.iter().map(|x| x * x).sum();

    GradientPotential {
        potential,
        gradient_flow,
        energy,
    }
}

/// Verify that a flow is curl-free: B₂ᵀ · flow ≈ 0.
pub fn is_curl_free(network: &BeliefNetwork, flow: &[f64]) -> bool {
    let triangles = network.detect_triangles();
    if triangles.is_empty() {
        return true;
    }
    let edges = network.to_edges();
    let m = edges.len();
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);
    let b2t = transpose(&b2);
    let curl = mat_vec_mul(&b2t, flow);
    curl.iter().all(|c| c.abs() < 1e-8)
}

/// Verify that a flow is divergence-free: B₁ · flow ≈ 0.
pub fn is_divergence_free(network: &BeliefNetwork, flow: &[f64]) -> bool {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let b1 = simplicial::incidence_matrix(n, &edges);
    let div = mat_vec_mul(&b1, flow);
    div.iter().all(|d| d.abs() < 1e-8)
}

/// Compute the energy (L2 norm squared) of a flow.
pub fn flow_energy(flow: &[f64]) -> f64 {
    flow.iter().map(|x| x * x).sum()
}

/// Simulate gradient descent on belief potential.
/// At each step, beliefs move toward lower potential neighbors.
pub fn gradient_flow_step(
    network: &mut BeliefNetwork,
    potential: &[f64],
    dt: f64,
) -> f64 {
    let _n = network.num_nodes();
    let mut total_change = 0.0;

    for e in &network.edges {
        let diff = potential[e.to] - potential[e.from];
        let change = dt * e.influence * e.agreement * diff;
        let dim = network.nodes[e.to].beliefs.len().max(1);
        for b in &mut network.nodes[e.to].beliefs {
            *b += change / dim as f64;
        }
        total_change += change.abs();
    }

    total_change
}

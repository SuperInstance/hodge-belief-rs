use serde::{Deserialize, Serialize};

use crate::belief_network::BeliefNetwork;
use crate::linalg::{mat_vec_mul, mat_mul, solve, transpose};
use crate::simplicial;

/// The three orthogonal components of the Hodge decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HodgeDecomposition {
    /// Gradient (exact) component: beliefs flowing along a potential.
    pub gradient: Vec<f64>,
    /// Harmonic component: circular beliefs (echo chambers).
    pub harmonic: Vec<f64>,
    /// Curl (coexact) component: divergent beliefs (conversion/apostasy).
    pub curl: Vec<f64>,
}

impl HodgeDecomposition {
    /// The original flow is the sum of the three components.
    pub fn total_flow(&self) -> Vec<f64> {
        self.gradient
            .iter()
            .zip(self.harmonic.iter().zip(self.curl.iter()))
            .map(|(g, (h, c))| g + h + c)
            .collect()
    }

    /// Relative energy of each component.
    pub fn energy_fractions(&self) -> (f64, f64, f64) {
        let g_energy: f64 = self.gradient.iter().map(|x| x * x).sum();
        let h_energy: f64 = self.harmonic.iter().map(|x| x * x).sum();
        let c_energy: f64 = self.curl.iter().map(|x| x * x).sum();
        let total = g_energy + h_energy + c_energy;
        if total.abs() < 1e-15 {
            (0.0, 0.0, 0.0)
        } else {
            (g_energy / total, h_energy / total, c_energy / total)
        }
    }
}

/// Solve a possibly-singular system Ax=b by grounding the first variable.
/// Removes row/col 0 and solves the reduced system, then sets x[0]=0.
fn solve_grounded(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n <= 1 {
        return Some(vec![0.0; n]);
    }
    // Remove row 0 and col 0
    let a_red: Vec<Vec<f64>> = a[1..]
        .iter()
        .map(|row| row[1..].to_vec())
        .collect();
    let b_red: Vec<f64> = b[1..].to_vec();
    let mut x = vec![0.0; n];
    if let Some(x_red) = solve(&a_red, &b_red) {
        for i in 0..x_red.len() {
            x[i + 1] = x_red[i];
        }
        Some(x)
    } else {
        // Fallback: try with regularization
        let lambda = 1e-8;
        let a_reg: Vec<Vec<f64>> = a_red
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| if i == j { v + lambda } else { v })
                    .collect()
            })
            .collect();
        if let Some(x_red) = solve(&a_reg, &b_red) {
            for i in 0..x_red.len() {
                x[i + 1] = x_red[i];
            }
            Some(x)
        } else {
            None
        }
    }
}

/// Perform the Hodge decomposition of a belief flow on the network.
///
/// Given flow ω on edges:
///   ω = d₀f + h + d₁ᵀβ
/// where:
///   - d₀f = B₁ᵀ · f  is the gradient (exact) component
///   - h ∈ ker(L₁)      is the harmonic component
///   - d₁ᵀβ = B₂ · β   is the coexact (curl) component
///
/// We solve using grounded Laplacian to avoid singularity:
///   gradient: solve L₀ f = B₁ ω (grounded), then gradient = B₁ᵀ f
///   curl:     solve B₂ᵀ B₂ β = B₂ᵀ residual (grounded), then curl = B₂ β
///   harmonic: h = ω - gradient - curl
pub fn decompose(network: &BeliefNetwork, flow: &[f64]) -> HodgeDecomposition {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let m = edges.len();

    if m == 0 || flow.is_empty() {
        return HodgeDecomposition {
            gradient: flow.to_vec(),
            harmonic: vec![0.0; flow.len()],
            curl: vec![0.0; flow.len()],
        };
    }

    let triangles = network.detect_triangles();
    let b1 = simplicial::incidence_matrix(n, &edges);
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);

    let b1t = transpose(&b1);

    // Gradient component: solve L₀ f = B₁ ω using grounded Laplacian
    let l0 = mat_mul(&b1, &b1t);
    let b1_flow = mat_vec_mul(&b1, flow);
    let gradient = if let Some(f) = solve_grounded(&l0, &b1_flow) {
        mat_vec_mul(&b1t, &f)
    } else {
        vec![0.0; m]
    };

    // Curl component: solve B₂ᵀ B₂ β = B₂ᵀ residual using grounded solve
    let residual_for_curl: Vec<f64> = flow
        .iter()
        .zip(gradient.iter())
        .map(|(f, g)| f - g)
        .collect();

    let b2t = transpose(&b2);
    let b2t_b2 = mat_mul(&b2t, &b2);
    let b2t_residual = mat_vec_mul(&b2t, &residual_for_curl);
    let curl = if !triangles.is_empty() {
        if let Some(beta) = solve_grounded(&b2t_b2, &b2t_residual) {
            mat_vec_mul(&b2, &beta)
        } else {
            vec![0.0; m]
        }
    } else {
        vec![0.0; m]
    };

    // Harmonic = flow - gradient - curl
    let harmonic: Vec<f64> = flow
        .iter()
        .zip(gradient.iter().zip(curl.iter()))
        .map(|(f, (g, c))| f - g - c)
        .collect();

    HodgeDecomposition {
        gradient,
        harmonic,
        curl,
    }
}

/// Compute the divergence at each node: div = B₁ · flow.
/// Positive divergence = belief source (conversion).
/// Negative divergence = belief sink (apostasy).
pub fn divergence(network: &BeliefNetwork, flow: &[f64]) -> Vec<f64> {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let b1 = simplicial::incidence_matrix(n, &edges);
    mat_vec_mul(&b1, flow)
}

/// Compute the curl on each triangle: curl = B₂ᵀ · flow.
pub fn curl_on_triangles(network: &BeliefNetwork, flow: &[f64]) -> Vec<f64> {
    let triangles = network.detect_triangles();
    let edges = network.to_edges();
    let m = edges.len();
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);
    let b2t = transpose(&b2);
    mat_vec_mul(&b2t, flow)
}

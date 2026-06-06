use serde::{Deserialize, Serialize};

use crate::belief_network::BeliefNetwork;
use crate::linalg::{least_squares_solve, mat_mul, mat_vec_mul, transpose};
use crate::simplicial;

/// Curl analysis result for a belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurlAnalysis {
    /// Divergence at each node.
    /// Positive = belief source (conversion).
    /// Negative = belief sink (apostasy).
    pub divergence: Vec<f64>,
    /// Curl on each triangle.
    pub curl_on_triangles: Vec<f64>,
    /// Curl (coexact) flow on edges.
    pub curl_flow: Vec<f64>,
    /// Total divergent energy.
    pub energy: f64,
}

/// Compute the curl (coexact) component of belief flow.
///
/// The curl component is d₁ᵀβ = B₂ · β for some β on triangles.
/// It represents beliefs that appear or disappear at nodes —
/// conversion (source) and apostasy (sink).
pub fn compute_curl(network: &BeliefNetwork, flow: &[f64]) -> CurlAnalysis {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let m = edges.len();
    let triangles = network.detect_triangles();

    if m == 0 || n == 0 {
        return CurlAnalysis {
            divergence: vec![0.0; n],
            curl_on_triangles: vec![],
            curl_flow: vec![0.0; m],
            energy: 0.0,
        };
    }

    let b1 = simplicial::incidence_matrix(n, &edges);
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);
    let b2t = transpose(&b2);

    // Divergence: B₁ · flow
    let divergence = mat_vec_mul(&b1, flow);

    // Curl on triangles: B₂ᵀ · flow
    let curl_on_tri = mat_vec_mul(&b2t, flow);

    // Curl flow: solve B₂ᵀB₂ β = B₂ᵀ flow, then curl_flow = B₂ β
    let curl_flow = if !triangles.is_empty() {
        let b2t_b2 = mat_mul(&b2t, &b2);
        let b2t_flow = mat_vec_mul(&b2t, flow);
        if let Some(beta) = least_squares_solve(&b2t_b2, &b2t_flow) {
            mat_vec_mul(&b2, &beta)
        } else {
            vec![0.0; m]
        }
    } else {
        vec![0.0; m]
    };

    let energy: f64 = curl_flow.iter().map(|x| x * x).sum();

    CurlAnalysis {
        divergence,
        curl_on_triangles: curl_on_tri,
        curl_flow,
        energy,
    }
}

/// Classify nodes as converters (source), apostates (sink), or neutral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    /// Belief source — this agent converts others.
    Converter,
    /// Belief sink — this agent abandons beliefs.
    Apostate,
    /// Neutral — neither source nor sink.
    Neutral,
}

/// Classify each node based on divergence.
pub fn classify_nodes(divergence: &[f64], threshold: f64) -> Vec<NodeType> {
    divergence
        .iter()
        .map(|&d| {
            if d > threshold {
                NodeType::Converter
            } else if d < -threshold {
                NodeType::Apostate
            } else {
                NodeType::Neutral
            }
        })
        .collect()
}

/// Compute the net belief creation rate.
pub fn net_creation_rate(divergence: &[f64]) -> f64 {
    divergence.iter().sum()
}

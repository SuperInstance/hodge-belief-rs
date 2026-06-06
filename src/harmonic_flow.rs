use serde::{Deserialize, Serialize};

use crate::belief_network::BeliefNetwork;
use crate::linalg::{mat_vec_mul, transpose};
use crate::simplicial;

/// Information about detected echo chambers (harmonic components).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoChamber {
    /// Node indices participating in this echo chamber.
    pub nodes: Vec<usize>,
    /// Edge indices carrying harmonic flow.
    pub edges: Vec<usize>,
    /// Strength of the echo chamber (L2 norm of harmonic flow).
    pub strength: f64,
}

/// Compute the harmonic component and detect echo chambers.
///
/// The harmonic component lives in ker(L₁) — it's orthogonal to both
/// gradient and curl components. These represent beliefs that circulate
/// in cycles with no source or sink — the mathematical signature of
/// echo chambers.
pub fn detect_echo_chambers(
    network: &BeliefNetwork,
    harmonic_flow: &[f64],
) -> Vec<EchoChamber> {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let m = edges.len();

    if m == 0 {
        return vec![];
    }

    let strength: f64 = harmonic_flow.iter().map(|x| x * x).sum();

    // If harmonic energy is negligible, no echo chambers
    if strength < 1e-10 {
        return vec![];
    }

    // Find edges with significant harmonic flow
    let threshold = 1e-8;
    let active_edges: Vec<usize> = harmonic_flow
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > threshold)
        .map(|(i, _)| i)
        .collect();

    // Find connected components among active edges
    let mut visited = vec![false; n];
    let mut chambers = Vec::new();

    for &start_edge in &active_edges {
        if visited[edges[start_edge].from] && visited[edges[start_edge].to] {
            continue;
        }

        // BFS/DFS to find connected component
        let mut chamber_nodes = Vec::new();
        let mut chamber_edges = Vec::new();
        let mut stack = vec![start_edge];

        while let Some(eidx) = stack.pop() {
            chamber_edges.push(eidx);
            let e = &edges[eidx];
            for &node in &[e.from, e.to] {
                if !visited[node] {
                    visited[node] = true;
                    chamber_nodes.push(node);
                    // Add adjacent active edges
                    for &ae in &active_edges {
                        let ae_edge = &edges[ae];
                        if (ae_edge.from == node || ae_edge.to == node)
                            && !chamber_edges.contains(&ae)
                        {
                            stack.push(ae);
                        }
                    }
                }
            }
        }

        if !chamber_nodes.is_empty() {
            let edge_strength: f64 = chamber_edges
                .iter()
                .map(|&i| harmonic_flow[i] * harmonic_flow[i])
                .sum();

            chambers.push(EchoChamber {
                nodes: {
                    let mut ns = chamber_nodes;
                    ns.sort();
                    ns.dedup();
                    ns
                },
                edges: {
                    let mut es = chamber_edges;
                    es.sort();
                    es.dedup();
                    es
                },
                strength: edge_strength,
            });
        }
    }

    chambers
}

/// Compute the dimension of the harmonic space (β₁, the first Betti number).
pub fn harmonic_dimension(network: &BeliefNetwork) -> usize {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let triangles = network.detect_triangles();
    let m = edges.len();

    if m == 0 {
        return 0;
    }

    let b1 = simplicial::incidence_matrix(n, &edges);
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);
    let l1 = simplicial::hodge_laplacian_1(&b1, &b2);
    simplicial::betti_number(&l1)
}

/// Verify that a flow is both divergence-free and curl-free (harmonic).
pub fn is_harmonic(network: &BeliefNetwork, flow: &[f64]) -> bool {
    let n = network.num_nodes();
    let edges = network.to_edges();
    let m = edges.len();

    // Check divergence-free: B₁ · flow = 0
    let b1 = simplicial::incidence_matrix(n, &edges);
    let div = mat_vec_mul(&b1, flow);
    let div_free = div.iter().all(|d| d.abs() < 1e-8);

    // Check curl-free: B₂ᵀ · flow = 0
    let triangles = network.detect_triangles();
    let b2 = simplicial::boundary2_matrix(m, &edges, &triangles);
    let b2t = transpose(&b2);
    let curl = mat_vec_mul(&b2t, flow);
    let curl_free = curl.iter().all(|c| c.abs() < 1e-8);

    div_free && curl_free
}

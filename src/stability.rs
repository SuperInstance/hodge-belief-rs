use serde::{Deserialize, Serialize};

use crate::belief_network::BeliefNetwork;
use crate::gradient_flow::flow_energy;
use crate::harmonic_flow::detect_echo_chambers;

/// Stability report for a belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    /// Nodes whose beliefs are stable (gradient-dominated).
    pub stable_nodes: Vec<usize>,
    /// Nodes whose beliefs are unstable (harmonic/curl-dominated).
    pub unstable_nodes: Vec<usize>,
    /// Detected echo chambers.
    pub echo_chambers: Vec<Vec<usize>>,
    /// Overall stability score ∈ [0, 1]. 1 = fully stable.
    pub stability_score: f64,
    /// Per-node stability scores.
    pub node_stability: Vec<f64>,
}

/// Analyze belief stability using Hodge decomposition.
///
/// A belief system is stable when it's dominated by the gradient component
/// (beliefs explained by a potential). Harmonic (echo chamber) and curl
/// (conversion) components indicate instability.
pub fn analyze_stability(
    network: &BeliefNetwork,
    _flow: &[f64],
    decomp: &crate::hodge_decomposition::HodgeDecomposition,
) -> StabilityReport {
    let n = network.num_nodes();
    let _m = network.num_edges();
    let edges = network.to_edges();

    let (g_frac, _h_frac, _c_frac) = decomp.energy_fractions();

    // Stability score: dominated by gradient → stable
    let stability_score = g_frac;

    // Per-node stability based on local flow characteristics
    let mut node_stability = vec![1.0; n];

    // For each edge, its harmonic and curl contribution affects connected nodes
    for (i, e) in edges.iter().enumerate() {
        let h_contrib = if i < decomp.harmonic.len() {
            decomp.harmonic[i].abs()
        } else {
            0.0
        };
        let c_contrib = if i < decomp.curl.len() {
            decomp.curl[i].abs()
        } else {
            0.0
        };
        let g_contrib = if i < decomp.gradient.len() {
            decomp.gradient[i].abs()
        } else {
            0.0
        };

        let total = h_contrib + c_contrib + g_contrib;
        if total > 1e-10 {
            let instability = (h_contrib + c_contrib) / total;
            // Reduce stability of connected nodes
            node_stability[e.from] = f64::min(node_stability[e.from], 1.0 - instability);
            node_stability[e.to] = f64::min(node_stability[e.to], 1.0 - instability);
        }
    }

    // Detect echo chambers from harmonic flow
    let echo_chambers_raw = detect_echo_chambers(network, &decomp.harmonic);
    let echo_chambers: Vec<Vec<usize>> = echo_chambers_raw
        .iter()
        .map(|ec| ec.nodes.clone())
        .collect();

    // Nodes in echo chambers are unstable
    for ec in &echo_chambers {
        for &node in ec {
            node_stability[node] = f64::min(node_stability[node], 0.3);
        }
    }

    // Classify stable vs unstable
    let threshold = 0.5;
    let mut stable_nodes = Vec::new();
    let mut unstable_nodes = Vec::new();

    for (i, &s) in node_stability.iter().enumerate() {
        if s >= threshold {
            stable_nodes.push(i);
        } else {
            unstable_nodes.push(i);
        }
    }

    StabilityReport {
        stable_nodes,
        unstable_nodes,
        echo_chambers,
        stability_score,
        node_stability,
    }
}

/// Lyapunov stability: check if belief dynamics are converging.
///
/// We treat the belief flow as a dynamical system and compute
/// a discrete Lyapunov function V = ||flow||². If V decreases
/// over time, the system is Lyapunov stable.
pub fn lyapunov_stability(flow_history: &[Vec<f64>]) -> LyapunovReport {
    if flow_history.len() < 2 {
        return LyapunovReport {
            is_stable: true,
            energy_trajectory: flow_history.iter().map(|f| flow_energy(f)).collect(),
            convergence_rate: 0.0,
        };
    }

    let energies: Vec<f64> = flow_history.iter().map(|f| flow_energy(f)).collect();

    // Check if energy is monotonically decreasing (allowing small numerical noise)
    let mut decreasing = true;
    let mut rate_sum = 0.0;
    for i in 1..energies.len() {
        if energies[i] > energies[i - 1] + 1e-10 {
            decreasing = false;
        }
        if energies[i - 1] > 1e-15 {
            rate_sum += (energies[i - 1] - energies[i]) / energies[i - 1];
        }
    }

    let convergence_rate = if energies.len() > 1 {
        rate_sum / (energies.len() - 1) as f64
    } else {
        0.0
    };

    LyapunovReport {
        is_stable: decreasing,
        energy_trajectory: energies,
        convergence_rate,
    }
}

/// Lyapunov stability report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovReport {
    /// Whether the system is Lyapunov stable.
    pub is_stable: bool,
    /// Energy at each timestep.
    pub energy_trajectory: Vec<f64>,
    /// Average convergence rate.
    pub convergence_rate: f64,
}

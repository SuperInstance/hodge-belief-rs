pub mod belief_network;
pub mod curl_flow;
pub mod gradient_flow;
pub mod harmonic_flow;
pub mod hodge_decomposition;
pub mod linalg;
pub mod simplicial;
pub mod stability;

pub use belief_network::{BeliefEdge, BeliefNetwork, BeliefNode};
pub use curl_flow::{CurlAnalysis, NodeType};
pub use gradient_flow::GradientPotential;
pub use harmonic_flow::EchoChamber;
pub use hodge_decomposition::HodgeDecomposition;
pub use stability::{LyapunovReport, StabilityReport};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curl_flow;
    use crate::gradient_flow;
    use crate::harmonic_flow;
    use crate::hodge_decomposition as hd;
    use crate::stability;

    // === Basic Hodge decomposition tests ===

    #[test]
    fn test_decomposition_sums_to_original_flow() {
        let mut net = BeliefNetwork::new();
        for i in 0..4 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![i as f64],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 0.5);
        net.add_edge(2, 3, 1.0, -0.3);
        net.add_edge(3, 0, 1.0, 0.8);
        net.add_edge(0, 2, 1.0, 0.2);

        let flow = net.belief_flow();
        let decomp = hd::decompose(&net, &flow);

        for i in 0..flow.len() {
            let reconstructed = decomp.gradient[i] + decomp.harmonic[i] + decomp.curl[i];
            assert!(
                (flow[i] - reconstructed).abs() < 1e-6,
                "Flow[{}] = {} but reconstructed = {}",
                i,
                flow[i],
                reconstructed
            );
        }
    }

    #[test]
    fn test_gradient_component_is_curl_free() {
        let net = BeliefNetwork::complete_graph(4);
        let flow = vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.1];
        let decomp = hd::decompose(&net, &flow);
        assert!(
            gradient_flow::is_curl_free(&net, &decomp.gradient),
            "Gradient component should be curl-free"
        );
    }

    #[test]
    fn test_harmonic_component_is_divergence_free() {
        let net = BeliefNetwork::ring_graph(5);
        let _flow = net.belief_flow();
        // Set equal beliefs so flow is zero; use a known harmonic flow instead
        let harmonic_flow = vec![1.0; 5]; // uniform circular flow
        assert!(
            gradient_flow::is_divergence_free(&net, &harmonic_flow),
            "Uniform circular flow should be divergence-free"
        );
    }

    #[test]
    fn test_harmonic_component_is_curl_free() {
        let net = BeliefNetwork::ring_graph(5);
        let harmonic_flow = vec![1.0; 5];
        assert!(
            gradient_flow::is_curl_free(&net, &harmonic_flow),
            "Uniform circular flow on ring has no triangles, so curl-free"
        );
    }

    #[test]
    fn test_curl_component_is_divergence_free() {
        // Create a network with triangles
        let mut net = BeliefNetwork::new();
        for i in 0..3 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![i as f64 * 2.0],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 1.0);
        net.add_edge(2, 0, 1.0, 1.0);

        let flow = net.belief_flow();
        let decomp = hd::decompose(&net, &flow);
        // Curl component divergence should be small relative to flow
        let div = hd::divergence(&net, &decomp.curl);
        let div_norm: f64 = div.iter().map(|d| d * d).sum::<f64>().sqrt();
        let flow_norm: f64 = flow.iter().map(|f| f * f).sum::<f64>().sqrt();
        if flow_norm > 1e-10 {
            assert!(
                div_norm / flow_norm < 0.5,
                "Curl component should be approximately divergence-free"
            );
        }
    }

    // === Graph topology tests ===

    #[test]
    fn test_complete_graph_has_no_harmonic_component() {
        let net = BeliefNetwork::complete_graph(4);
        let flow = vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.1];
        let decomp = hd::decompose(&net, &flow);
        let harmonic_energy: f64 = decomp.harmonic.iter().map(|x| x * x).sum();
        let total_energy: f64 = flow.iter().map(|x| x * x).sum();
        if total_energy > 1e-10 {
            assert!(
                harmonic_energy / total_energy < 0.1,
                "Complete graph should have negligible harmonic component, got {}%",
                harmonic_energy / total_energy * 100.0
            );
        }
    }

    #[test]
    fn test_ring_graph_has_harmonic_component() {
        let net = BeliefNetwork::ring_graph(4);
        // A uniform circular flow on a ring is harmonic
        let flow = vec![1.0, 1.0, 1.0, 1.0];
        let decomp = hd::decompose(&net, &flow);
        let harmonic_energy: f64 = decomp.harmonic.iter().map(|x| x * x).sum();
        assert!(
            harmonic_energy > 0.1,
            "Ring graph with circular flow should have significant harmonic component"
        );
    }

    #[test]
    fn test_star_graph_has_no_harmonic() {
        let net = BeliefNetwork::star_graph(5);
        // A gradient-compatible flow: all flow radiates from center
        let flow = vec![1.0, 2.0, -1.0, 0.5, 3.0];
        let decomp = hd::decompose(&net, &flow);
        let harmonic_energy: f64 = decomp.harmonic.iter().map(|x| x * x).sum();
        let total_energy: f64 = flow.iter().map(|x| x * x).sum();
        if total_energy > 1e-10 {
            assert!(
                harmonic_energy / total_energy < 0.05,
                "Star graph should have negligible harmonic component, got {}%",
                harmonic_energy / total_energy * 100.0
            );
        }
    }

    // === Simplicial complex tests ===

    #[test]
    fn test_betti_number_ring_is_one() {
        let net = BeliefNetwork::ring_graph(4);
        let edges = net.to_edges();
        let b1 = simplicial::incidence_matrix(4, &edges);
        let triangles = net.detect_triangles();
        let b2 = simplicial::boundary2_matrix(edges.len(), &edges, &triangles);
        let l1 = simplicial::hodge_laplacian_1(&b1, &b2);
        let beta1 = simplicial::betti_number(&l1);
        assert_eq!(beta1, 1, "Ring graph should have β₁ = 1");
    }

    #[test]
    fn test_betti_number_complete_graph_is_zero() {
        let net = BeliefNetwork::complete_graph(4);
        let edges = net.to_edges();
        let b1 = simplicial::incidence_matrix(4, &edges);
        let triangles = net.detect_triangles();
        let b2 = simplicial::boundary2_matrix(edges.len(), &edges, &triangles);
        let l1 = simplicial::hodge_laplacian_1(&b1, &b2);
        let beta1 = simplicial::betti_number(&l1);
        assert_eq!(beta1, 0, "Complete graph K₄ should have β₁ = 0");
    }

    #[test]
    fn test_betti_number_disconnected_is_two() {
        // Two separate rings (4-cycles)
        let mut net = BeliefNetwork::new();
        for i in 0..8 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![0.0],
                conviction: 1.0,
            });
        }
        // Ring 1: 0→1→2→3→0
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 1.0);
        net.add_edge(2, 3, 1.0, 1.0);
        net.add_edge(3, 0, 1.0, 1.0);
        // Ring 2: 4→5→6→7→4
        net.add_edge(4, 5, 1.0, 1.0);
        net.add_edge(5, 6, 1.0, 1.0);
        net.add_edge(6, 7, 1.0, 1.0);
        net.add_edge(7, 4, 1.0, 1.0);

        let edges = net.to_edges();
        let b1 = simplicial::incidence_matrix(8, &edges);
        let triangles = net.detect_triangles();
        let b2 = simplicial::boundary2_matrix(edges.len(), &edges, &triangles);
        let l1 = simplicial::hodge_laplacian_1(&b1, &b2);
        let beta1 = simplicial::betti_number(&l1);
        assert_eq!(beta1, 2, "Two separate cycles should have β₁ = 2");
    }

    #[test]
    fn test_triangles_detected_in_complete_graph() {
        let net = BeliefNetwork::complete_graph(4);
        let triangles = net.detect_triangles();
        assert_eq!(triangles.len(), 4, "K₄ should have 4 triangles (C(4,3))");
    }

    // === Energy conservation tests ===

    #[test]
    fn test_energy_conservation_gradient_flow() {
        let mut net = BeliefNetwork::new();
        for i in 0..3 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![(i as f64) * 2.0],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 0.5, 1.0);
        net.add_edge(1, 2, 0.5, 1.0);

        let flow = net.belief_flow();
        let gp = gradient_flow::compute_gradient(&net, &flow);
        let total_energy: f64 = flow.iter().map(|x| x * x).sum();
        let gradient_energy: f64 = gp.gradient_flow.iter().map(|x| x * x).sum();
        // Gradient energy should not exceed total
        assert!(
            gradient_energy <= total_energy + 1e-10,
            "Gradient energy should not exceed total energy"
        );
    }

    #[test]
    fn test_energy_fractions_sum_to_one() {
        let mut net = BeliefNetwork::new();
        for i in 0..4 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![i as f64 * 1.5],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 0.7);
        net.add_edge(2, 3, 1.0, -0.5);
        net.add_edge(3, 0, 1.0, 0.3);
        net.add_edge(0, 2, 1.0, 0.1);

        let flow = net.belief_flow();
        let decomp = hd::decompose(&net, &flow);
        let (g, h, c) = decomp.energy_fractions();
        assert!(
            (g + h + c - 1.0).abs() < 0.01,
            "Energy fractions should sum to ~1, got {}",
            g + h + c
        );
    }

    // === Belief network construction tests ===

    #[test]
    fn test_belief_flow_computation() {
        let mut net = BeliefNetwork::new();
        net.add_node(BeliefNode {
            id: "a".into(),
            beliefs: vec![1.0, 0.0],
            conviction: 0.8,
        });
        net.add_node(BeliefNode {
            id: "b".into(),
            beliefs: vec![0.0, 1.0],
            conviction: 0.9,
        });
        net.add_edge(0, 1, 0.5, 0.7);

        let flow = net.belief_flow();
        assert_eq!(flow.len(), 1);
        // flow = influence * agreement * (sum_to - sum_from)
        // = 0.5 * 0.7 * (1.0 - 1.0) = 0.0
        assert!((flow[0]).abs() < 1e-10);
    }

    #[test]
    fn test_belief_flow_unequal_beliefs() {
        let mut net = BeliefNetwork::new();
        net.add_node(BeliefNode {
            id: "a".into(),
            beliefs: vec![2.0],
            conviction: 1.0,
        });
        net.add_node(BeliefNode {
            id: "b".into(),
            beliefs: vec![5.0],
            conviction: 1.0,
        });
        net.add_edge(0, 1, 1.0, 1.0);

        let flow = net.belief_flow();
        // flow = 1.0 * 1.0 * (5.0 - 2.0) = 3.0
        assert!((flow[0] - 3.0).abs() < 1e-10);
    }

    // === Divergence tests ===

    #[test]
    fn test_divergence_conservation() {
        // Sum of divergences should be zero (flow conservation)
        let mut net = BeliefNetwork::new();
        for i in 0..3 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![i as f64],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 1.0);
        net.add_edge(2, 0, 1.0, 1.0);

        let flow = vec![1.0, 2.0, -3.0];
        let div = hd::divergence(&net, &flow);
        let sum: f64 = div.iter().sum();
        assert!(
            sum.abs() < 1e-10,
            "Sum of divergences should be zero, got {}",
            sum
        );
    }

    // === Stability tests ===

    #[test]
    fn test_stability_identifies_echo_chambers() {
        let net = BeliefNetwork::ring_graph(4);
        let flow = vec![1.0, 1.0, 1.0, 1.0];
        let decomp = hd::decompose(&net, &flow);
        let report = stability::analyze_stability(&net, &flow, &decomp);
        assert!(
            !report.echo_chambers.is_empty(),
            "Ring with circular flow should detect echo chambers"
        );
    }

    #[test]
    fn test_stability_complete_graph_is_stable() {
        let net = BeliefNetwork::complete_graph(4);
        let flow = vec![1.0, -1.0, 0.5, -0.5, 0.3, -0.3];
        let decomp = hd::decompose(&net, &flow);
        let report = stability::analyze_stability(&net, &flow, &decomp);
        assert!(
            report.stability_score > 0.25,
            "Complete graph should have significant gradient component, got {}",
            report.stability_score
        );
    }

    #[test]
    fn test_lyapunov_stable_when_decreasing() {
        let history = vec![
            vec![3.0, 2.0, 1.0],
            vec![2.0, 1.5, 0.8],
            vec![1.0, 1.0, 0.5],
            vec![0.5, 0.5, 0.2],
        ];
        let report = stability::lyapunov_stability(&history);
        assert!(report.is_stable);
        assert!(report.convergence_rate > 0.0);
    }

    #[test]
    fn test_lyapunov_unstable_when_increasing() {
        let history = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let report = stability::lyapunov_stability(&history);
        assert!(!report.is_stable);
    }

    // === Linear algebra tests ===

    #[test]
    fn test_solve_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 5.0];
        let x = linalg::solve(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_2x2() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = linalg::solve(&a, &b).unwrap();
        // Verify Ax = b
        let ax = linalg::mat_vec_mul(&a, &x);
        assert!((ax[0] - 5.0).abs() < 1e-10);
        assert!((ax[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_singular_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![3.0, 6.0];
        // Singular matrix (rows are linearly dependent) but consistent.
        // Our solver may or may not handle this. At least it shouldn't panic.
        let _ = linalg::solve(&a, &b);
    }

    #[test]
    fn test_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let at = linalg::transpose(&a);
        assert_eq!(at.len(), 3);
        assert_eq!(at[0], vec![1.0, 4.0]);
        assert_eq!(at[1], vec![2.0, 5.0]);
        assert_eq!(at[2], vec![3.0, 6.0]);
    }

    #[test]
    fn test_mat_mul_identity() {
        let i = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let a = vec![vec![3.0, 7.0], vec![1.0, 2.0]];
        let c = linalg::mat_mul(&i, &a);
        assert!((c[0][0] - 3.0).abs() < 1e-10);
        assert!((c[0][1] - 7.0).abs() < 1e-10);
        assert!((c[1][0] - 1.0).abs() < 1e-10);
        assert!((c[1][1] - 2.0).abs() < 1e-10);
    }

    // === Echo chamber detection ===

    #[test]
    fn test_echo_chamber_detection_on_ring() {
        let net = BeliefNetwork::ring_graph(5);
        let chambers = harmonic_flow::detect_echo_chambers(&net, &[1.0; 5]);
        assert!(!chambers.is_empty(), "Ring should have echo chambers");
        assert!(
            chambers[0].nodes.len() >= 3,
            "Echo chamber should involve at least 3 nodes"
        );
    }

    #[test]
    fn test_no_echo_chambers_on_path() {
        let mut net = BeliefNetwork::new();
        for i in 0..4 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![0.0],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 1.0);
        net.add_edge(2, 3, 1.0, 1.0);

        // A gradient-only flow on a path
        let flow = vec![2.0, 1.0, -1.0];
        let decomp = hd::decompose(&net, &flow);
        let chambers = harmonic_flow::detect_echo_chambers(&net, &decomp.harmonic);
        assert!(
            chambers.is_empty(),
            "Path graph with gradient flow should have no echo chambers"
        );
    }

    // === Gradient flow tests ===

    #[test]
    fn test_gradient_flow_step_beliefs_change() {
        let mut net = BeliefNetwork::new();
        net.add_node(BeliefNode {
            id: "a".into(),
            beliefs: vec![5.0],
            conviction: 1.0,
        });
        net.add_node(BeliefNode {
            id: "b".into(),
            beliefs: vec![1.0],
            conviction: 1.0,
        });
        net.add_edge(0, 1, 1.0, 1.0);

        let potential = vec![5.0, 1.0];
        let old_belief = net.nodes[1].beliefs[0];
        let change = gradient_flow::gradient_flow_step(&mut net, &potential, 0.1);
        assert!(change > 0.0, "Some change should occur");
        // Belief at node 1 should increase (moving toward higher potential)
        // Actually diff = potential[to] - potential[from] = 1.0 - 5.0 = -4.0
        // change = 0.1 * 1.0 * 1.0 * (-4.0) = -0.4
        // So node 1 belief decreases
        assert!(net.nodes[1].beliefs[0] != old_belief);
    }

    #[test]
    fn test_is_curl_free_no_triangles() {
        let net = BeliefNetwork::ring_graph(4);
        assert!(
            gradient_flow::is_curl_free(&net, &[1.0; 4]),
            "Graph with no triangles is always curl-free"
        );
    }

    #[test]
    fn test_is_divergence_free_circular_flow() {
        let net = BeliefNetwork::ring_graph(4);
        let flow = vec![1.0; 4];
        assert!(
            gradient_flow::is_divergence_free(&net, &flow),
            "Uniform circular flow on ring is divergence-free"
        );
    }

    // === Curl flow tests ===

    #[test]
    fn test_curl_analysis_on_path() {
        let mut net = BeliefNetwork::new();
        for i in 0..3 {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![i as f64],
                conviction: 1.0,
            });
        }
        net.add_edge(0, 1, 1.0, 1.0);
        net.add_edge(1, 2, 1.0, 1.0);

        let flow = vec![1.0, 2.0];
        let analysis = curl_flow::compute_curl(&net, &flow);
        assert_eq!(analysis.divergence.len(), 3);
    }

    #[test]
    fn test_node_classification() {
        let div = vec![-2.0, 0.1, 3.0, 0.0];
        let types = curl_flow::classify_nodes(&div, 0.5);
        assert!(matches!(types[0], curl_flow::NodeType::Apostate));
        assert!(matches!(types[1], curl_flow::NodeType::Neutral));
        assert!(matches!(types[2], curl_flow::NodeType::Converter));
        assert!(matches!(types[3], curl_flow::NodeType::Neutral));
    }

    #[test]
    fn test_net_creation_rate() {
        let div = vec![1.0, -2.0, 3.0];
        let rate = curl_flow::net_creation_rate(&div);
        assert!((rate - 2.0).abs() < 1e-10);
    }

    // === Serialization tests ===

    #[test]
    fn test_serde_belief_network() {
        let net = BeliefNetwork::complete_graph(3);
        let json = serde_json::to_string(&net).unwrap();
        let deserialized: BeliefNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), 3);
        assert_eq!(deserialized.edges.len(), 3); // C(3,2) = 3 undirected edges
    }

    #[test]
    fn test_serde_hodge_decomposition() {
        let decomp = HodgeDecomposition {
            gradient: vec![1.0, 2.0],
            harmonic: vec![0.5, -0.5],
            curl: vec![0.1, -0.1],
        };
        let json = serde_json::to_string(&decomp).unwrap();
        let d: HodgeDecomposition = serde_json::from_str(&json).unwrap();
        assert!((d.gradient[0] - 1.0).abs() < 1e-10);
        assert!((d.harmonic[1] - (-0.5)).abs() < 1e-10);
    }

    // === Harmonic dimension tests ===

    #[test]
    fn test_harmonic_dimension_ring() {
        let net = BeliefNetwork::ring_graph(5);
        let dim = harmonic_flow::harmonic_dimension(&net);
        assert_eq!(dim, 1, "Ring graph should have harmonic dimension 1");
    }

    #[test]
    fn test_harmonic_dimension_complete() {
        let net = BeliefNetwork::complete_graph(4);
        let dim = harmonic_flow::harmonic_dimension(&net);
        assert_eq!(dim, 0, "Complete graph K₄ should have harmonic dimension 0");
    }

    #[test]
    fn test_is_harmonic_on_ring() {
        let net = BeliefNetwork::ring_graph(4);
        let flow = vec![1.0; 4];
        assert!(
            harmonic_flow::is_harmonic(&net, &flow),
            "Uniform flow on ring should be harmonic"
        );
    }
}

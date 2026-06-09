//! Tutorial: hodge-belief-rs — Hodge decomposition of agent belief states

use hodge_belief_rs::{
    BeliefNetwork, BeliefNode,
    hodge_decomposition::{decompose, divergence},
    gradient_flow::{compute_gradient, is_curl_free, flow_energy},
    curl_flow::{compute_curl, classify_nodes, net_creation_rate},
    harmonic_flow::{detect_echo_chambers, harmonic_dimension, is_harmonic},
    stability::analyze_stability,
};

fn main() {
    println!("=== Hodge Belief Tutorial ===\n");

    // Part 1: Build belief network
    println!("Part 1: Belief network");
    let mut net = BeliefNetwork::new();
    let n0 = net.add_node(BeliefNode { id: "proposer".into(), beliefs: vec![0.8, 0.2], conviction: 0.9 });
    let n1 = net.add_node(BeliefNode { id: "skeptic".into(), beliefs: vec![0.3, 0.7], conviction: 0.6 });
    let n2 = net.add_node(BeliefNode { id: "moderator".into(), beliefs: vec![0.5, 0.5], conviction: 0.4 });
    net.add_edge(n0, n1, 0.5, -0.2);
    net.add_edge(n1, n2, 0.3, 0.1);
    net.add_edge(n2, n0, 0.4, 0.4);
    println!("  {} nodes, {} edges", net.num_nodes(), net.num_edges());
    println!();

    // Part 2: Belief flow & energy
    println!("Part 2: Belief flow");
    let flow = net.belief_flow();
    println!("  Flow: {:?}", flow.iter().map(|f| format!("{:.3}", f)).collect::<Vec<_>>());
    println!("  Energy: {:.4}", flow_energy(&flow));
    println!();

    // Part 3: Hodge decomposition
    println!("Part 3: Hodge decomposition");
    let decomp = decompose(&net, &flow);
    let (grad_frac, curl_frac, harm_frac) = decomp.energy_fractions();
    println!("  Gradient (resolvable):     {:.4}", grad_frac);
    println!("  Curl (cyclic disputes):    {:.4}", curl_frac);
    println!("  Harmonic (irreconcilable): {:.4}", harm_frac);
    println!();

    // Part 4: Gradient potential
    println!("Part 4: Gradient potential");
    let _grad = compute_gradient(&net, &flow);
    println!("  Is curl-free: {}", is_curl_free(&net, &flow));
    println!();

    // Part 5: Curl & divergence
    println!("Part 5: Curl analysis");
    let _curl = compute_curl(&net, &flow);
    let div = divergence(&net, &flow);
    let node_types = classify_nodes(&div, 0.1);
    println!("  Net creation rate: {:.4}", net_creation_rate(&div));
    println!("  Node types: {:?}", node_types);
    println!();

    // Part 6: Echo chambers
    println!("Part 6: Echo chamber detection");
    let chambers = detect_echo_chambers(&net, &flow);
    println!("  Harmonic dimension: {}", harmonic_dimension(&net));
    println!("  Is harmonic: {}", is_harmonic(&net, &flow));
    println!("  Echo chambers: {}", chambers.len());
    println!();

    // Part 7: Stability
    println!("Part 7: Stability analysis");
    let _stability = analyze_stability(&net, &flow, &decomp);
    println!("  Stability computed");
    println!();

    // Part 8: Complete graph
    println!("Part 8: Complete graph (4 nodes)");
    let complete = BeliefNetwork::complete_graph(4);
    let flow2 = complete.belief_flow();
    let decomp2 = decompose(&complete, &flow2);
    let (g, c, h) = decomp2.energy_fractions();
    println!("  gradient={:.4} curl={:.4} harmonic={:.4}", g, c, h);
}

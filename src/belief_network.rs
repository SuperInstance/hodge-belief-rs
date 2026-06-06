use serde::{Deserialize, Serialize};

use crate::simplicial::Edge;

/// A single agent holding a belief state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefNode {
    pub id: String,
    /// Probability vector over belief topics (need not sum to 1).
    pub beliefs: Vec<f64>,
    /// Overall conviction strength [0, 1].
    pub conviction: f64,
}

/// A directed influence edge between two agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefEdge {
    pub from: usize,
    pub to: usize,
    /// Influence weight ∈ [0, 1].
    pub influence: f64,
    /// Agreement score between the two agents ∈ [-1, 1].
    pub agreement: f64,
}

/// The full belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefNetwork {
    pub nodes: Vec<BeliefNode>,
    pub edges: Vec<BeliefEdge>,
}

impl BeliefNetwork {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: BeliefNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn add_edge(&mut self, from: usize, to: usize, influence: f64, agreement: f64) {
        self.edges.push(BeliefEdge {
            from,
            to,
            influence,
            agreement,
        });
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Compute belief flow on each edge: the signed difference in belief energy
    /// weighted by influence and agreement.
    ///
    /// flow[e] = influence[e] * agreement[e] * (Σ beliefs[to] - Σ beliefs[from])
    pub fn belief_flow(&self) -> Vec<f64> {
        self.edges
            .iter()
            .map(|e| {
                let sum_from: f64 = self.nodes[e.from].beliefs.iter().sum();
                let sum_to: f64 = self.nodes[e.to].beliefs.iter().sum();
                e.influence * e.agreement * (sum_to - sum_from)
            })
            .collect()
    }

    /// Convert to internal edge representation.
    pub fn to_edges(&self) -> Vec<Edge> {
        self.edges
            .iter()
            .map(|e| Edge {
                from: e.from,
                to: e.to,
                weight: e.influence,
            })
            .collect()
    }

    /// Detect triangles (3-cliques) in the network.
    pub fn detect_triangles(&self) -> Vec<[usize; 3]> {
        let n = self.num_nodes();
        let mut adj = vec![vec![false; n]; n];
        for e in &self.edges {
            adj[e.from][e.to] = true;
        }
        let mut triangles = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                if !adj[a][b] && !adj[b][a] {
                    continue;
                }
                for c in (b + 1)..n {
                    let ab = adj[a][b] || adj[b][a];
                    let ac = adj[a][c] || adj[c][a];
                    let bc = adj[b][c] || adj[c][b];
                    if ab && ac && bc {
                        triangles.push([a, b, c]);
                    }
                }
            }
        }
        triangles
    }

    /// Build a complete graph on n nodes with uniform influence and agreement.
    /// Uses one directed edge per unordered pair (i < j) for a clean simplicial complex.
    pub fn complete_graph(n: usize) -> Self {
        let mut net = Self::new();
        for i in 0..n {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![0.0],
                conviction: 1.0,
            });
        }
        for i in 0..n {
            for j in (i + 1)..n {
                net.add_edge(i, j, 1.0, 1.0);
            }
        }
        net
    }

    /// Build a ring graph on n nodes.
    pub fn ring_graph(n: usize) -> Self {
        let mut net = Self::new();
        for i in 0..n {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![0.0],
                conviction: 1.0,
            });
        }
        for i in 0..n {
            net.add_edge(i, (i + 1) % n, 1.0, 1.0);
        }
        net
    }

    /// Build a star graph: center node 0 connected to all others.
    pub fn star_graph(n: usize) -> Self {
        let mut net = Self::new();
        for i in 0..n {
            net.add_node(BeliefNode {
                id: format!("n{i}"),
                beliefs: vec![0.0],
                conviction: 1.0,
            });
        }
        for i in 1..n {
            net.add_edge(0, i, 1.0, 1.0);
        }
        net
    }
}

impl Default for BeliefNetwork {
    fn default() -> Self {
        Self::new()
    }
}

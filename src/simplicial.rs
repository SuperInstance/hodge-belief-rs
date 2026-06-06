use serde::{Deserialize, Serialize};

/// Internal edge representation using indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Build the incidence matrix B₁ (nodes × edges) of a simplicial complex.
///
/// For each directed edge e = (u → v):
///   B₁[u][e] = -1  (tail)
///   B₁[v][e] = +1  (head)
pub fn incidence_matrix(num_nodes: usize, edges: &[Edge]) -> Vec<Vec<f64>> {
    let m = edges.len();
    let mut b1 = vec![vec![0.0; m]; num_nodes];
    for (j, e) in edges.iter().enumerate() {
        b1[e.from][j] = -1.0;
        b1[e.to][j] = 1.0;
    }
    b1
}

/// Compute the graph Laplacian L₀ = B₁ · B₁ᵀ.
pub fn graph_laplacian(b1: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b1.len();
    let m = if n > 0 { b1[0].len() } else { 0 };
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += b1[i][k] * b1[j][k];
            }
            l[i][j] = sum;
        }
    }
    l
}

/// Compute the edge Laplacian L₁ = B₁ᵀ · B₁ (up-Laplacian on edges).
pub fn edge_laplacian(b1: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b1.len();
    let m = if n > 0 { b1[0].len() } else { 0 };
    let mut l = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += b1[k][i] * b1[k][j];
            }
            l[i][j] = sum;
        }
    }
    l
}

/// Build boundary operator B₂ (edges × triangles).
/// For a triangle (a, b, c) with orientation, the boundary is:
///   (b,c) - (a,c) + (a,b)
pub fn boundary2_matrix(
    _num_edges: usize,
    edges: &[Edge],
    triangles: &[[usize; 3]],
) -> Vec<Vec<f64>> {
    let m = edges.len();
    let t = triangles.len();
    let mut b2 = vec![vec![0.0; t]; m];

    for (col, tri) in triangles.iter().enumerate() {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        // Boundary of (a,b,c) = (b,c) - (a,c) + (a,b)
        let pairs: [(usize, usize, f64); 3] = [(b, c, 1.0), (a, c, -1.0), (a, b, 1.0)];
        for (u, v, sign) in &pairs {
            if let Some(row) = edges.iter().position(|e| e.from == *u && e.to == *v) {
                b2[row][col] = *sign;
            } else if let Some(row) = edges.iter().position(|e| e.from == *v && e.to == *u) {
                b2[row][col] = -(*sign);
            }
        }
    }
    b2
}

/// Full 1-Laplacian: L₁ = B₁ᵀB₁ + B₂B₂ᵀ (Hodge Laplacian on edges).
pub fn hodge_laplacian_1(b1: &[Vec<f64>], b2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = if !b1.is_empty() { b1[0].len() } else if !b2.is_empty() { b2.len() } else { 0 };
    let n = b1.len();
    let t = if !b2.is_empty() { b2[0].len() } else { 0 };

    // L₁_up = B₁ᵀ B₁
    let mut l = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += b1[k][i] * b1[k][j];
            }
            l[i][j] = sum;
        }
    }
    // L₁_down = B₂ B₂ᵀ
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..t {
                sum += b2[i][k] * b2[j][k];
            }
            l[i][j] += sum;
        }
    }
    l
}

/// Compute Betti numbers: β₀ = dim(ker L₀), β₁ = dim(ker L₁), β₂ = dim(ker L₂).
/// For now: β₁ = dim(ker L₁).
pub fn betti_number(laplacian: &[Vec<f64>]) -> usize {
    let n = laplacian.len();
    if n == 0 {
        return 0;
    }
    // Count zero eigenvalues via rank deficiency
    let rank = matrix_rank(laplacian);
    n - rank
}

/// Matrix rank via Gaussian elimination.
fn matrix_rank(a: &[Vec<f64>]) -> usize {
    if a.is_empty() {
        return 0;
    }
    let m = a.len();
    let n = a[0].len();
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rank = 0;
    for col in 0..n {
        // Find pivot
        let pivot = (rank..m).find(|&r| mat[r][col].abs() > 1e-10);
        if let Some(p) = pivot {
            mat.swap(rank, p);
            let piv = mat[rank][col];
            for row in (rank + 1)..m {
                let factor = mat[row][col] / piv;
                for c in 0..n {
                    mat[row][c] -= factor * mat[rank][c];
                }
            }
            rank += 1;
        }
    }
    rank
}

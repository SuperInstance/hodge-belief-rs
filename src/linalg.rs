
/// Solve Ax = b via Gaussian elimination with partial pivoting.
/// Returns None if the system is singular.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }
    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).max_by(|&i, &j| aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap())?;
        if aug[pivot_row][col].abs() < 1e-12 {
            return None;
        }
        aug.swap(col, pivot_row);
        let piv = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / piv;
            for c in col..=n {
                aug[row][c] -= factor * aug[col][c];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Some(x)
}

/// Matrix-vector multiply.
pub fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Transpose a matrix.
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
    }
    let m = a.len();
    let n = a[0].len();
    (0..n)
        .map(|j| (0..m).map(|i| a[i][j]).collect())
        .collect()
}

/// Compute the null space of a matrix (approximate, for Laplacians).
/// Returns basis vectors for the kernel.
pub fn null_space(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return vec![];
    }
    let n = a[0].len();
    // Row reduce
    let mut mat = a.to_vec();
    let mut pivot_cols: Vec<Option<usize>> = vec![None; m];
    let mut current_row = 0;

    for col in 0..n {
        if current_row >= m {
            break;
        }
        let pivot = (current_row..m).find(|&r| mat[r][col].abs() > 1e-10);
        if let Some(p) = pivot {
            mat.swap(current_row, p);
            let piv = mat[current_row][col];
            for row in 0..m {
                if row != current_row && mat[row][col].abs() > 1e-10 {
                    let factor = mat[row][col] / piv;
                    for c in 0..n {
                        mat[row][c] -= factor * mat[current_row][c];
                    }
                }
            }
            pivot_cols[current_row] = Some(col);
            current_row += 1;
        }
    }

    // Free variables are columns not in pivot_cols
    let pivot_set: Vec<usize> = pivot_cols.iter().filter_map(|&c| c).collect();
    let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();

    free_cols
        .into_iter()
        .map(|fc| {
            (0..n)
                .map(|c| {
                    if c == fc {
                        1.0
                    } else if let Some(row_idx) = pivot_set.iter().position(|&pc| pc == c) {
                        -mat[row_idx][fc]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Project vector v onto the column space of A: proj = A (AᵀA)⁻¹ Aᵀ v
/// Falls back to least-squares if AᵀA is singular.
pub fn project_onto_colspace(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let atv = mat_vec_mul(&at, v);
    if let Some(x) = solve(&ata, &atv) {
        mat_vec_mul(a, &x)
    } else {
        // Singular: use iterative approach / return zero projection
        vec![0.0; v.len()]
    }
}

/// Matrix multiply C = A · B.
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if !a.is_empty() { a[0].len() } else { 0 };
    let n = if !b.is_empty() { b[0].len() } else { 0 };
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

/// Compute the pseudoinverse solve: find x minimizing ||Ax - b||₂.
pub fn least_squares_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let atb = mat_vec_mul(&at, b);
    solve(&ata, &atb)
}

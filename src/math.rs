#![allow(dead_code)]

use rand::Rng;
use std::ops::{Add, Div, Mul, Sub};

// ======================================== VECTOR3 ========================================
// MARK: VECTOR3
#[derive(Clone, Debug)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn show(&self, n: &str) {
        println!("{n}({}, {}, {})", self.x, self.y, self.z);
    }

    pub fn to_quaternion(&self) -> Quaternion {
        Quaternion::new(0.0, self.x, self.y, self.z)
    }

    pub fn normalize(&self) -> Self {
        let norm: f64 = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl Add for Vector3 {
    type Output = Vector3;

    fn add(self, other: Vector3) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self, other: Vector3) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;
    fn mul(self, other: f64) -> Self {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Div<f64> for Vector3 {
    type Output = Vector3;
    fn div(self, other: f64) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}
// ======================================== END VECTOR3 ========================================

// ! ===========================================================================================

// ======================================== QUATERNION ========================================
// MARK: QUATERNION
#[derive(Copy, Clone, Debug)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn show(&self) {
        println!("q({} {} {} {})", self.w, self.x, self.y, self.z);
    }

    pub fn conj(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn normalize(&self) -> Self {
        let norm: f64 =
            (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        if norm <= 0.0 {
            return Self {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
        }

        Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    pub fn add(&self, other: &Quaternion) -> Self {
        Self {
            w: self.w + other.w,
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn dot(&self, other: &Quaternion) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.y * other.z + self.z * other.x + self.x * other.y,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    pub fn mul(&self, n: f64) -> Self {
        Self {
            w: self.w * n,
            x: self.x * n,
            y: self.y * n,
            z: self.z * n,
        }
    }

    pub fn sub(&self, n: f64) -> Self {
        Self {
            w: self.w - n,
            x: self.x - n,
            y: self.y - n,
            z: self.z - n,
        }
    }

    pub fn subq(&self, other: &Quaternion) -> Self {
        self.add(&other.mul(-1.0))
    }
}
// ======================================== END QUATERNION ========================================

// MARK: - MATRIX
#[derive(Clone, Debug)]
pub struct Matrix {
    pub m: Vec<Vec<f64>>,
}
impl Matrix {
    pub fn new(m: Vec<Vec<f64>>) -> Matrix {
        Matrix { m }
    }

    // MARK: Random
    pub fn random(r: i32, c: i32) -> Matrix {
        let mut m = vec![vec![0.0; c as usize]; r as usize];
        for i in 0..r {
            for j in 0..c {
                m[i as usize][j as usize] = rand::thread_rng().gen::<f64>();
            }
        }
        Matrix::new(m)
    }

    // MARK : Concatenate
    // Concatenates the rows of `other` matrix to `self`
    pub fn concatenate(&self, other: &Matrix) -> Matrix {
        assert!(
            self.m.len() == other.m.len(),
            "Matrices must have the same number of rows in order to concatenate"
        );
        let mut concatenated_matrix = self.m.clone(); // Clone the current matrix's rows

        // Append all rows from the other matrix
        for row in &other.m {
            concatenated_matrix.push(row.clone());
        }

        Matrix::new(concatenated_matrix)
    }

    // MARK: Size
    pub fn size(&self) -> (i32, i32) {
        println!("{}x{}", self.m.len(), self.m[0].len());
        (self.m.len() as i32, self.m[0].len() as i32)
    }

    // MARK: Sigmoid
    pub fn sigmoid(&self) -> Matrix {
        let mut m = self.m.clone();

        for i in 0..m.len() {
            for j in 0..m[0].len() {
                m[i][j] = 1.0 / (1.0 + (-m[i][j]).exp());
            }
        }

        Matrix::new(m)
    }

    // MARK: Tanh
    pub fn tanh(&self) -> Matrix {
        let mut m = self.m.clone();

        for i in 0..m.len() {
            for j in 0..m[0].len() {
                m[i][j] = m[i][j].tanh();
            }
        }

        Matrix::new(m)
    }

    // MARK : Diagonal
    pub fn diag(&self, n: &Matrix) -> Matrix {
        let self_size = self.m.len();
        let n_size = n.m.len();
        let total_size = self_size + n_size;

        // Initialize the resulting matrix with zeroes
        let mut result = vec![vec![0.0; total_size]; total_size];

        // Place 'self' on the diagonal
        for i in 0..self_size {
            for j in 0..self_size {
                result[i][j] = self.m[i][j];
            }
        }

        // Place 'n' on the diagonal, offset by the size of 'self'
        for i in 0..n_size {
            for j in 0..n_size {
                result[self_size + i][self_size + j] = n.m[i][j];
            }
        }

        Matrix::new(result)
    }

    // MARK: Identity
    pub fn identity(size: usize) -> Matrix {
        let mut result = Vec::new();
        for i in 0..size {
            result.push(Vec::new());
            for j in 0..size {
                if i == j {
                    result[i].push(1.0);
                } else {
                    result[i].push(0.0);
                }
            }
        }
        Matrix::new(result)
    }

    // MARK: Transpose
    pub fn transpose(&self) -> Matrix {
        let mut result = Vec::new();
        for i in 0..self.m[0].len() {
            result.push(Vec::new());
            for j in 0..self.m.len() {
                result[i].push(self.m[j][i]);
            }
        }
        Matrix::new(result)
    }

    // MARK: Add
    pub fn add(&self, other: &Matrix) -> Matrix {
        // Ensure both matrices have the same dimensions
        assert!(
        self.m.len() == other.m.len() && self.m[0].len() == other.m[0].len(),
        "{}", format!("(Add) Dimensions of matrix a must match dimensions of matrix b\n\nmatrix a:\n{}\nmatrix b:\n{}", self.prettify(), other.prettify())
    );

        let mut result = Vec::new();

        for i in 0..self.m.len() {
            let mut row = Vec::new();
            for j in 0..self.m[i].len() {
                // Perform element-wise addition
                row.push(self.m[i][j] + other.m[i][j]);
            }
            result.push(row);
        }
        Matrix::new(result)
    }

    // MARK: Subtract
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert!(
            self.m[0].len() == other.m.len(),
            "(Subtract) Number of columns of matrix a must match number of rows of matrix b"
        );
        let mut result = Vec::new();
        for i in 0..self.m.len() {
            result.push(Vec::new());
            for j in 0..self.m[0].len() {
                result[i].push(self.m[i][j] - other.m[i][j]);
            }
        }
        Matrix::new(result)
    }

    pub fn subn(&self, n: f64) -> Matrix {
        let mut result: Matrix = self.clone();

        for i in 0..result.m.len() {
            for j in 0..result.m[0].len() {
                result.m[i][j] -= n;
            }
        }
        result
    }

    // MARK: Multiply
    pub fn muln(&self, n: f64) -> Matrix {
        let mut result = self.clone();

        for i in 0..result.m.len() {
            for j in 0..result.m[0].len() {
                result.m[i][j] *= n;
            }
        }

        return result;
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert!(
            self.size() == other.size(),
            "(Element-wise multiplication) Matrices must be the same size",
        );

        let mut result = Vec::new();

        for i in 0..self.m.len() {
            let mut row = Vec::new();
            for j in 0..self.m[i].len() {
                row.push(self.m[i][j] * other.m[i][j]);
            }
            result.push(row);
        }

        Matrix::new(result)
    }

    // MARK: Dot
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert!(
            self.m[0].len() == other.m.len(),
            "{}", format!("(Dot) Number of columns of matrix a must match number of rows of matrix b\n\nmatrix a:\n{}\nmatrix b:\n{}", self.prettify(), other.prettify())
        );

        let mut result = Vec::new();

        // Iterate over rows of the first matrix
        for i in 0..self.m.len() {
            let mut row_result = Vec::new();
            // Iterate over columns of the second matrix
            for j in 0..other.m[0].len() {
                let mut sum = 0.0;
                // Compute dot product of the i-th row of 'self' and j-th column of 'other'
                for k in 0..self.m[0].len() {
                    sum += self.m[i][k] * other.m[k][j];
                }
                row_result.push(sum);
            }
            result.push(row_result);
        }
        Matrix::new(result)
    }

    // MARK: Inverse
    pub fn inverse(&self) -> Matrix {
        assert!(
            self.m.len() == self.m[0].len(),
            "(Inverse) Matrix must be square"
        );

        let matrix_size = self.m.len();
        let mut augmented_matrix = self.m.clone();
        let mut inverse_matrix_m = Matrix::identity(matrix_size).m;

        for i in 0..matrix_size {
            let mut max = i;
            for row in i + 1..matrix_size {
                if augmented_matrix[row][i].abs() > augmented_matrix[max][i].abs() {
                    max = row;
                }
            }

            if augmented_matrix[max][i] == 0.0 {
                println!("====> ERROR: Matrix is singular");
                //TODO: Instead of printing, return an error
            }

            augmented_matrix.swap(i, max);
            inverse_matrix_m.swap(i, max);

            let i_val = augmented_matrix[i][i];

            for col in 0..matrix_size {
                augmented_matrix[i][col] /= i_val;
                inverse_matrix_m[i][col] /= i_val;
            }

            for row in 0..matrix_size {
                if row != i {
                    let factor = augmented_matrix[row][i];

                    for col in 0..matrix_size {
                        augmented_matrix[row][col] -= factor * augmented_matrix[i][col];
                        inverse_matrix_m[row][col] -= factor * inverse_matrix_m[i][col];
                    }
                }
            }
        }

        Matrix::new(inverse_matrix_m)
    }

    pub fn normalize(&self) -> Self {
        let mut result = self.clone();
        let mut sum = 0.0;

        for i in 0..self.m.len() {
            for j in 0..self.m[0].len() {
                sum += self.m[i][j] * self.m[i][j];
            }
        }

        //? Avoiding division by zero
        //? If true then matrix is a zero matrix and should be returned as is
        if sum.is_nan() || sum == 0.0 {
            return result;
        }

        sum = sum.sqrt();
        for i in 0..self.m.len() {
            for j in 0..self.m[0].len() {
                result.m[i][j] /= sum;
            }
        }
        result
    }

    // MARK: Power
    pub fn pow(&self, n: usize) -> Matrix {
        let mut result = self.clone();
        for _ in 0..n {
            result = result.dot(&self);
        }
        result
    }

    // MARK: Show
    pub fn show(&self) {
        println!("matrix");
        for vec in &self.m {
            println!("[{}]", Self::format_vector(vec).join(", "));
        }
    }

    // MARK: Prettify
    pub fn prettify(&self) -> String {
        let mut result = String::new();
        for vec in &self.m {
            let formatted = Self::format_vector(vec).join(", ");
            result.push('[');
            result.push_str(&formatted);
            result.push_str("]\n");
        }
        result
    }

    fn format_vector(vec: &[f64]) -> Vec<String> {
        vec.iter().map(|&num| format!("{:.3}", num)).collect()
    }

    // MARK: - To quat
    pub fn to_quat(&self) -> Quaternion {
        assert!(
            self.m.len() == 1 && self.m[0].len() == 4,
            "Matrix must be 1x4 to convert to quaternion"
        );

        Quaternion::new(self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3])
    }
}

// ======================================== MACROS ========================================
// MARK: MACROS
#[macro_export]
macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Vector3::new($x, $y, $z)
    };
}

#[macro_export]
macro_rules! quat {
    {$w:expr, $x:expr, $y:expr, $z:expr} => {
        Quaternion::new($w, $x, $y, $z)
    };
}

#[macro_export]
macro_rules! matrix {
    ($($vec:expr), + $(,)?) => {
        Matrix::new(vec![$($vec),+])
    };
}
// ======================================== END MACROS ========================================

pub fn normalize(v: Vec<f64>) -> f64 {
    let mut norm: f64 = 0.;

    for val in v {
        norm += val * val;
    }

    norm.sqrt()
}

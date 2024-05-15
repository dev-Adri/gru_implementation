use std::fmt::Write;

pub struct Matrix {
    pub rows: Vec<Vec<f64>>,
    pub cols: Vec<Vec<f64>>
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Matrix {
        if data.is_empty() || data[0].is_empty() {
            // Handle the case where data is empty or contains empty rows.
            // This might need to be adjusted based on how you want to handle these cases.
            return Matrix { rows: vec![], cols: vec![] };
        }

        // Ensure all rows have the same length to form a valid matrix
        let first_row_len = data[0].len();

        for row in &data {
            if row.len() != first_row_len {
                panic!("All rows must have the same number of elements");
            }
        }

        let mut columns: Vec<Vec<f64>> = Vec::new();

        for i in 0..first_row_len {
            let mut column: Vec<f64> = Vec::new();
            for row in &data {
                column.push(row[i]);
            }
            columns.push(column);
        }

        Matrix {
            rows: data,
            cols: columns
        }
    }

    pub fn show(&self) {
        let mut mt = String::new();

        for vec in &self.rows {
            // mt.push_str(format!("{:?}\n", vec));
            write!(&mut mt, "{:?}\n", vec).unwrap();
        }

        println!("{}", mt);
    }
}

impl std::ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols.is_empty() || rhs.rows.is_empty() || self.cols.len() != rhs.rows[0].len() {
            // Handle the case where matrices cannot be multiplied due to incompatible dimensions.
            panic!("Matrices have incompatible dimensions for multiplication.");
        }

        let mut result_data: Vec<Vec<f64>> = Vec::new();

        for row in &self.rows {
            let mut result_row: Vec<f64> = Vec::new();
            for col in &rhs.cols {
                let mut sum = 0.0;
                for (a, b) in row.iter().zip(col.iter()) {
                    sum += a * b;
                }
                result_row.push(sum);
            }
            result_data.push(result_row);
        }

        // Assuming Matrix::new handles the creation of the cols from the rows
        Matrix::new(result_data)
    }
}

#[macro_export]
macro_rules! matrix {
    ($($vec:expr), + $(,)?) => {
        Matrix::new(vec![$($vec),+])
    };
}

#![allow(dead_code)]
#![allow(unused_variables)]

use super::math::Matrix;

use std::fs::File;
use std::io::{self, Write};

// MARK: GruCell
//? ~~ todo ~~
#[derive(Clone, Debug)]
pub struct GruNetwork {}
impl GruNetwork {}

#[derive(Clone, Debug)]
pub struct GruCell {
    reset_xw: Matrix,
    reset_hw: Matrix,
    reset_b: Matrix,

    update_xw: Matrix,
    update_hw: Matrix,
    update_b: Matrix,

    state_xw: Matrix,
    state_b: Matrix,
}

impl GruCell {
    /// h -> Hidden state size
    ///
    /// i -> Input size
    pub fn new(h: i32, i: i32) -> GruCell {
        // hidden state and input should be vectors

        // size of input weights -> sizeof(hidden state) x sizeof(input)
        // size of hidden state weights -> sizeof(hidden state) x sizeof(hidden state)
        // size of biases -> sizeof(hidden state)

        GruCell {
            reset_xw: Matrix::random(h, i),
            reset_hw: Matrix::random(h, h),
            reset_b: Matrix::random(h, 1),

            update_xw: Matrix::random(h, i),
            update_hw: Matrix::random(h, h),
            update_b: Matrix::random(h, 1),

            state_xw: Matrix::random(h, i),
            state_b: Matrix::random(h, 1),
        }
    }

    pub fn update_cell(&mut self, input: &Matrix, hidden: &Matrix) -> Matrix {
        let input: &Matrix = &input.transpose();
        let hidden: &Matrix = &hidden.transpose();

        // ======= Gates =======
        let reset_gate: Matrix = self
            .reset_hw
            .clone()
            .dot(hidden)
            .add(&self.reset_xw.clone().dot(input))
            .add(&self.reset_b)
            .sigmoid();

        let update_gate: Matrix = self
            .update_hw
            .clone()
            .dot(hidden)
            .add(&self.update_xw.clone().dot(input))
            .add(&self.update_b)
            .sigmoid();

        let canditate_state_gate: Matrix = self
            .state_xw
            .clone()
            .dot(input)
            .add(&reset_gate)
            .add(&self.state_b)
            .tanh();
        // ======== Gates ========

        let minus_one_update: Matrix = reset_gate.clone().subn(1.0).muln(-1.0);
        let h_dot_update: Matrix = hidden.clone().mul(&update_gate.clone());

        let can_gate_dot_min_one_update: Matrix =
            canditate_state_gate.clone().mul(&minus_one_update);

        h_dot_update.add(&can_gate_dot_min_one_update)
    }

    pub fn loss(&self, input: &Matrix, hidden: &Matrix, output: &Matrix) -> f64 {
        todo!()
    }

    pub fn train_cell(&mut self, cost: f64) {
        todo!()
    }

    pub fn save_cell(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;

        // Helper function to serialize a matrix
        fn serialize_matrix(matrix: &Matrix) -> String {
            matrix
                .m
                .iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter()
                            .map(|&val| val.to_string())
                            .collect::<Vec<String>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<String>>()
                .join(";")
        }

        // Serialize and write each field to the file
        writeln!(file, "reset_xw: {}", serialize_matrix(&self.reset_xw))?;
        writeln!(file, "reset_hw: {}", serialize_matrix(&self.reset_hw))?;
        writeln!(file, "reset_b: {}", serialize_matrix(&self.reset_b))?;
        writeln!(file, "update_xw: {}", serialize_matrix(&self.update_xw))?;
        writeln!(file, "update_hw: {}", serialize_matrix(&self.update_hw))?;
        writeln!(file, "update_b: {}", serialize_matrix(&self.update_b))?;
        writeln!(file, "state_xw: {}", serialize_matrix(&self.state_xw))?;
        writeln!(file, "state_b: {}", serialize_matrix(&self.state_b))?;

        Ok(())
    }
}

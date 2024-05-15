#![allow(unused_imports)]

mod gru;
mod math;
mod nn;

use gru::GruCell;
use math::Matrix;
use nn::NeuralNetwork;

use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "full");
    // h = hidden size
    // i = input size
    // let mut x: GruCell = GruCell::new(4, 3);
    // x.update_cell(
    //     &matrix!(vec![0.32, 0.12, 0.33]),   // Input
    //     &matrix!(vec![0.1, 0.2, 0.3, 0.4]), // Hidden
    // )
    // .show();

    // x.save_cell("SAVED.grucell").unwrap_or(());

    let nn: NeuralNetwork = NeuralNetwork::construct(8, 12, 0, 0, 0);
    nn.feed_forward(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
}

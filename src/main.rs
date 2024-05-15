mod gru;
mod math;

use gru::GruCell;
use math::Matrix;

use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    // h = hidden size
    // i = input size
    let mut x: GruCell = GruCell::new(4, 3);
    x.update_cell(
        &matrix!(vec![0.32, 0.12, 0.33]),   // Input
        &matrix!(vec![0.1, 0.2, 0.3, 0.4]), // Hidden
    )
    .show();

    x.save_cell("SAVED.grcell").unwrap_or(());
}

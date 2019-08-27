pub mod activation {
    pub fn sigmoid(val: f64) -> f64 {
        1.0 / (1.0 + (-1.0 * val).exp())
    }
}

pub mod loss {
    pub fn squared(desired: f64, guess: f64) -> f64 {
        (desired - guess).powi(2) * 0.5
    }
}

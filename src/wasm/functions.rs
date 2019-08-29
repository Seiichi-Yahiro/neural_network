pub mod activation {
    pub fn sigmoid(val: f64) -> f64 {
        1.0 / (1.0 + (-val).exp())
    }

    pub fn sigmoid_derived(val: f64) -> f64 {
        val * (1.0 - val)
    }
}

pub mod loss {
    pub fn squared(expected: f64, guess: f64) -> f64 {
        (expected - guess).powi(2) * 0.5
    }

    pub fn squared_derived(expected: f64, guess: f64) -> f64 {
        expected - guess
    }
}

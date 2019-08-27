use rand::Rng;
use std::option::Option;

pub struct Neuron {
    weights: Vec<f64>,
}

impl Neuron {
    pub fn new(number_of_inputs: i32) -> Self {
        let mut weights: Vec<f64> = vec![];

        for _ in 0..number_of_inputs {
            weights.push(rand::thread_rng().gen_range(-1.0, 1.0))
        }

        Self { weights }
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>, activate: &Fn(f64) -> f64) -> Option<f64> {
        if inputs.len() != self.weights.len() {
            return None;
        }

        let mut sum: f64 = 0.0;

        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }

        Some(activate(sum))
    }

    /* pub fn train(&mut self, inputs: Vec<f64>, desired: f64) {
        let guess = self.feed_forward(&inputs);
        let error = desired - guess;

        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_constant * error * inputs[i];
        }
    }*/
}

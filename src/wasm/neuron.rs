use rand::Rng;
use wasm_bindgen::__rt::core::cell::RefCell;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub last_output: RefCell<Option<f64>>,
    pub delta: f64,
}

impl Neuron {
    pub fn new(number_of_inputs: u32) -> Self {
        let mut rng = rand::thread_rng();

        let weights: Vec<f64> = (0..number_of_inputs)
            .map(|_| rng.gen_range(-1.0, 1.0))
            .collect();

        Self {
            weights,
            bias: rng.gen_range(-1.0, 1.0),
            last_output: RefCell::new(None),
            delta: 0.0,
        }
    }

    pub fn load(weights: &[f64], bias: f64) -> Self {
        Self {
            weights: Vec::from(weights),
            bias,
            last_output: RefCell::new(None),
            delta: 0.0,
        }
    }

    pub fn propagate_forward(&self, inputs: &[f64], activate: &dyn Fn(f64) -> f64) -> f64 {
        if inputs.len() != self.weights.len() {
            panic!("Number of inputs doesn't match number of input neurons");
        }

        let sum = self.bias
            + self
                .weights
                .iter()
                .zip(inputs)
                .map(|(weight, input)| weight * input)
                .sum::<f64>();

        let output = activate(sum);
        *self.last_output.borrow_mut() = Some(output);
        output
    }
}

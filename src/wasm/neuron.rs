use rand::Rng;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub last_output: Option<f64>,
    pub delta: f64,
}

impl Neuron {
    pub fn new(number_of_inputs: u32) -> Self {
        let mut weights: Vec<f64> = vec![];

        for _ in 0..number_of_inputs {
            weights.push(rand::thread_rng().gen_range(-1.0, 1.0))
        }

        Self {
            weights,
            bias: rand::thread_rng().gen_range(-1.0, 1.0),
            last_output: None,
            delta: 0.0,
        }
    }

    pub fn load(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            weights,
            bias,
            last_output: None,
            delta: 0.0,
        }
    }

    pub fn propagate_forward(&mut self, inputs: &Vec<f64>, activate: &dyn Fn(f64) -> f64) -> f64 {
        if inputs.len() != self.weights.len() {
            panic!("Number of inputs doesn't match number of input neurons");
        }

        let mut sum: f64 = self.bias;

        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }

        let output = activate(sum);
        self.last_output = Some(output);
        output
    }
}

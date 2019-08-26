use crate::neuron::Neuron;

pub struct NeuralNetwork {
    neurons: Vec<Vec<Neuron>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<i32>, learning_constant: f64) -> Self {
        let mut neurons: Vec<Vec<Neuron>> = vec![vec![Neuron::new_pseudo(layers[0])]];
        let mut previous_layer_neurons: i32 = 1;

        for current_layer_neurons in layers {
            let mut layer_neurons: Vec<Neuron> = vec![];

            for _ in 0..current_layer_neurons {
                let neuron = Neuron::new(previous_layer_neurons, learning_constant);
                layer_neurons.push(neuron);
            }

            previous_layer_neurons = layer_neurons.len() as i32;
            neurons.push(layer_neurons);
        }

        Self { neurons }
    }

    pub fn feed_forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut next_inputs: Vec<f64> = inputs;

        for layer_neurons in &self.neurons {
            let mut outputs: Vec<f64> = vec![];

            for neuron in layer_neurons {
                let output = neuron.feed_forward(&next_inputs);
                outputs.push(Self::activate(output));
            }

            next_inputs = outputs;
        }

        next_inputs
    }

    fn activate(val: f64) -> f64 {
        1.0 / (1.0 + (-1.0 * val).exp()) // sigmoid
    }

    pub fn error(desired: Vec<f64>, guess: Vec<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..desired.len() {
            sum += Self::loss(desired[i], gues[i]);
        }

        sum
    }

    fn loss(desired: f64, guess: f64) -> f64 {
        (desired - guess).powi(2) * 0.5
    }
}

// -(desired - guess) * guess * (1- guess) * guess of prev node
// new weight = weight - ^ * learning rate

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let n = NeuralNetwork::new(vec![2, 2, 1], 0.1);
        println!("test: {:?}", n.feed_forward(vec![1.0, 1.0]));
    }
}

use crate::functions::{activation, loss};
use crate::neuron::Neuron;

pub struct NeuralNetwork {
    neurons: Vec<Vec<Neuron>>,
    learning_constant: f64,
}

pub struct TrainingResult {
    guess: Vec<f64>,
    error: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<u32>, learning_constant: f64) -> Self {
        let mut neurons: Vec<Vec<Neuron>> = vec![];

        if layers.contains(&0) {
            panic!("Can't create a neural network with a layer of 0 neurons!");
        }

        layers
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(layer_index, number_of_neurons)| {
                let mut layer_neurons: Vec<Neuron> = vec![];

                for _ in 0..*number_of_neurons {
                    let neuron = Neuron::new(layers[layer_index - 1]);
                    layer_neurons.push(neuron);
                }

                neurons.push(layer_neurons);
            });

        Self {
            neurons,
            learning_constant,
        }
    }

    pub fn load(layers: Vec<Vec<(Vec<f64>, f64)>>, learning_constant: f64) -> Self {
        let loaded_neurons = layers
            .into_iter()
            .map(|neurons| {
                neurons
                    .into_iter()
                    .map(|(weights, bias)| Neuron::load(weights, bias))
                    .collect()
            })
            .collect();

        Self {
            neurons: loaded_neurons,
            learning_constant,
        }
    }

    pub fn guess(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.propagate_forward(inputs)
    }

    pub fn train(&mut self, inputs: &Vec<f64>, expected: &Vec<f64>) -> TrainingResult {
        let guess = self.propagate_forward(inputs);
        self.propagate_backwards(expected);
        self.update_weights(inputs);

        TrainingResult {
            error: Self::error(expected, &guess),
            guess,
        }
    }

    fn propagate_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons
            .iter()
            .fold(inputs.clone(), |next_inputs, layer| {
                layer
                    .iter()
                    .map(|neuron| neuron.propagate_forward(&next_inputs, &activation::sigmoid))
                    .collect()
            })
    }

    fn propagate_backwards(&mut self, expected: &Vec<f64>) {
        let last_output_error_msg = "Cannot propagate backwards without forward propagating first!";

        for layer_index in (0..self.neurons.len()).rev() {
            if layer_index == self.neurons.len() - 1 {
                for (neuron_index, neuron) in self.neurons[layer_index].iter_mut().enumerate() {
                    let output = neuron.last_output.borrow().expect(last_output_error_msg);
                    let error = loss::squared_derived(expected[neuron_index], output);
                    neuron.delta = error * activation::sigmoid_derived(output);
                }
            } else {
                for neuron_index in 0..self.neurons[layer_index].len() {
                    let mut error = 0.0;

                    for prev_layer_neuron in self.neurons[layer_index + 1].iter() {
                        error += prev_layer_neuron.weights[neuron_index] * prev_layer_neuron.delta;
                    }

                    let output = self.neurons[layer_index][neuron_index]
                        .last_output
                        .borrow()
                        .expect(last_output_error_msg);

                    self.neurons[layer_index][neuron_index].delta =
                        error * activation::sigmoid_derived(output);
                }
            }
        }
    }

    fn update_weights(&mut self, inputs: &Vec<f64>) {
        let mut inputs = inputs.clone();

        for layer in self.neurons.iter_mut() {
            let mut next_inputs = vec![];

            for neuron in layer.iter_mut() {
                let learn_delta = self.learning_constant * neuron.delta;

                for (weight_index, weight) in neuron.weights.iter_mut().enumerate() {
                    *weight += learn_delta * inputs[weight_index];
                }

                neuron.bias += learn_delta;
                next_inputs.push(neuron.last_output.borrow().unwrap());
            }

            inputs = next_inputs;
        }
    }

    fn error(expected: &Vec<f64>, guess: &Vec<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..expected.len() {
            sum += loss::squared(expected[i], guess[i]);
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::activation::sigmoid;
    use rand::Rng;

    #[test]
    fn test() {
        let mut n = NeuralNetwork::new(vec![2, 2, 1], 1.0);

        let data = vec![
            (vec![1.0, 1.0], vec![0.0]),
            (vec![-1.0, -1.0], vec![0.0]),
            (vec![1.0, -1.0], vec![1.0]),
            (vec![-1.0, 1.0], vec![1.0]),
        ];

        for _ in 0..2000 {
            let i = rand::thread_rng().gen_range(0, 4);
            let (input, expected) = &data[i];
            let TrainingResult { guess, error } = n.train(input, expected);
            println!(
                "input: {:?}, expect: {:?}, guess: {:?}, error: {}",
                input,
                expected,
                guess.iter().map(|v| v.round()).collect::<Vec<f64>>(),
                error
            );
        }
    }

    #[test]
    fn test_forward_propagate() {
        let load = vec![
            vec![(vec![0.2, 0.4], 0.3), (vec![0.6, 0.8], 0.7)],
            vec![(vec![0.1, 0.5], 1.0)],
        ];
        let n = NeuralNetwork::load(load, 0.1);
        let output = n.guess(&vec![1.0, 0.0]);

        let h1 = sigmoid(1.0 * 0.2 + 0.0 * 0.4 + 0.3);
        let h2 = sigmoid(1.0 * 0.6 + 0.0 * 0.8 + 0.7);
        let o1 = sigmoid(h1 * 0.1 + h2 * 0.5 + 1.0);
        assert_eq!(output[0], o1);
    }
}

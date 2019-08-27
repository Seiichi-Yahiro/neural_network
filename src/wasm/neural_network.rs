use crate::functions::{activation, loss};
use crate::neuron::Neuron;
use std::option::Option;
use std::result::Result;

pub struct NeuralNetwork {
    neurons: Vec<Vec<Neuron>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<u32>, learning_constant: f64) -> Self {
        let mut neurons: Vec<Vec<Neuron>> = vec![];
        let mut previous_layer_neurons: u32 = layers[0];

        layers.iter().for_each(|number_of_neurons| {
            if *number_of_neurons == 0 {
                panic!("Can't create a neural network with a layer of 0 neurons!");
            }
        });

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

        Self { neurons }
    }

    pub fn feed_forward(&self, inputs: Vec<f64>) -> Option<Vec<f64>> {
        match self.feed_forward_memoized(inputs) {
            Some(output) => output.last().map(|last| last.clone()),
            None => None,
        }
    }

    fn feed_forward_memoized(&self, inputs: Vec<f64>) -> Option<Vec<Vec<f64>>> {
        if inputs.len() != self.neurons[0].len() {
            return None;
        }

        let mut result: Vec<Vec<f64>> = vec![];
        let mut next_inputs: Vec<f64> = inputs;

        for (layer_index, layer_neurons) in self.neurons.iter().enumerate() {
            let mut outputs: Vec<f64> = vec![];

            for (neuron_index, neuron) in layer_neurons.iter().enumerate() {
                let output = neuron
                    .feed_forward(&next_inputs, &activation::sigmoid)
                    .unwrap();

                outputs.push(output);
            }

            result.push(outputs.clone());
            next_inputs = outputs;
        }

        Some(result)
    }

    fn error(desired: &Vec<f64>, guess: &Vec<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..desired.len() {
            sum += loss::squared(desired[i], guess[i]);
        }

        sum
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
        println!(
            "test: {:?}",
            n.feed_forward_memoized(vec![1.0, 1.0]).unwrap()
        );
    }
}

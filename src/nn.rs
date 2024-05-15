#![allow(dead_code)]
#![allow(unused_variables)]

use rand::Rng;

pub struct NeuralNetwork {
    layers: Vec<Vec<Neuron>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Vec<Neuron>>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }
    /// zero_layer_size -> The size (or how many neurons) of the first layer (input layer)
    ///
    /// input_size -> The size of the input data
    ///
    /// num_of_hidden_layers -> The number of hidden layers. If 0, then there are no hidden layers.
    ///
    /// size_of_hidden_layers -> The size of each hidden layer
    ///
    /// last_layer_size -> The size of the last layer (output layer). If 0, then there is no output layer.
    pub fn construct(
        zero_layer_size: i32,
        input_size: i32,
        num_of_hidden_layers: i32,
        size_of_hidden_layers: i32,
        last_layer_size: i32,
    ) -> NeuralNetwork {
        let mut network: Vec<Vec<Neuron>> = vec![Self::init_layer(zero_layer_size, input_size)];

        if num_of_hidden_layers > 1 {
            // If more than 1 that means that we have a complex neural network, therefore add hidden layers
            for i in 0..num_of_hidden_layers {
                network.push(Self::init_layer(
                    size_of_hidden_layers,
                    network[i as usize].len() as i32,
                ));
            }
        }

        if last_layer_size != 0 {
            // If we have an output layer, then add it
            network.push(Self::init_layer(
                last_layer_size,
                network.last().unwrap().len() as i32,
            ));
        }

        NeuralNetwork::new(network)
    }

    fn init_layer(layer_size: i32, input_size: i32) -> Vec<Neuron> {
        vec![Neuron::random(input_size as usize); layer_size as usize]
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        println!("{:#?}", self.layers);

        vec![]
    }
}

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    value: f64,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, value: f64) -> Neuron {
        Neuron {
            weights,
            bias,
            value,
        }
    }

    pub fn random(size: usize) -> Neuron {
        let mut weights: Vec<f64> = vec![0.0; size];

        for i in 0..size {
            weights[i] = rand::thread_rng().gen::<f64>();
        }
        Neuron::new(weights, rand::thread_rng().gen::<f64>(), 0.0)
    }

    pub fn activate(&self) -> f64 {
        Self::sigmoid(self.value)
    }

    pub fn modify_neuron(&mut self, weights: Vec<f64>, bias: f64, value: f64) {
        self.weights = weights;
        self.bias = bias;
        self.value = value;
    }

    fn relu(val: f64) -> f64 {
        if val > 0.0 {
            val
        } else {
            0.0
        }
    }

    fn sigmoid(val: f64) -> f64 {
        1.0 / (1.0 + (-val).exp())
    }

    fn tanh(val: f64) -> f64 {
        val.tanh()
    }

    fn softmax(val: f64) -> f64 {
        let exp: f64 = (-val).exp();
        exp / (exp + 1.0)
    }
}

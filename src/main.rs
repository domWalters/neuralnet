use std::time::{Instant, Duration};

use network::NeuralNetwork;
use vektor::Vektor;
use tdata::TData;

pub mod network;
pub mod vektor;
pub mod matrix;
pub mod tdata;

fn main() {

    let batch_size = 20;
    let epsilon = 0.0001;
    let epochs = 1000;
    let training_data = TData::new_from_funct(1000, 5, 0.0, 10.0, Box::new(| &x | 2.0 * x + 1.0));

    let pattern = [5, 5, 5];
    let count = 10;
    let mut total_1 = Duration::new(0, 0);
    let mut total_2 = Duration::new(0, 0);
    for i in 0..count {
        let mut nn_1 = NeuralNetwork::new_random(&pattern, NeuralNetwork::leaky_relu(), NeuralNetwork::leaky_relu_diff());
        let mut nn_2 = NeuralNetwork::new_random(&pattern, NeuralNetwork::leaky_relu(), NeuralNetwork::leaky_relu_diff());
        let now = Instant::now();
        nn_1.gd(&training_data, batch_size, epsilon, epochs);
        total_1 = total_1.checked_add(now.elapsed()).expect("Overflow");
        println!("i={} nn_1 completed.", i);
        let now = Instant::now();
        nn_2.gd_w_m(&training_data, batch_size, epsilon, epochs);
        total_2 = total_2.checked_add(now.elapsed()).expect("Overflow");
        println!("i={} nn_2 completed.", i);
    }
    println!("GD: {:?}", total_1.checked_div(count));
    println!("GD W M: {:?}", total_2.checked_div(count));

    //test(&mut nn_1);
}

fn test(nn: &mut NeuralNetwork) {
    nn.feedforward(&Vektor { v: vec![1.0, 2.0, 3.0, 4.0, 5.0] });
    println!("{:?}, {:?}", Vektor { v: vec![1.0, 2.0, 3.0, 4.0, 5.0] }, nn.extract_outputs());
    nn.feedforward(&Vektor { v: vec![5.0, 4.0, 3.0, 2.0, 1.0] });
    println!("{:?}, {:?}", Vektor { v: vec![5.0, 4.0, 3.0, 2.0, 1.0] }, nn.extract_outputs());
    nn.feedforward(&Vektor { v: vec![1.0, 1.0, 1.0, 1.0, 1.0] });
    println!("{:?}, {:?}", Vektor { v: vec![1.0, 1.0, 1.0, 1.0, 1.0] }, nn.extract_outputs());
    nn.feedforward(&Vektor { v: vec![1.0, 6.0, 3.0, 8.0, 5.0] });
    println!("{:?}, {:?}", Vektor { v: vec![1.0, 6.0, 3.0, 8.0, 5.0] }, nn.extract_outputs());
    nn.feedforward(&Vektor { v: vec![9.0, 4.0, 7.0, 2.0, 5.0] });
    println!("{:?}, {:?}", Vektor { v: vec![9.0, 4.0, 7.0, 2.0, 5.0] }, nn.extract_outputs());
}

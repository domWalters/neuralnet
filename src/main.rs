use network::NeuralNetwork;
use vektor::Vektor;
use tdata::TData;

pub mod network;
pub mod vektor;
pub mod matrix;
pub mod tdata;

fn main() {

    let pattern = [5, 5, 5];
    let mut nn = NeuralNetwork::new_random(&pattern, NeuralNetwork::leaky_relu(), NeuralNetwork::leaky_relu_diff());

    let batch_size = 20;
    let epsilon = 0.0001;
    let epochs = 1000;
    //let training_data = TData::new(1000, 5, 0.0, 10.0);
    let training_data = TData::new_from_funct(1000, 5, 0.0, 10.0, Box::new(| &x | 2.0 * x + 1.0));

    nn.sgd(&training_data, batch_size, epsilon, epochs);

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

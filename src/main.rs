extern crate rand;

use rand::Rng;
use rand::thread_rng;

use network::NeuralNetwork;
use matrix::Matrix;
use vektor::Vektor;
use tdata::TData;

pub mod network;
pub mod vektor;
pub mod matrix;
pub mod tdata;

fn main() {

    let pattern = [5, 5, 5];
    let mut nn = NeuralNetwork::new_random(&pattern);

    let batch_size = 20;
    let epsilon = 0.0001;
    let epochs = 1000;
    let training_data = TData::new(1000, 5, 0.0, 10.0);

    sgd(&mut nn, &training_data, batch_size, epsilon, epochs);
    nn.feedforward(&training_data[0].0);
    println!("{:?}, {:?}, {:?}", training_data[0].0, training_data[0].1, nn.extract_outputs());
}

fn aggregate_square_distance(x: &Vektor, y: &Vektor) -> f64 {
    let difference = x.sub(&y);
    let square = difference.map(|x| x*x);
    square.sum() / (x.len() as f64)
}

fn sgd(nn: &mut NeuralNetwork,
       training_data: &Vec<(Vektor, Vektor)>,
       batch_size: usize,
       epsilon: f64,
       epochs: u64) {
    let n = training_data.len();
    let num_batches = n / batch_size;
    for t in 0..epochs {
        println!("Starting epoch {}", t);
        let mut permutation: Vec<usize> = (0..n).collect();
        thread_rng().shuffle(&mut permutation);
        for k in 0..num_batches {
            //println!("Starting batch {}", k);
            let mut del_w = Vec::new();
            let mut del_b = Vec::new();
            for l in 0..nn.a.len() {
                if l > 0 {
                    del_w.push(Matrix::new(nn.w[l].m.len(), nn.w[l - 1].m.len()));
                } else {
                    del_w.push(Matrix::new(nn.w[l].m.len(), nn.w[l].m.len()));
                }
                del_b.push(Vektor::new(nn.a[l].len()));
            }
            let mut batch_loss = 0.0;
            for i in 0..batch_size {
                // Select Data Point
                let (ref x, ref y) = training_data[permutation[k * batch_size + i]];
                // Forward
                nn.feedforward(x);
                // Backward
                let (dc_dw, dc_db) = nn.backpropogate(y);
                // Accumulate Delta
                for l in 1..nn.a.len() {
                    del_w[l] = del_w[l].add(&dc_dw[l]);
                    del_b[l] = del_b[l].add(&dc_db[l]);
                }
                batch_loss += aggregate_square_distance(&nn.a[nn.a.len() - 1], y);
            }
            if k == num_batches - 1 {
                println!("Average Batch Loss Per Element = {}", batch_loss / (batch_size as f64));
            }
            for l in 1..nn.a.len() {
                nn.w[l] = nn.w[l].add(&del_w[l].scalar_mult(-(epsilon / (batch_size as f64))));
                nn.b[l] = nn.b[l].add(&del_b[l].scalar_mult(-(epsilon / (batch_size as f64))));
            }
        }
    }
}

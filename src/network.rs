extern crate rand;

use self::rand::Rng;
use self::rand::thread_rng;

use matrix::Matrix;
use vektor::Vektor;

pub struct NeuralNetwork {
    pub w: Vec<Matrix>, // layer l, node j, weight from (l-1)(k) to (l)(j)
    pub b: Vec<Vektor>, // layer l, node j
    pub a: Vec<Vektor>, // layer l, node j
    pub z: Vec<Vektor>, // layer l, node j
    pub act_funct: Box<Fn(&f64) -> f64>,
    pub act_funct_diff: Box<Fn(&f64) -> f64>,
}

impl NeuralNetwork {

    pub fn new_blank(act_funct: Box<Fn(&f64) -> f64>, act_funct_diff: Box<Fn(&f64) -> f64>) -> NeuralNetwork {
        NeuralNetwork {
            w: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
            z: Vec::new(),
            act_funct: act_funct,
            act_funct_diff: act_funct_diff,
        }
    }

    pub fn new_random(pattern: &[usize], act_funct: Box<Fn(&f64) -> f64>, act_funct_diff: Box<Fn(&f64) -> f64>) -> NeuralNetwork {
        let mut nn = NeuralNetwork::new_blank(act_funct, act_funct_diff);
        for l in 0..pattern.len() {
            let mut weights;
            if l == 0 {
                weights = Matrix::new(pattern[l], pattern[l]);
            } else {
                weights = Matrix::new(pattern[l], pattern[l - 1]);
            }
            let mut biases = Vektor::new(pattern[l]);
            for j in 0..pattern[l] {
                if l > 0 {
                    let mut weights_into_node: Vec<f64> = Vec::new();
                    for _ in 0..pattern[l-1] {
                        weights_into_node.push(thread_rng().gen_range(0.0, 1.0));
                    }
                    weights.m[j] = Vektor::new_from_vec(weights_into_node);         // strictly the first layer doesn't have one, but we need it for indexing
                }
                biases.v[j] = thread_rng().gen_range(0.0, 1.0);                     // strictly the first layer doesn't have one, but we need it for indexing
            }
            nn.w.push(weights);
            nn.b.push(biases);
            nn.a.push(Vektor::new(pattern[l]));
            nn.z.push(Vektor::new(pattern[l]));
        }
        nn
    }

    fn set_inputs(&mut self, inputs: &Vektor) {
        self.a[0] = inputs.clone();
    }

    pub fn extract_outputs(&self) -> &Vektor {
        &self.a[self.a.len() - 1]
    }

    pub fn feedforward(&mut self, inputs: &Vektor) {
        self.set_inputs(inputs);
        for l in 1..self.a.len() {
            self.z[l] = self.w[l].vec_mult(&self.a[l-1])
                                 .add(&self.b[l]);
            self.a[l] = self.z[l].map(&*self.act_funct);
        }
    }

    pub fn backpropogate(&mut self, y: &Vektor) -> (Vec<Matrix>, Vec<Vektor>) { // This is highly abstracted and inefficient. Still O(n), but about 5 times more data loops than it should have.
        let mut dc_dw: Vec<Matrix> = Vec::new();
        let mut dc_db: Vec<Vektor> = Vec::new();

        for l in 0..self.a.len() {
            dc_db.push(Vektor::new(self.a[l].len()));
            if l > 0 {
                dc_dw.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
            } else {
                dc_dw.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
            }
        }

        for l in (1..self.a.len()).rev() {
            if l < self.a.len() - 1 {    // Local gradient for other layers
                dc_db[l] = self.w[l+1].transpose().vec_mult(&dc_db[l+1]).hadamaud_prod(&self.z[l].map(&*self.act_funct_diff));
            } else {                    // Local gradient for output
                dc_db[l] = self.z[l].map(&*self.act_funct_diff).hadamaud_prod(&self.a[l].sub(&y));  //hard coded dc/da[j][L]
            }
            dc_dw[l] = dc_db[l].mult(&self.a[l - 1]);
        }
        (dc_dw, dc_db)
    }

    pub fn sgd(&mut self,
           training_data: &Vec<(Vektor, Vektor)>,
           batch_size: usize,
           epsilon: f64,
           epochs: u64) {
        let n = training_data.len();
        let num_batches = n / batch_size;
        //let alpha = 0.5; // 0.9, 0.99
        let aggregate_square_distance = |x: &Vektor, y: &Vektor| x.sub(y).map(|x| x*x).sum() / (x.len() as f64);
        for t in 0..epochs {
            println!("Starting epoch {}", t);
            let mut permutation: Vec<usize> = (0..n).collect();
            thread_rng().shuffle(&mut permutation);
            for k in 0..num_batches {
                //println!("Starting batch {}", k);
                let mut del_w = Vec::new();
                //let mut w_vel = Vec::new();
                let mut del_b = Vec::new();
                //let mut b_vel = Vec::new();
                for l in 0..self.a.len() {
                    if l > 0 {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
                        //w_vel.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
                    } else {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
                        //w_vel.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
                    }
                    del_b.push(Vektor::new(self.a[l].len()));
                    //b_vel.push(Vektor::new(self.a[l].len()));
                }
                let mut batch_loss = 0.0;
                for i in 0..batch_size {
                    // Select Data Point
                    let (ref x, ref y) = training_data[permutation[k * batch_size + i]];
                    // Forward
                    self.feedforward(x);
                    // Backward
                    let (dc_dw, dc_db) = self.backpropogate(y);
                    // Accumulate Delta
                    for l in 1..self.a.len() {
                        del_w[l] = del_w[l].add(&dc_dw[l]);
                        del_b[l] = del_b[l].add(&dc_db[l]);
                    }
                    batch_loss += aggregate_square_distance(&self.a[self.a.len() - 1], &y);
                }
                if k == num_batches - 1 {
                    println!("Average Batch Loss Per Element = {}", batch_loss / (batch_size as f64));
                }
                for l in 1..self.a.len() {    // Grad Descent (Pleb Version)
                    self.w[l] = self.w[l].add(&del_w[l].scalar_mult(-(epsilon / (batch_size as f64))));
                    self.b[l] = self.b[l].add(&del_b[l].scalar_mult(-(epsilon / (batch_size as f64))));
                }
                // for l in 1..self.a.len() {    // Grad Descent (With Momentum)
                //     w_vel[l] = w_vel[l].scalar_mult(alpha).add(&del_w[l].scalar_mult(-(epsilon / (batch_size as f64))));
                //     self.w[l] = self.w[l].add(&w_vel[l]);
                //     b_vel[l] = b_vel[l].scalar_mult(alpha).add(&del_b[l].scalar_mult(-(epsilon / (batch_size as f64))));
                //     self.b[l] = self.b[l].add(&b_vel[l]);
                // }
            }
        }
    }

    pub fn relu() -> Box<Fn(&f64) -> f64> {
        Box::new(| x: &f64 | {
            (*x).max(0.0)
        })
    }

    pub fn relu_diff() -> Box<Fn(&f64) -> f64> {
        Box::new(| x: &f64 | {
            if *x < 0.0 {
                0.0
            } else {
                1.0
            }
        })
    }

    pub fn leaky_relu() -> Box<Fn(&f64) -> f64> {
        Box::new(| x: &f64 | {
            (*x).max(0.01 * *x)
        })
    }

    pub fn leaky_relu_diff() -> Box<Fn(&f64) -> f64> {
        Box::new(| x: &f64 | {
            if *x < 0.0 {
                0.01
            } else {
                1.0
            }
        })
    }

}

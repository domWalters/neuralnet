extern crate rand;

use rand::Rng;
use rand::thread_rng;

use matrix::Matrix;
use vektor::Vektor;

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub w: Vec<Matrix>, // layer l, node j, weight from (l-1)(k) to (l)(j)
    pub b: Vec<Vektor>, // layer l, node j
    pub a: Vec<Vektor>, // layer l, node j
    pub z: Vec<Vektor>, // layer l, node j
}

impl NeuralNetwork {

    pub fn new_blank() -> NeuralNetwork {
        NeuralNetwork {
            w: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
            z: Vec::new(),
        }
    }

    pub fn new_random(pattern: &[usize]) -> NeuralNetwork {
        let mut nn = NeuralNetwork::new_blank();
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

    pub fn set_inputs(&mut self, inputs: &Vektor) {
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
            self.a[l] = self.z[l].map(| x | {
                if *x < 0.0 {
                    0.0
                } else {
                    *x
                }
            });
        }
    }

    pub fn backpropogate(&mut self, y: &Vektor) -> (Vec<Matrix>, Vec<Vektor>) { // This is highly abstracted and inefficient. Still O(n), but about 5 times more data loops than it should have.
        let mut dc_dw: Vec<Matrix> = Vec::new();
        let mut dc_db: Vec<Vektor> = Vec::new();

        let relu_diff = | x: &f64 | {
            if *x < 0.0 {
                0.0
            } else {
                1.0
            }
        };

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
                dc_db[l] = self.w[l+1].transpose().vec_mult(&dc_db[l+1]).hadamaud_prod(&self.z[l].map(relu_diff));
            } else {                    // Local gradient for output
                dc_db[l] = self.z[l].map(relu_diff).hadamaud_prod(&self.a[l].sub(&y));  //hard coded dc/da[j][L]
            }
            dc_dw[l] = dc_db[l].mult(&self.a[l - 1]);
        }
        (dc_dw, dc_db)
    }

}

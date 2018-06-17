extern crate rand;

use self::rand::Rng;
use self::rand::thread_rng;

use std::error::Error;
use std::io::BufReader;
use std::io::prelude::*;
use std::io;
use std::fs::File;
use std::path::Path;

use matrix::Matrix;
use vektor::Vektor;

pub struct NeuralNetwork {
    w: Vec<Matrix>, // layer l, node j, weight from (l-1)(k) to (l)(j)
    b: Vec<Vektor>, // layer l, node j
    a: Vec<Vektor>, // layer l, node j
    z: Vec<Vektor>, // layer l, node j
    act_funct: Box<Fn(&f64) -> f64>,
    act_funct_diff: Box<Fn(&f64) -> f64>,
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
            if l == 0 {
                nn.w.push(Matrix::new_random_f64(pattern[l], pattern[l], 0.0, 1.0));        // This won't ever be accessed. The zero-th layer doesn't have ws, but we need it for indexing
            } else {
                nn.w.push(Matrix::new_random_f64(pattern[l], pattern[l - 1], 0.0, 1.0));
            }
            nn.b.push(Vektor::new_random_f64(pattern[l], 0.0, 1.0));                        // strictly the first layer doesn't have one, but we need it for indexing
            nn.a.push(Vektor::new(pattern[l]));                                             // initialised to all zeroes
            nn.z.push(Vektor::new(pattern[l]));                                             // initialised to all zeroes
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

    pub fn gd(&mut self,
           training_data: &Vec<(Vektor, Vektor)>,
           batch_size: usize,
           epsilon: f64,
           epochs: u64) {
        let num_batches = training_data.len() / batch_size;
        for t in 0..epochs {
            // println!("Starting epoch {}", t);
            let mut permutation: Vec<usize> = (0..training_data.len()).collect();
            thread_rng().shuffle(&mut permutation);
            for k in 0..num_batches {
                let mut del_w = Vec::new();
                let mut del_b = Vec::new();
                for l in 0..self.a.len() {
                    if l > 0 {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
                    } else {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
                    }
                    del_b.push(Vektor::new(self.a[l].len()));
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
                    batch_loss += NeuralNetwork::aggregate_square_distance(&self.a[self.a.len() - 1], &y);
                }
                // if k == num_batches - 1 {
                //     println!("Average Batch Loss Per Element = {}", batch_loss / (batch_size as f64));
                // }
                for l in 1..self.a.len() {
                    self.w[l] = self.w[l].add(&del_w[l].scalar_mult(-(epsilon / (batch_size as f64))));
                    self.b[l] = self.b[l].add(&del_b[l].scalar_mult(-(epsilon / (batch_size as f64))));
                }
                if batch_loss / (batch_size as f64) < 0.01 {
                    return;
                }
            }
        }
    }

    pub fn gd_w_m(&mut self,
           training_data: &Vec<(Vektor, Vektor)>,
           batch_size: usize,
           epsilon: f64,
           epochs: u64) {
        let num_batches = training_data.len() / batch_size;
        let alpha = 0.5; // 0.9, 0.99
        for t in 0..epochs {
            // println!("Starting epoch {}", t);
            let mut permutation: Vec<usize> = (0..training_data.len()).collect();
            thread_rng().shuffle(&mut permutation);
            let mut w_vel = Vec::new();
            let mut b_vel = Vec::new();
            for l in 0..self.a.len() {
                if l > 0 {
                    w_vel.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
                } else {
                    w_vel.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
                }
                b_vel.push(Vektor::new(self.a[l].len()));
            }
            for k in 0..num_batches {
                let mut del_w = Vec::new();
                let mut del_b = Vec::new();
                for l in 0..self.a.len() {
                    if l > 0 {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l - 1].m.len()));
                    } else {
                        del_w.push(Matrix::new(self.w[l].m.len(), self.w[l].m.len()));
                    }
                    del_b.push(Vektor::new(self.a[l].len()));
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
                    batch_loss += NeuralNetwork::aggregate_square_distance(&self.a[self.a.len() - 1], &y);
                }
                if k == num_batches - 1 {
                    println!("Average Batch Loss Per Element = {}", batch_loss / (batch_size as f64));
                }
                for l in 1..self.a.len() {
                    w_vel[l] = w_vel[l].scalar_mult(alpha).add(&del_w[l].scalar_mult(-(epsilon / (batch_size as f64))));
                    self.w[l] = self.w[l].add(&w_vel[l]);
                    b_vel[l] = b_vel[l].scalar_mult(alpha).add(&del_b[l].scalar_mult(-(epsilon / (batch_size as f64))));
                    self.b[l] = self.b[l].add(&b_vel[l]);
                }
                if batch_loss / (batch_size as f64) < 0.01 {
                    return;
                }
            }
        }
    }

    fn aggregate_square_distance(x: &Vektor, y: &Vektor) -> f64 {
        x.sub(y).map(|x| x * x).sum() / (x.len() as f64)
    }

    pub fn save_all(&self, file_name: &str) {
        // Get file to save to
        let path = Path::new(file_name);
        let display = path.display();
        let mut file = match File::create(&path) {
            Err(why) => panic!("Couldn't create file {}: {}", display, why.description()),
            Ok(file) => file,
        };
        // Save self.w
        for l in 0..self.w.len() {
            match NeuralNetwork::save_entity(&mut file, &self.w[l].save_format(), &NeuralNetwork::entity_preamble("w", l, self.w[l].m.len())) {
                Err(why) => {
                    panic!("Couldn't write to {}: {}", file_name, why.description())
                },
                Ok(_) => println!("Successfully wrote to {}", file_name),
            }
        }
        // Save self.b
        for l in 0..self.b.len() {
            let mut save_data = self.b[l].save_format();
            save_data.push_str("\n");
            match NeuralNetwork::save_entity(&mut file, &save_data, &NeuralNetwork::entity_preamble("b", l, 1)) {
                Err(why) => {
                    panic!("Couldn't write to {}: {}", file_name, why.description())
                },
                Ok(_) => println!("Successfully wrote to {}", file_name),
            }
        }
    }

    fn save_entity(file: &mut File, save_data: &str, name: &str) -> io::Result<()> {
        file.write_all(name.as_bytes())?;
        file.write_all(save_data.as_bytes())?;
        Ok(())
    }

    fn entity_preamble(name: &str, index: usize, size: usize) -> String {
        let mut preamble = String::new();
        preamble.push_str(name);
        preamble.push_str("[");
        preamble.push_str(&index.to_string());
        preamble.push_str("] ");
        preamble.push_str(&size.to_string());
        preamble.push_str("\n");
        preamble
    }

    pub fn load(file_name: &str, act_funct: Box<Fn(&f64) -> f64>, act_funct_diff: Box<Fn(&f64) -> f64>) -> NeuralNetwork {
        // Get a blank network
        let mut nn = NeuralNetwork::new_blank(act_funct, act_funct_diff);
        // Get file to read from
        let path = Path::new(file_name);
        let display = path.display();
        let file = match File::open(&path) {
            Err(why) => panic!("Couldn't open file {}: {}", display, why.description()),
            Ok(file) => file,
        };
        // Setup data stores
        let mut buf_reader = BufReader::new(file);
        let mut entity = String::new();
        let mut preamble = String::new();
        // Read initial preamble
        buf_reader.read_line(&mut preamble).expect("Couldn't read line (preamble ended early).");
        // Read W
        while preamble.contains("w") {
            let l = preamble[5..preamble.len() - 1].parse::<usize>().expect("Preamble is in the wrong format.");
            for _ in 0..l {
                buf_reader.read_line(&mut entity).expect("Couldn't read line.");
            }
            nn.w.push(Matrix::load(&entity));
            preamble.clear();
            entity.clear();
            buf_reader.read_line(&mut preamble).expect("Couldn't read line (preamble ended early).");
        }
        let mut b_count = 1;
        // Read B
        while preamble.contains("b") {
            buf_reader.read_line(&mut entity).expect("Couldn't read line.");
            nn.b.push(Vektor::load(&entity));
            nn.a.push(Vektor::new(nn.b[b_count - 1].len())); // initialised to all zeroes
            nn.z.push(Vektor::new(nn.b[b_count - 1].len())); // initialised to all zeroes
            preamble.clear();
            entity.clear();
            if b_count < nn.w.len() {
                buf_reader.read_line(&mut preamble).expect("Couldn't read line (preamble ended early).");
                b_count += 1;
            }
        }
        nn
    }

    pub fn dimension_printer(&self) {
        if self.w.len() == self.b.len() {
            println!("W and B have same layer count.");
            if self.w.len() == self.a.len() {
                println!("W and A have same layer count.");
                if self.w.len() == self.z.len() {
                    println!("W and Z have same layer count.");
                }
            }
        }

        // B dimensions
        let mut dim = Vec::new();
        for i in 0..self.b.len() {
            dim.push(self.b[i].len());
        }
        println!("B dimensions: {:?}", dim);

        // A dimensions
        let mut dim = Vec::new();
        for i in 0..self.a.len() {
            dim.push(self.a[i].len());
        }
        println!("A dimensions: {:?}", dim);

        // Z dimensions
        let mut dim = Vec::new();
        for i in 0..self.z.len() {
            dim.push(self.z[i].len());
        }
        println!("Z dimensions: {:?}", dim);

        // W dimensions
        let mut dim = Vec::new();
        for i in 0..self.w.len() {
            dim.push(self.w[i].dimension_check());
        }
        println!("W dimensions: {:?}", dim);

    }

}

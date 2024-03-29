extern crate rand;

use self::rand::Rng;
use self::rand::thread_rng;

use std::error::Error;

use matrix::Matrix;

#[derive(Clone, Debug)]
pub struct Vektor {
    pub v: Vec<f64>,
}

impl Vektor {

    pub fn new(n: usize) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for _ in 0..n {
            result.v.push(0.0);
        }
        result
    }

    pub fn new_from_vec(vec: Vec<f64>) -> Vektor {
        Vektor {
            v: vec,
        }
    }

    pub fn new_random_f64(n: usize, lb: f64, ub: f64) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for _ in 0..n {
            result.v.push(thread_rng().gen_range(lb, ub));
        }
        result
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn reverse(&self) -> Vektor {
        let mut new_v = self.v.clone();
        new_v.reverse();
        Vektor {
            v: new_v,
        }
    }

    pub fn dot_prod(&self, other: &Vektor) -> f64 {
        let mut aggregate = 0.0;
        for i in 0..self.v.len() {
            aggregate += self.v[i] * other.v[i];
        }
        aggregate
    }

    pub fn hadamaud_prod(&self, other: &Vektor) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for i in 0..self.v.len() {
            result.v.push(self.v[i] * other.v[i]);
        }
        result
    }

    pub fn add(&self, other: &Vektor) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for i in 0..self.v.len() {
            result.v.push(self.v[i] + other.v[i]);
        }
        result
    }

    pub fn sub(&self, other: &Vektor) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for i in 0..self.v.len() {
            result.v.push(self.v[i] - other.v[i]);
        }
        result
    }

    pub fn scalar_mult(&self, scalar: f64) -> Vektor {
        self.map(|x| scalar * x)
    }

    pub fn sum(&self) -> f64 {
        self.v.iter().fold(0.0, |total, next| total + next)
    }

    pub fn map<F>(&self, f: F) -> Vektor where F: FnMut(&f64) -> f64 {
        Vektor {
            v: self.v.iter().map(f).collect(),
        }
    }

    pub fn mult(&self, other: &Vektor) -> Matrix {
        let mut result = Matrix {
            m: Vec::new(),
        };
        for j in 0..self.v.len() {
            result.m.push(other.scalar_mult(self.v[j]));
        }
        result
    }

    pub fn save_format(&self) -> String {
        let mut result = String::new();
        result.push_str(&self.v[0].to_string());
        for i in 1..self.v.len() {
            result.push_str("\t");
            result.push_str(&self.v[i].to_string());
        }
        result
    }

    pub fn load(load_data: &str) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        let elements = load_data.split("\t");
        for e in elements {
            match e.trim().parse::<f64>() {
                Err(why) => panic!("Couldn't read from {}: {}", load_data, why.description()),
                Ok(num) => result.v.push(num),
            }
        }
        result
    }

}

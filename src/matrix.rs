use vektor::Vektor;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub m: Vec<Vektor>, // Collection of Row Vectors. m[i] is the ith row.
}

impl Matrix {

    pub fn new(n: usize, m: usize) -> Matrix {
        let mut result = Matrix {
            m: Vec::new(),
        };
        for _ in 0..n {
            result.m.push(Vektor::new(m));
        }
        result
    }

    pub fn new_random_f64(n: usize, m: usize, lb: f64, ub: f64) -> Matrix {
        let mut result = Matrix {
            m: Vec::new(),
        };
        for _ in 0..n {
            result.m.push(Vektor::new_random_f64(m, lb, ub));
        }
        result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix {
            m: Vec::new(),
        };
        for i in 0..self.m.len() {
            result.m.push(self.m[i].add(&other.m[i]));
        }
        result
    }

    pub fn scalar_mult(&self, scalar: f64) -> Matrix {
        self.map(|x| x.scalar_mult(scalar))
    }

    pub fn hadamaud_prod(&self, other: &Matrix) -> Matrix {
        let mut matrix = Matrix {
            m: Vec::new(),
        };
        for i in 0..other.m.len() {
            matrix.m.push(self.m[i].hadamaud_prod(&other.m[i]));
        }
        matrix
    }

    pub fn vec_mult(&self, other: &Vektor) -> Vektor {
        let mut result = Vektor {
            v: Vec::new(),
        };
        for i in 0..other.len() {
            result.v.push(self.m[i].hadamaud_prod(other).sum());
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut mat = Matrix {
            m: Vec::new(),
        };
        for i in 0..self.m.len() {
            let mut vek = Vektor {
                v: Vec::new(),
            };
            for j in 0..self.m[i].len() {
                vek.v.push(self.m[j].v[i]);
            }
            mat.m.push(vek);
        }
        mat
    }

    pub fn map<F>(&self, f: F) -> Matrix where F: Fn(&Vektor) -> Vektor {
        let mut matrix = Matrix {
            m: Vec::new(),
        };
        for i in 0..self.m.len() {
            matrix.m.push(f(&self.m[i]));
        }
        matrix
    }

    pub fn save_format(&self) -> String {
        let mut result = String::new();
        result.push_str(&self.m[0].save_format());
        result.push_str("\n");
        for i in 1..self.m.len() {
            result.push_str(&self.m[i].save_format());
            result.push_str("\n");
        }
        result
    }

}

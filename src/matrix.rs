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

    pub fn add(&mut self, other: &Matrix) -> Matrix {
        let mut result = Matrix {
            m: Vec::new(),
        };
        for i in 0..self.m.len() {
            result.m.push(self.m[i].add(&other.m[i]));
        }
        result
    }

    pub fn scalar_mult(&mut self, scalar: f64) -> Matrix {
        Matrix {
            m: self.m.iter().map(|x| x.scalar_mult(scalar)).collect(),
        }
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

}

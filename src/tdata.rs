use vektor::Vektor;

#[derive(Clone, Debug)]
pub struct TData {
    pub data: Vec<(Vektor, Vektor)>,
}

impl TData {

    pub fn new(n: usize, depth: usize, lb: f64, ub: f64) -> Vec<(Vektor, Vektor)> {
        let mut result = Vec::new();
        for _ in 0..n {
            let mut x = Vektor::new_random_f64(depth, lb, ub);
            let mut y = x.clone();
            y = y.reverse();
            result.push((x, y));
        }
        result
    }

}

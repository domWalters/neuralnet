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

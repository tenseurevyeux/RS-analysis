pub const DEFAULT_EPSILON: f64 = 1e-5;
const MAX_WEIGHTS: usize = 10_000;

pub fn get_weights(d: f64, epsilon: f64) -> Vec<f64> {
    let mut w = vec![1.0];
    let mut k = 1;

    loop {
        let w_k = -w[k - 1] * (d - (k as f64 - 1.0)) / (k as f64);
        if w_k.abs() <= epsilon || k >= MAX_WEIGHTS {
            break;
        }
        w.push(w_k);
        k += 1;
    }

    w
}

pub fn get_inv_weight(d: f64, epsilon: f64) -> Vec<f64> {
    let mut w = vec![1.0];
    let mut k = 1;
    loop {
        let w_k = w[k - 1] * (d + (k as f64 - 1.0)) / (k as f64);
        if w_k.abs() <= epsilon || k >= MAX_WEIGHTS {
            break;
        }
        w.push(w_k);
        k += 1;
    }

    w
}

pub fn frac_diff(series: &[f64], d: f64, epsilon: f64) -> Vec<f64> {
    if series.is_empty() {
        return vec![];
    }
    let weights = get_weights(d, epsilon);
    let max_k = (weights.len() - 1).min(series.len() - 1);
    let mut result = Vec::with_capacity(series.len());

    for t in 0..series.len() {
        let mut sum = 0.0;
        let limit = t.min(max_k);
        for j in 0..=limit {
            sum += weights[j] * series[t - j];
        }
        result.push(sum);
    }
    result
}

pub fn reverse_frac_diff(series: &[f64], d: f64, epsilon: f64) -> Vec<f64> {
    if series.is_empty() {
        return vec![];
    }
    let weights = get_weights(d, epsilon);
    let mut result: Vec<f64> = Vec::with_capacity(series.len());

    for t in 0..series.len() {
        let limit = t.min(weights.len() - 1);
        let history: f64 = (1..=limit).map(|j| weights[j] * result[t - j]).sum();
        result.push(series[t] - history);
    }
    result
}

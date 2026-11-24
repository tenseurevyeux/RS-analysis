use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptySeries,
    ParamLengthMismatch { expected: usize, got: usize },
    LengthMismatch { left: usize, right: usize },
    NonFiniteInput { index: usize },
    InvalidParameter { name: &'static str, message: String },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeries => write!(f, "input series is empty"),
            Self::ParamLengthMismatch { expected, got } => write!(
                f,
                "parameter vector too short: expected at least {expected} values, got {got}"
            ),
            Self::LengthMismatch { left, right } => {
                write!(f, "length mismatch: {left} vs {right}")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "input contains a non-finite value at index {index}")
            }
            Self::InvalidParameter { name, message } => {
                write!(f, "invalid parameter `{name}`: {message}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

pub fn mean_squared_error(y_real: &[f64], y_pred: &[f64]) -> Result<f64, ModelError> {
    if y_real.len() != y_pred.len() {
        return Err(ModelError::LengthMismatch {
            left: y_real.len(),
            right: y_pred.len(),
        });
    }
    if y_real.is_empty() {
        return Err(ModelError::EmptySeries);
    }

    let n = y_real.len() as f64;
    let mse = y_real
        .iter()
        .zip(y_pred)
        .map(|(r, p)| {
            let diff = r - p;
            diff * diff
        })
        .sum::<f64>();
    Ok(mse / n)
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn var(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mu = mean(x);
    x.iter()
        .map(|&x| {
            let diff = x - mu;
            diff * diff
        })
        .sum::<f64>()
        / x.len() as f64
}

pub fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut res = x.to_vec();
    res.sort_by(|a, b| a.total_cmp(b));
    let n = res.len();
    if n % 2 == 0 {
        let mid = n / 2;
        (res[mid - 1] + res[mid]) / 2.0
    } else {
        res[n / 2]
    }
}

pub fn autocov(x: &[f64], nlags: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let max_lag = nlags.min(n - 1);
    let mean = mean(x);
    let mut gamma = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let cov = (0..(n - lag))
            .map(|i| (x[i] - mean) * (x[lag + i] - mean))
            .sum::<f64>()
            / n as f64;
        gamma.push(cov);
    }
    gamma
}

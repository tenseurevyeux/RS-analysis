use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentOpt;
use std::fmt;
use utils::var;

const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum BoxCoxError {
    NonPositiveInput {
        index: usize,
        value: f64,
    },
    InvalidLambda(f64),
    OutOfDomain {
        index: usize,
        value: f64,
        lambda: f64,
    },
}

impl fmt::Display for BoxCoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveInput { index, value } => write!(
                f,
                "Box-Cox requires positive finite data, got {value} at index {index}"
            ),
            Self::InvalidLambda(l) => write!(f, "Box-Cox lambda must be finite, got {l}"),
            Self::OutOfDomain {
                index,
                value,
                lambda,
            } => write!(
                f,
                "value {value} at index {index} is outside the inverse Box-Cox domain for lambda = {lambda}"
            ),
        }
    }
}

impl std::error::Error for BoxCoxError {}

fn checked_logs(data: &[f64]) -> Result<Vec<f64>, BoxCoxError> {
    data.iter()
        .enumerate()
        .map(|(index, &value)| {
            if value.is_finite() && value > 0.0 {
                Ok(value.ln())
            } else {
                Err(BoxCoxError::NonPositiveInput { index, value })
            }
        })
        .collect()
}

fn transform_logs(logs: &[f64], lambda: f64) -> Vec<f64> {
    if lambda.abs() <= EPSILON {
        logs.to_vec()
    } else {
        logs.iter()
            .map(|&l| (lambda * l).exp_m1() / lambda)
            .collect()
    }
}

#[derive(Clone)]
struct BoxCoxMLE {
    log_data: Vec<f64>,
}

impl BoxCoxMLE {
    fn new(data: &[f64]) -> Result<Self, BoxCoxError> {
        Ok(Self {
            log_data: checked_logs(data)?,
        })
    }

    fn transform(&self, lambda: f64) -> Vec<f64> {
        transform_logs(&self.log_data, lambda)
    }
}

impl CostFunction for BoxCoxMLE {
    type Param = f64;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let lambda = *p;

        let z = self.transform(lambda);
        let n = z.len() as f64;

        let sigma2 = var(&z);
        if !sigma2.is_finite() {
            return Ok(f64::MAX);
        }
        let sigma2 = sigma2.max(f64::EPSILON);

        let sum_y_ln = self.log_data.iter().sum::<f64>();
        let result = (n / 2.0) * sigma2.ln() - (lambda - 1.0) * sum_y_ln;

        Ok(result)
    }
}

pub fn calc_lambda(data: Vec<f64>, max_iters: u64) -> Result<Vec<Vec<f64>>, Error> {
    let problem = BoxCoxMLE::new(&data)?;
    let problem_cloned = problem.clone();

    let left_border = -5.0;
    let right_border = 5.0;

    let solver = BrentOpt::new(left_border, right_border);

    let res = Executor::new(problem_cloned, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;

    let best_param = match res.state.best_param {
        Some(best_param) => best_param,
        None => return Err(Error::msg("No best_param for box cox transformation")),
    };

    let best_cost = res.state.best_cost;
    let transformed_data = problem.transform(best_param);

    Ok(vec![vec![best_param, best_cost], transformed_data])
}

pub fn box_cox(data: &[f64], lambda: f64) -> Result<Vec<f64>, BoxCoxError> {
    if !lambda.is_finite() {
        return Err(BoxCoxError::InvalidLambda(lambda));
    }
    Ok(transform_logs(&checked_logs(data)?, lambda))
}

pub fn z_score(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
    let z_scores = data
        .iter()
        .map(|x| (x - mean) / std_dev)
        .collect::<Vec<f64>>();

    z_scores
}

pub fn reverse_box_cox(z: &[f64], lambda: f64) -> Result<Vec<f64>, BoxCoxError> {
    if !lambda.is_finite() {
        return Err(BoxCoxError::InvalidLambda(lambda));
    }
    z.iter()
        .enumerate()
        .map(|(index, &zi)| {
            let out_of_domain = BoxCoxError::OutOfDomain {
                index,
                value: zi,
                lambda,
            };
            if !zi.is_finite() {
                return Err(out_of_domain);
            }
            if lambda.abs() <= EPSILON {
                return Ok(zi.exp());
            }
            let arg = lambda * zi;
            if arg <= -1.0 {
                return Err(out_of_domain);
            }
            Ok((arg.ln_1p() / lambda).exp())
        })
        .collect()
}

pub fn reverse_z_score(z: &[f64], mu: f64, std_dev: f64) -> Vec<f64> {
    z.iter().map(|zi| zi * std_dev + mu).collect::<Vec<f64>>()
}

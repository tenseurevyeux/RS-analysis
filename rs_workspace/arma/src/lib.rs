use argmin::core::Error;
use argmin::core::{CostFunction, Executor, Gradient, IterState};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;
use pacf_acf::{acf, durbin_yw};

const PACF_BOUND: f64 = 0.999;
const FD_STEP: f64 = 1e-5;

fn pacf_to_coeffs(kappas: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::new();
    for (k, kappa) in kappas.into_iter().enumerate() {
        let prev = phi.clone();
        for j in 0..k {
            phi[j] = prev[j] - kappa * prev[k - 1 - j];
        }
        phi.push(kappa);
    }
    phi
}

fn constrain(raw: &[f64], p: usize, q: usize) -> Vec<f64> {
    debug_assert_eq!(raw.len(), 1 + p + q);
    let squash = |&x: &f64| PACF_BOUND * x.tanh();

    let mut out = Vec::with_capacity(raw.len());
    out.push(raw[0]);
    out.extend(pacf_to_coeffs(raw[1..=p].iter().map(squash)));
    out.extend(
        pacf_to_coeffs(raw[1 + p..].iter().map(squash))
            .into_iter()
            .map(|x| -x),
    );
    out
}

fn arma_residuals(z: &[f64], alpha: f64, phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let mut eps: Vec<f64> = Vec::with_capacity(z.len());
    for t in 0..z.len() {
        let mut e = z[t] - alpha;
        for (i, &ph) in phi.iter().enumerate().take(t) {
            e -= ph * z[t - 1 - i];
        }
        for (j, &th) in theta.iter().enumerate().take(t) {
            e -= th * eps[t - 1 - j];
        }
        eps.push(e);
    }
    eps
}

#[derive(Clone)]
struct ARMA {
    z: Vec<f64>,
    p: usize,
    q: usize,
}

impl ARMA {
    fn new(z: Vec<f64>, p: usize, q: usize) -> ARMA {
        Self { z, p, q }
    }
}

impl CostFunction for ARMA {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, psi: &Self::Param) -> Result<Self::Output, Error> {
        if psi.len() != 1 + self.p + self.q {
            return Err(Error::msg(format!(
                "expected {} parameters for ARMA({}, {}), got {}",
                1 + self.p + self.q,
                self.p,
                self.q,
                psi.len()
            )));
        }
        if psi.iter().any(|x| !x.is_finite()) {
            return Err(Error::msg("non-finite ARMA parameters"));
        }

        let coeffs = constrain(psi, self.p, self.q);
        let eps = arma_residuals(
            &self.z,
            coeffs[0],
            &coeffs[1..=self.p],
            &coeffs[1 + self.p..],
        );

        let ssr = eps.iter().map(|&e| e * e).sum::<f64>();
        if !ssr.is_finite() {
            return Err(Error::msg("ARMA residuals overflowed"));
        }

        let t = self.z.len() as f64;
        let sigma2 = (ssr / t).max(1e-12);
        let log_likelihood =
            -0.5 * t * ((2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - ssr / (2.0 * sigma2);

        Ok(-log_likelihood)
    }
}

impl Gradient for ARMA {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, psi: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut x = psi.clone();
        let mut grad = vec![0.0; psi.len()];
        for i in 0..psi.len() {
            let h = FD_STEP * (1.0 + psi[i].abs());
            x[i] = psi[i] + h;
            let up = self.cost(&x)?;
            x[i] = psi[i] - h;
            let down = self.cost(&x)?;
            x[i] = psi[i];
            grad[i] = (up - down) / (2.0 * h);
        }
        Ok(grad)
    }
}

fn information_criterion(name: &str, neg_log_lik: f64, k: f64, t: f64) -> f64 {
    let aic = 2.0 * neg_log_lik + 2.0 * k;
    match name {
        "aic_c" => {
            if t - k - 1.0 > 0.0 {
                aic + 2.0 * k * (k + 1.0) / (t - k - 1.0)
            } else {
                aic
            }
        }
        "bic" => 2.0 * neg_log_lik + k * t.ln(),
        _ => aic,
    }
}

fn raw_from_pacf(kappa: f64) -> f64 {
    (kappa / PACF_BOUND).clamp(-0.99, 0.99).atanh()
}

pub fn arma_optim(
    data: &[f64],
    p_vec: Vec<usize>,
    q_vec: Vec<usize>,
    max_iters: u64,
    optimizer: &str,
) -> Result<(Vec<usize>, Vec<f64>), Error> {
    if !matches!(optimizer, "aic" | "aic_c" | "bic") {
        return Err(Error::msg(format!(
            "unknown information criterion `{optimizer}`; expected one of: aic, aic_c, bic"
        )));
    }
    if data.is_empty() {
        return Err(Error::msg("cannot fit ARMA to an empty series"));
    }
    if let Some(i) = data.iter().position(|x| !x.is_finite()) {
        return Err(Error::msg(format!("non-finite value at index {i}")));
    }

    let p_vec = if p_vec.is_empty() { vec![0] } else { p_vec };
    let q_vec = if q_vec.is_empty() { vec![0] } else { q_vec };

    let t = data.len() as f64;

    let p_max = p_vec.iter().copied().max().unwrap_or(0);
    let q_max = q_vec.iter().copied().max().unwrap_or(0);
    let yw = durbin_yw(data, p_max);
    let acf_vals = acf(data, q_max);

    let mut best: Option<(f64, usize, usize, Vec<f64>)> = None;
    let mut last_err: Option<Error> = None;

    for &p in p_vec.iter() {
        for &q in q_vec.iter() {
            let problem = ARMA::new(data.to_vec(), p, q);

            let condition = ArmijoCondition::new(1e-4)?;
            let ls = BacktrackingLineSearch::new(condition);
            let solver = LBFGS::new(ls, 5);

            let mut psi: Vec<f64> = Vec::with_capacity(1 + p + q);
            psi.push(0.0);
            psi.extend((1..=p).map(|k| raw_from_pacf(yw.pacf[k])));
            psi.extend((1..=q).map(|k| raw_from_pacf(-0.1 * acf_vals[k])));

            let outcome = Executor::new(problem, solver)
                .configure(|state: IterState<Vec<f64>, Vec<f64>, (), (), (), f64>| {
                    state.param(psi.clone()).max_iters(max_iters)
                })
                .run();

            let executor = match outcome {
                Ok(executor) => executor,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            let k = (p + q + 1) as f64;
            let best_cost = executor.state.get_best_cost();
            let criterion = information_criterion(optimizer, best_cost, k, t);
            if !criterion.is_finite() {
                continue;
            }

            if let Some(param) = executor.state.best_param {
                if best.as_ref().is_none_or(|(b, ..)| criterion < *b) {
                    best = Some((criterion, p, q, param));
                }
            }
        }
    }

    match best {
        Some((_, p, q, raw)) => Ok((vec![p, q], constrain(&raw, p, q))),
        None => Err(match last_err {
            Some(e) => Error::msg(format!("no candidate ARMA model could be fitted: {e}")),
            None => Error::msg("no candidate ARMA model could be fitted"),
        }),
    }
}

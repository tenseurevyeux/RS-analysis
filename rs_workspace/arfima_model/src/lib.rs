use arma::arma_optim;
use box_cox::{box_cox, calc_lambda, reverse_box_cox, reverse_z_score, z_score};
use confidence_intervals::{confidence_intervals, estimate_noise_variance, impulse};
use forecast::forecast;
use frac_integration::{DEFAULT_EPSILON, frac_diff, reverse_frac_diff};
use numpy::{PyArray, PyArray1};
use pacf_acf::{acf, durbin_yw, significant_lags};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rs_analysis::calc_herst;
use s_h_esd::{esd_test, repair_anomalies};
use utils::{mean, var};

const MAX_ABS_D: f64 = 0.499;

#[pyclass]
pub struct ARFIMA {
    // Parameters of preparation
    freq: usize,
    alpha: Option<f64>,
    ub: Option<f64>,
    hybrid: Option<bool>,
    max_iter_box_cox: u64,
    min_window: usize,
    n_points: usize,
    nlags: usize,
    max_iters_optim: u64,
    optimizer: String,

    // Parameters of data
    mean: f64,
    std_dev: f64,
    lambda: f64,
    noise: f64,

    // Parameters of model [ARFIMA]
    p: usize,
    d: f64,
    q: usize,
    psi: Vec<f64>, // alpha = psi[0], phi = psi[1:=p] theta = psi[p+1:]
}

#[pymethods]
impl ARFIMA {
    #[new]
    fn new(
        freq: usize,
        alpha: Option<f64>,
        ub: Option<f64>,
        hybrid: Option<bool>,
        max_iter_box_cox: u64,
        min_window: usize,
        n_points: usize,
        nlags: usize,
        max_iters_optim: u64,
        optimizer: String,
    ) -> Self {
        Self {
            freq,
            alpha,
            ub,
            hybrid,
            max_iter_box_cox,
            min_window,
            n_points,
            nlags,
            max_iters_optim,
            optimizer,

            mean: 0.0,
            std_dev: 0.0,
            lambda: 0.0,
            noise: 0.0,

            p: 0,
            d: 0.0,
            q: 0,
            psi: vec![],
        }
    }

    fn fit(&mut self, input: Vec<f64>) -> PyResult<()> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Data shouldn't be empty"));
        }

        /*
           Remove anomalies from input data
        */
        let anomalies =
            esd_test(&input, self.freq, self.alpha, self.ub, self.hybrid).map_err(to_py_err)?;
        let data = repair_anomalies(&input, &anomalies);

        /*
           Transformation of data [Box-Cox + z_score]
        */
        let (lambda, _cost, data) = match calc_lambda(data, self.max_iter_box_cox) {
            Ok(v) => (v[0][0], v[0][1], v[1].clone()),
            Err(e) => return Err(PyValueError::new_err(format!("{}", e))),
        };

        let data_mean = mean(&data);
        let data_std = var(&data).sqrt();
        if !(data_std.is_finite() && data_std > 0.0) {
            return Err(PyValueError::new_err(
                "series has zero variance after Box-Cox; cannot standardise",
            ));
        }

        // data is already Box-Cox transformed by calc_lambda, apply z_score:
        let data = z_score(&data, data_mean, data_std);

        let returns: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
        let data = returns;

        /*
           Fractional difference
        */
        let (herst, _intercept) = match calc_herst(&data, self.min_window, self.n_points) {
            Ok(v) => v,
            Err(e) => return Err(PyValueError::new_err(format!("{}", e))),
        };

        let d = (herst - 0.5).clamp(-MAX_ABS_D, MAX_ABS_D);
        let data = frac_diff(&data, d, DEFAULT_EPSILON);

        /*
           Optimization of model
        */
        let thr = 1.96 / (data.len() as f64).sqrt();

        let pacf = durbin_yw(&data, self.nlags).pacf;
        let acf = acf(&data, self.nlags);

        let p0 = significant_lags(&pacf, thr);
        let q0 = significant_lags(&acf, thr);

        let (params, psi) = match arma_optim(&data, p0, q0, self.max_iters_optim, &self.optimizer) {
            Ok(v) => v,
            Err(e) => return Err(PyValueError::new_err(format!("{}", e))),
        };

        let (p, q) = (params[0], params[1]);

        /*
           Compute estimate noise
        */
        let noise = estimate_noise_variance(&data, &psi, p, q).map_err(to_py_err)?;

        self.mean = data_mean;
        self.std_dev = data_std;
        self.lambda = lambda;
        self.d = d;
        self.p = p;
        self.q = q;
        self.psi = psi;
        self.noise = noise;

        Ok(())
    }

    fn forecast<'py>(
        &self,
        py: Python<'py>,
        data: Vec<f64>,
        horizon: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_fitted()?;

        let levels = z_score(
            &box_cox(&data, self.lambda).map_err(to_py_err)?,
            self.mean,
            self.std_dev,
        );
        if levels.len() < 2 {
            return Err(PyValueError::new_err("need at least 2 observations"));
        }

        let last_level = levels[levels.len() - 1];
        let returns: Vec<f64> = levels.windows(2).map(|w| w[1] - w[0]).collect();
        let y = frac_diff(&returns, self.d, DEFAULT_EPSILON);

        let y_hat = forecast(&y, &self.psi, self.p, self.q, horizon).map_err(to_py_err)?;
        let mut full = y;
        full.extend_from_slice(&y_hat);
        let inv = reverse_frac_diff(&full, self.d, DEFAULT_EPSILON);
        let ret_hat = &inv[inv.len().saturating_sub(horizon)..];

        let mut level = last_level;
        let preds_z: Vec<f64> = ret_hat
            .iter()
            .map(|r| {
                level += r;
                level
            })
            .collect();

        let preds = reverse_z_score(&preds_z, self.mean, self.std_dev);
        let preds = reverse_box_cox(&preds, self.lambda).map_err(to_py_err)?;

        Ok(PyArray::from_vec(py, preds))
    }

    fn confidence_intervals(
        &self,
        preds: Vec<f64>,
        horizon: usize,
        epsilon: Option<f64>,
        quantile: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let _ = epsilon;
        self.ensure_fitted()?;
        if preds.is_empty() {
            return Err(PyValueError::new_err("Predictions vector cannot be empty"));
        }

        let imp = impulse(horizon.max(preds.len()), &self.psi, self.p, self.d, self.q)
            .map_err(to_py_err)?;
        let mut acc = 0.0;
        let imp: Vec<f64> = imp
            .iter()
            .map(|x| {
                acc += x;
                acc
            })
            .collect();
        let sigma = self.noise.sqrt();

        let preds_bc = box_cox(&preds, self.lambda).map_err(to_py_err)?;
        let preds_z = z_score(&preds_bc, self.mean, self.std_dev);

        let (down_z, top_z) =
            confidence_intervals(&preds_z, &imp, sigma, quantile).map_err(to_py_err)?;

        let down_ci = reverse_z_score(&down_z, self.mean, self.std_dev);
        let down_ci = reverse_box_cox(&down_ci, self.lambda).map_err(to_py_err)?;

        let top_ci = reverse_z_score(&top_z, self.mean, self.std_dev);
        let top_ci = reverse_box_cox(&top_ci, self.lambda).map_err(to_py_err)?;

        Ok((down_ci, top_ci))
    }

    fn get_mean(&self) -> PyResult<f64> {
        Ok(self.mean)
    }

    fn get_std_dev(&self) -> PyResult<f64> {
        Ok(self.std_dev)
    }

    fn get_lambda(&self) -> PyResult<f64> {
        Ok(self.lambda)
    }

    fn get_noise(&self) -> PyResult<f64> {
        Ok(self.noise)
    }

    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = vec![self.p as f64, self.d, self.q as f64];
        Ok(PyArray::from_vec(py, p))
    }

    fn get_psi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_vec(py, self.psi.clone()))
    }
}

impl ARFIMA {
    fn ensure_fitted(&self) -> PyResult<()> {
        if self.psi.is_empty() {
            return Err(PyValueError::new_err(
                "model is not fitted: call fit() first",
            ));
        }
        Ok(())
    }
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymodule]
fn arfima_model(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ARFIMA", py.get_type::<ARFIMA>())?;
    Ok(())
}

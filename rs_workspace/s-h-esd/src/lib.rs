use statrs::distribution::{ContinuousCDF, StudentsT};
use utils::{ModelError, mean, median, var};

fn t_ppf(prob: f64, df: f64) -> Option<f64> {
    if df <= 0.0 || prob <= 0.0 || prob >= 1.0 {
        return None;
    }
    StudentsT::new(0.0, 1.0, df)
        .ok()
        .map(|dist| dist.inverse_cdf(prob))
}

fn mad(x: &[f64]) -> f64 {
    let med = median(x);
    let abs_dev: Vec<f64> = x.iter().map(|&v| (v - med).abs()).collect();
    median(&abs_dev)
}

fn seasonal_mean(x: &[f64], freq: usize) -> Vec<f64> {
    if freq == 0 {
        return vec![0.0];
    }
    let mut means = Vec::with_capacity(freq);
    for i in 0..freq {
        let values: Vec<f64> = x
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % freq == i)
            .map(|(_, &val)| val)
            .collect();

        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        means.push(mean);
    }

    means
}

fn ts_s_md_decomposition(x: &[f64], freq: usize) -> Vec<f64> {
    let nobs = x.len();
    if nobs == 0 {
        return vec![];
    }

    let period_averages = seasonal_mean(x, freq);
    let mut seasonal = Vec::with_capacity(nobs);
    while seasonal.len() < nobs {
        seasonal.extend_from_slice(&period_averages);
    }
    seasonal.truncate(nobs);

    let med_value = median(x);

    x.iter()
        .zip(seasonal.iter())
        .map(|(&obs, &s)| obs - s - med_value)
        .collect()
}

fn esd_test_statistics(x: &[f64], hybrid: Option<bool>) -> (f64, f64) {
    let hybrid = hybrid.unwrap_or(true);

    if hybrid {
        let m = mad(x) * 1.4826;
        if m > 1e-12 {
            (median(x), m)
        } else {
            let std = var(x).sqrt();
            (median(x), std)
        }
    } else {
        (mean(x), var(x).sqrt())
    }
}

pub fn esd_test(
    x: &[f64],
    freq: usize,
    alpha: Option<f64>,
    ub: Option<f64>,
    hybrid: Option<bool>,
) -> Result<Vec<usize>, ModelError> {
    if freq == 0 {
        return Err(ModelError::InvalidParameter {
            name: "freq",
            message: "seasonal period must be at least 1".to_string(),
        });
    }
    let significance = alpha.unwrap_or(0.05);
    if !(significance > 0.0 && significance <= 0.5) {
        return Err(ModelError::InvalidParameter {
            name: "alpha",
            message: format!(
                "alpha is a significance level in (0, 0.5], e.g. 0.05; got {significance}. \
                 If you meant a confidence level, pass 1 - confidence."
            ),
        });
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(ModelError::NonFiniteInput { index });
    }
    if x.is_empty() {
        return Ok(vec![]);
    }

    let mut ub = ub.unwrap_or(0.499);
    let hybrid = hybrid.unwrap_or(true);

    if ub > 0.4999 {
        ub = 0.499;
    }

    let nobs = x.len();
    let k = (ub * nobs as f64).floor().max(1.0) as usize;

    let residuals = ts_s_md_decomposition(x, freq);
    let mut res: Vec<Option<f64>> = residuals.into_iter().map(Some).collect();
    let mut candidates: Vec<usize> = Vec::new();
    let mut n_outliers = 0;

    for i in 1..=k {
        let unmasked: Vec<f64> = res.iter().filter_map(|&val| val).collect();
        if unmasked.len() <= 2 {
            break;
        }

        let (location, dispersion) = esd_test_statistics(&unmasked, Some(hybrid));
        if dispersion < 1e-12 || !dispersion.is_finite() {
            break;
        }

        let tmp: Vec<f64> = res
            .iter()
            .map(|opt| match opt {
                Some(val) => (val - location).abs() / dispersion,
                None => -f64::INFINITY,
            })
            .collect();

        let idx_opt = tmp
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index);

        if let Some(idx) = idx_opt {
            let test_statistic = tmp[idx];
            let n_cur = unmasked.len();

            if let Some(critical_value) = gesd_critical_value(n_cur, significance) {
                if test_statistic > critical_value {
                    n_outliers = i;
                }
            }

            candidates.push(idx);
            res[idx] = None;
        }
    }

    candidates.truncate(n_outliers);
    Ok(candidates)
}

pub fn repair_anomalies(data: &[f64], anomalies: &[usize]) -> Vec<f64> {
    let n = data.len();
    let mut is_bad = vec![false; n];
    for &i in anomalies {
        if i < n {
            is_bad[i] = true;
        }
    }

    let mut out = data.to_vec();
    let mut prev_good: Option<usize> = None;
    let mut next_good = vec![None; n];
    let mut next: Option<usize> = None;
    for i in (0..n).rev() {
        next_good[i] = next;
        if !is_bad[i] {
            next = Some(i);
        }
    }

    for i in 0..n {
        if !is_bad[i] {
            prev_good = Some(i);
            continue;
        }
        out[i] = match (prev_good, next_good[i]) {
            (Some(l), Some(r)) => {
                let frac = (i - l) as f64 / (r - l) as f64;
                data[l] + (data[r] - data[l]) * frac
            }
            (Some(l), None) => data[l],
            (None, Some(r)) => data[r],
            (None, None) => data[i],
        };
    }
    out
}

fn gesd_critical_value(n_cur: usize, significance: f64) -> Option<f64> {
    if n_cur < 3 {
        return None;
    }
    let n = n_cur as f64;
    let df = n - 2.0;
    let p = 1.0 - significance / (2.0 * n);
    let t = t_ppf(p, df)?;
    Some((n - 1.0) * t / ((df + t * t) * n).sqrt())
}

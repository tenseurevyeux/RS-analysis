use utils::{autocov, var};

pub struct YwResult {
    pub pacf: Vec<f64>,
    pub phi: Vec<Vec<f64>>,
}

pub fn durbin_yw(series: &[f64], nlags: usize) -> YwResult {
    let n = series.len();
    let mut pacf = vec![0.0; nlags + 1];
    let mut phi = vec![vec![0.0; nlags + 1]; nlags + 1];
    pacf[0] = 1.0;
    if n == 0 {
        return YwResult { pacf, phi };
    }

    let eff = nlags.min(n - 1);
    let gamma = autocov(series, eff);
    let mut sigma2 = vec![0.0; eff + 1];
    sigma2[0] = gamma[0].max(1e-12);

    for lag in 1..=eff {
        let mut acc = 0.0;
        for j in 1..lag {
            acc += phi[lag - 1][j] * gamma[lag - j];
        }

        let kappa = if sigma2[lag - 1] > 1e-12 {
            ((gamma[lag] - acc) / sigma2[lag - 1]).clamp(-0.9999, 0.9999)
        } else {
            0.0
        };

        pacf[lag] = kappa;
        phi[lag][lag] = kappa;

        for j in 1..lag {
            phi[lag][j] = phi[lag - 1][j] - kappa * phi[lag - 1][lag - j];
        }

        sigma2[lag] = (sigma2[lag - 1] * (1.0 - kappa * kappa)).max(1e-12);
    }

    YwResult { pacf, phi }
}

pub fn acf(series: &[f64], nlags: usize) -> Vec<f64> {
    let mut out = vec![0.0; nlags + 1];
    out[0] = 1.0;

    let v = var(series);
    if series.is_empty() || v < 1e-12 {
        return out;
    }
    for (slot, cov) in out.iter_mut().zip(autocov(series, nlags)).skip(1) {
        *slot = (cov / v).clamp(-1.0, 1.0);
    }
    out
}

pub fn significant_lags(vals: &[f64], threshold: f64) -> Vec<usize> {
    vals.iter()
        .enumerate()
        .filter_map(|(idx, &v)| {
            if idx >= 1 && v.abs() > threshold {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

use utils::ModelError;

pub fn estimate_noise_variance(
    data: &[f64],
    psi: &[f64],
    p: usize,
    q: usize,
) -> Result<f64, ModelError> {
    let length = data.len();
    if length == 0 {
        return Err(ModelError::EmptySeries);
    }
    if psi.len() < 1 + p + q {
        return Err(ModelError::ParamLengthMismatch {
            expected: 1 + p + q,
            got: psi.len(),
        });
    }

    let alpha = psi[0];
    let phi = &psi[1..=p];
    let theta = &psi[p + 1..=p + q];

    let mut eps: Vec<f64> = Vec::with_capacity(length);
    for t in 0..length {
        let mut e = data[t] - alpha;

        // AR component
        for i in 0..p {
            if t > i {
                e -= phi[i] * data[t - 1 - i];
            }
        }

        // MA component
        for j in 0..q {
            if t > j {
                e -= theta[j] * eps[t - 1 - j];
            }
        }
        eps.push(e);
    }

    let ssr: f64 = eps.iter().map(|&e| e * e).sum();
    let df = length.saturating_sub(1 + p + q).max(1) as f64;
    Ok(ssr / df)
}

pub fn impulse(
    horizon: usize,
    psi: &[f64],
    p: usize,
    d: f64,
    q: usize,
) -> Result<Vec<f64>, ModelError> {
    if psi.len() < 1 + p + q {
        return Err(ModelError::ParamLengthMismatch {
            expected: 1 + p + q,
            got: psi.len(),
        });
    }

    let phi = &psi[1..=p];
    let theta = &psi[p + 1..=p + q];

    let mut w = vec![0.0; horizon + 1];
    w[0] = 1.0;
    for k in 1..=horizon {
        w[k] = w[k - 1] * (d + (k as f64 - 1.0)) / (k as f64);
    }

    let mut pi = vec![0.0; horizon + 1];
    pi[0] = 1.0;
    for i in 1..=horizon {
        let end = p.min(i);
        let mut sum = 0.0;
        for j in 0..end {
            sum += phi[j] * pi[i - 1 - j];
        }
        pi[i] = sum;
    }

    let mut u = vec![0.0; horizon + 1];
    for i in 0..=horizon {
        let mut sum = w[i];
        let end = q.min(i);
        for j in 0..end {
            sum += theta[j] * w[i - 1 - j];
        }
        u[i] = sum;
    }

    let mut imp = vec![0.0; horizon + 1];
    for i in 0..=horizon {
        let mut acc = 0.0;
        for j in 0..=i {
            acc += u[j] * pi[i - j];
        }
        imp[i] = acc;
    }

    Ok(imp)
}

pub fn confidence_intervals(
    preds: &[f64],
    impulse: &[f64],
    sigma: f64,
    q: f64,
) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
    if impulse.len() < preds.len() {
        return Err(ModelError::LengthMismatch {
            left: preds.len(),
            right: impulse.len(),
        });
    }
    let mut ci_top = Vec::with_capacity(preds.len());
    let mut ci_down = Vec::with_capacity(preds.len());
    let mut acc = 0.0;

    for (h, &y) in preds.iter().enumerate() {
        acc += impulse[h] * impulse[h];

        let err = sigma * acc.sqrt();
        let lower = y - q * err;
        let upper = y + q * err;
        ci_top.push(upper);
        ci_down.push(lower);
    }

    Ok((ci_down, ci_top))
}

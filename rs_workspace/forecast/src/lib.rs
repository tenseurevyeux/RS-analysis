use utils::ModelError;

pub fn forecast(
    z: &[f64],
    psi: &[f64],
    p: usize,
    q: usize,
    horizon: usize,
) -> Result<Vec<f64>, ModelError> {
    if z.is_empty() {
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

    let mut eps_hat = Vec::with_capacity(z.len() + horizon);
    for t in 0..z.len() {
        let mut e = z[t] - alpha;
        for i in 0..p {
            if t > i {
                e -= phi[i] * z[t - 1 - i];
            }
        }
        for j in 0..q {
            if t > j {
                e -= theta[j] * eps_hat[t - 1 - j];
            }
        }
        eps_hat.push(e);
    }

    let mut z_hat = z.to_vec();

    for _h in 1..=horizon {
        let t = z_hat.len();
        let mut val = alpha;
        for i in 0..p {
            if t > i {
                val += phi[i] * z_hat[t - 1 - i];
            }
        }
        for j in 0..q {
            if t > j {
                val += theta[j] * eps_hat[t - 1 - j];
            }
        }
        z_hat.push(val);
        eps_hat.push(0.0);
    }

    Ok(z_hat[z.len()..].to_vec())
}

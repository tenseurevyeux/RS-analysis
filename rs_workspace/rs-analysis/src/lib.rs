use statrs::function::gamma::ln_gamma;

fn linear_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    let sum_x = x.iter().sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
    let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();

    let denom = n * sum_x2 - sum_x.powi(2);
    if denom.abs() < 1e-12 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = sum_y / n - slope * (sum_x / n);
    Some((slope, intercept))
}

fn expected_rs(n: usize) -> f64 {
    let nf = n as f64;
    let sum: f64 = (1..n).map(|i| ((nf - i as f64) / i as f64).sqrt()).sum();
    if n <= 340 {
        let ratio = (ln_gamma((nf - 1.0) / 2.0) - ln_gamma(nf / 2.0)).exp();
        ratio / std::f64::consts::PI.sqrt() * sum
    } else {
        (nf - 0.5) / nf * sum / (nf * std::f64::consts::FRAC_PI_2).sqrt()
    }
}

fn window_sizes(n: usize, min_window: usize, n_iter: usize) -> Vec<usize> {
    let min_window = min_window.max(2);
    let max_window = (n / 2).max(min_window);
    let num_points = n_iter.max(2);

    let log_min = (min_window as f64).log10();
    let log_max = (max_window as f64).log10();

    let mut windows: Vec<usize> = (0..num_points)
        .map(|i| {
            let exponent = log_min + (log_max - log_min) * (i as f64) / ((num_points - 1) as f64);
            (10f64.powf(exponent).round() as usize)
                .max(min_window)
                .min(n)
        })
        .collect();
    windows.dedup();
    windows
}

fn mean_rs_for_window(series: &[f64], window: usize) -> Option<f64> {
    let segments = series.len() / window;
    let mut sum = 0.0;
    let mut count = 0usize;

    for segment in series.chunks_exact(window).take(segments) {
        let mean = segment.iter().sum::<f64>() / window as f64;

        let (mut cum, mut lo, mut hi) = (0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
        for &value in segment {
            cum += value - mean;
            lo = lo.min(cum);
            hi = hi.max(cum);
        }
        let r = hi - lo;

        let s = (segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window as f64).sqrt();

        if s > 1e-12 {
            sum += r / s;
            count += 1;
        }
    }

    (count > 0).then(|| sum / count as f64)
}

fn rs_analysis(series: &[f64], min_window: usize, n_iter: usize) -> (Vec<usize>, Vec<f64>) {
    window_sizes(series.len(), min_window, n_iter)
        .into_iter()
        .filter_map(|w| mean_rs_for_window(series, w).map(|rs| (w, rs)))
        .unzip()
}

pub fn calc_herst(series: &[f64], min_window: usize, n_iter: usize) -> Result<(f64, f64), String> {
    if series.is_empty() {
        return Err("Error[rs-analysis]: empty data series".to_string());
    }
    if let Some(i) = series.iter().position(|v| !v.is_finite()) {
        return Err(format!("Error[rs-analysis]: non-finite value at index {i}"));
    }
    if series.len() < min_window.max(2) * 2 {
        return Err(format!(
            "Error[rs-analysis]: series length {} too small for min_window {}",
            series.len(),
            min_window
        ));
    }

    let (windows, rs_series) = rs_analysis(series, min_window, n_iter);

    let mut log_w = Vec::with_capacity(windows.len());
    let mut log_rs = Vec::with_capacity(windows.len());
    for (&w, &rs) in windows.iter().zip(rs_series.iter()) {
        if rs.is_finite() && rs > 0.0 {
            log_w.push((w as f64).log10());
            log_rs.push(rs.log10() - expected_rs(w).log10());
        }
    }

    if log_w.len() < 2 {
        return Err("Error[rs-analysis]: not enough valid R/S points for regression".to_string());
    }

    let (slope, intercept) = linear_regression(&log_w, &log_rs).ok_or_else(|| {
        "Error[rs-analysis]: degenerate regression (all window sizes identical)".to_string()
    })?;
    Ok((0.5 + slope, intercept))
}

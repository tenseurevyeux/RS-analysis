from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arfima_model import ARFIMA

OUT = Path(__file__).resolve().parents[3] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT = 0.87
HORIZON = 100
RNG = np.random.default_rng(42)

PARAMS = dict(
    freq=7, alpha=0.05, ub=0.499, hybrid=True,
    max_box_cox_iters=200, min_window=50, rs_iters=2000,
    nlags=20, max_iters_optim=50, optimizer="aic",
)


def frac_integrated(n, d, rng):
    psi = np.ones(n)
    for k in range(1, n):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    eps = rng.standard_normal(2 * n)
    return np.convolve(eps, psi)[n:2 * n]


def make_series(n=1500):
    t = np.arange(n)
    rw = 100 + np.cumsum(0.05 + rng_noise(n))
    ar = np.zeros(n)
    e = RNG.standard_normal(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    trend_ar = 100 + 0.03 * t + 2.0 * ar
    long_mem = 100 * np.exp(0.005 * np.cumsum(frac_integrated(n, 0.3, RNG)))
    return {
        "Random walk + drift": (rw, None),
        "Trend + AR(1)": (trend_ar, None),
        "Fractional noise (true d=0.3)": (long_mem, 0.3),
    }


def rng_noise(n):
    return RNG.standard_normal(n)


def metrics(y, yhat):
    err = y - yhat
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE %": float(np.mean(np.abs(err / y)) * 100),
    }


def run_case(name, series, true_d):
    cut = int(len(series) * SPLIT)
    train, test = series[:cut], series[cut:cut + HORIZON]
    h = len(test)

    rust = ARFIMA(*PARAMS.values())
    rust.fit(train)
    p, d, q = rust.get_params()
    p, q = int(p), int(q)
    rust_pred = np.asarray(rust.forecast(train, h))
    lo, hi = rust.confidence_intervals(list(rust_pred), h, None, 0.95)

    py_fit = ARIMA(train, order=(p, 1, q)).fit()
    py_pred = np.asarray(py_fit.forecast(steps=h))
    py_ci = py_fit.get_forecast(h).conf_int(alpha=0.05)

    m_rust, m_py = metrics(test, rust_pred), metrics(test, py_pred)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1],
                           sharex=True)
    ctx = 200
    x_tr = np.arange(cut - ctx, cut)
    x_te = np.arange(cut, cut + h)
    ax[0].plot(x_tr, train[-ctx:], color="gray", label="train")
    ax[0].plot(x_te, test, color="black", label="actual")
    ax[0].plot(x_te, rust_pred, color="tab:orange",
               label=f"Rust ARFIMA(p={p}, d={d:.2f}, q={q})")
    ax[0].fill_between(x_te, lo, hi, color="tab:orange", alpha=0.15)
    ax[0].plot(x_te, py_pred, color="tab:blue",
               label=f"statsmodels ARIMA({p},1,{q})")
    ax[0].fill_between(x_te, py_ci[:, 0], py_ci[:, 1], color="tab:blue",
                       alpha=0.12)
    ax[0].axvline(cut, ls="--", color="k", lw=0.8)
    title = name + (f"  (true d={true_d})" if true_d is not None else "")
    ax[0].set_title(f"{title}\nRMSE  Rust={m_rust['RMSE']:.3f}  "
                    f"ARIMA={m_py['RMSE']:.3f}")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(x_te, test - rust_pred, color="tab:orange", label="Rust err")
    ax[1].plot(x_te, test - py_pred, color="tab:blue", label="ARIMA err")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("error")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()

    fname = OUT / (name.split(" (")[0].lower()
                   .replace(" + ", "_").replace(" ", "_") + ".png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"saved {fname}")

    return {"name": name, "rust": m_rust, "py": m_py, "d": d, "true_d": true_d}


def summary_plot(results):
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    w = 0.38
    ax[0].bar(x - w / 2, [r["rust"]["RMSE"] for r in results], w,
              label="Rust ARFIMA", color="tab:orange")
    ax[0].bar(x + w / 2, [r["py"]["RMSE"] for r in results], w,
              label="statsmodels ARIMA", color="tab:blue")
    ax[0].set_xticks(x, [n.split(" (")[0] for n in names], rotation=15)
    ax[0].set_ylabel("RMSE (lower is better)")
    ax[0].set_title("Forecast error")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(x, [r["d"] for r in results], 0.5, color="tab:green",
              label="estimated d (R/S)")
    for i, r in enumerate(results):
        if r["true_d"] is not None:
            ax[1].hlines(r["true_d"], i - 0.35, i + 0.35, color="red",
                         ls="--", label="true d" if i == 2 else None)
    ax[1].set_xticks(x, [n.split(" (")[0] for n in names], rotation=15)
    ax[1].set_title("Estimated fractional order d")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT / "summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def check(results):
    for r in results:
        assert -0.5 < r["d"] < 0.5, f"{r['name']}: d out of range"
        assert np.isfinite(r["rust"]["RMSE"]), f"{r['name']}: NaN forecast"
    print("sanity checks passed")


if __name__ == "__main__":
    results = [run_case(n, s, td) for n, (s, td) in make_series().items()]
    summary_plot(results)
    check(results)

    print(f"\n{'case':35s} {'Rust RMSE':>10s} {'ARIMA RMSE':>11s} {'d':>6s}")
    for r in results:
        print(f"{r['name']:35s} {r['rust']['RMSE']:10.3f} "
              f"{r['py']['RMSE']:11.3f} {r['d']:6.2f}")

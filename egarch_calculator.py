"""EGARCH(1,1) volatility & VaR calculator (Nelson, 1991).

Models log-variance instead of variance directly, so no positivity constraints are
needed on the parameters, and captures the leverage effect through a term that
responds smoothly to the sign of the shock (rather than GJR-GARCH's on/off indicator).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from garch_calculator import fit_garch, garch_volatility
from gjr_garch_calculator import fit_gjr_garch, gjr_volatility

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}

E_ABS_Z = np.sqrt(2 / np.pi)  # 표준정규분포에서 E[|z|]


def _log_variance_path(returns, omega, alpha, gamma, beta):
    """EGARCH(1,1) 파라미터가 주어졌을 때, 전체 기간의 (로그)분산 경로를 계산."""
    n = len(returns)
    log_sigma2 = np.empty(n)
    sigma2 = np.empty(n)
    log_sigma2[0] = np.log(np.var(returns))
    sigma2[0] = np.exp(log_sigma2[0])
    for t in range(1, n):
        z_prev = returns[t - 1] / np.sqrt(sigma2[t - 1])  # 표준화된 어제 충격
        log_sigma2[t] = (
            omega
            + alpha * (abs(z_prev) - E_ABS_Z)
            + gamma * z_prev
            + beta * log_sigma2[t - 1]
        )
        sigma2[t] = np.exp(log_sigma2[t])
    return sigma2, log_sigma2


def _negative_log_likelihood(params, returns):
    """최적화 목적함수: 음의 로그우도."""
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1:  # 로그분산이 AR(1) 과정이므로 |beta|<1이 정상성 조건
        return 1e10
    sigma2, _ = _log_variance_path(returns, omega, alpha, gamma, beta)
    if np.any(sigma2 <= 0) or np.any(~np.isfinite(sigma2)):
        return 1e10
    log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -np.sum(log_likelihood)


def fit_egarch(returns):
    """과거 수익률에 가장 잘 맞는 EGARCH(1,1) 파라미터를 추정."""
    returns = np.asarray(returns)
    initial_guess = [np.log(np.var(returns)) * 0.1, 0.1, -0.05, 0.9]
    bounds = [(None, None), (0, 1), (-1, 1), (-0.999, 0.999)]

    result = minimize(
        _negative_log_likelihood,
        initial_guess,
        args=(returns,),
        method="L-BFGS-B",
        bounds=bounds,
    )
    omega, alpha, gamma, beta = result.x
    return {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "converged": result.success,
    }


def egarch_volatility(returns, params=None):
    """EGARCH(1,1)로 추정한 다음 날(t+1)의 변동성."""
    returns = np.asarray(returns)
    if params is None:
        params = fit_egarch(returns)
    sigma2, log_sigma2 = _log_variance_path(
        returns, params["omega"], params["alpha"], params["gamma"], params["beta"]
    )
    z_prev = returns[-1] / np.sqrt(sigma2[-1])
    next_log_sigma2 = (
        params["omega"]
        + params["alpha"] * (abs(z_prev) - E_ABS_Z)
        + params["gamma"] * z_prev
        + params["beta"] * log_sigma2[-1]
    )
    return np.sqrt(np.exp(next_log_sigma2))


def egarch_var(returns, confidence=0.95, portfolio_value=1.0, params=None):
    """EGARCH(1,1) 기반 변동성을 이용한 모수적 VaR (평균은 0으로 가정)."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    sigma = egarch_volatility(returns, params)
    z = Z_SCORES[confidence]
    return z * sigma * portfolio_value


def plot_asymmetry(results, shock_size, out_path="asymmetry_comparison.png"):
    """같은 크기의 상승/하락 충격 후 변동성 예측을 모델별로 막대그래프로 비교."""
    names = list(results)
    ups = [results[n][0] * 100 for n in names]
    downs = [results[n][1] * 100 for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ups, width, color="#4C72B0", label=f"After +{shock_size:.0%} shock")
    ax.bar(x + width / 2, downs, width, color="#C44E52", label=f"After -{shock_size:.0%} shock")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Next-day volatility forecast (%)")
    ax.set_title("Leverage Effect: Same-Size Up vs Down Shock, Different Volatility Response")
    ax.set_ylim(0, max(downs + ups) * 1.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[안내] 차트를 '{out_path}'에 저장했습니다.")


if __name__ == "__main__":
    # GJR-GARCH 데모와 동일한 방식: 비대칭성이 내재된 가짜 데이터 생성
    rng = np.random.default_rng(7)
    n = 500
    returns = np.empty(n)
    sigma = 0.015
    for t in range(n):
        returns[t] = sigma * rng.normal()
        indicator = 1.0 if returns[t] < 0 else 0.0
        sigma = np.sqrt(0.00001 + (0.03 + 0.08 * indicator) * returns[t] ** 2 + 0.85 * sigma**2)

    e_params = fit_egarch(returns)
    print(
        f"EGARCH 추정 파라미터: omega={e_params['omega']:.6f}, alpha={e_params['alpha']:.4f}, "
        f"gamma={e_params['gamma']:.4f}, beta={e_params['beta']:.4f}"
    )
    print("(gamma가 음수면 하락 충격이 로그변동성을 더 크게 키운다는 뜻 - 레버리지 효과 존재)\n")

    # 3개 모델(GARCH, GJR-GARCH, EGARCH) 비교: 똑같은 크기(3%)의 상승/하락 충격 반응 비교
    shock_size = 0.03
    history = returns[:-1]
    returns_after_up = np.append(history, shock_size)
    returns_after_down = np.append(history, -shock_size)

    garch_params = fit_garch(returns)
    gjr_params = fit_gjr_garch(returns)

    results = {
        "GARCH(1,1)": (
            garch_volatility(returns_after_up, garch_params),
            garch_volatility(returns_after_down, garch_params),
        ),
        "GJR-GARCH": (
            gjr_volatility(returns_after_up, gjr_params),
            gjr_volatility(returns_after_down, gjr_params),
        ),
        "EGARCH": (
            egarch_volatility(returns_after_up, e_params),
            egarch_volatility(returns_after_down, e_params),
        ),
    }

    print(f"{'모델':<12} {'+3% 충격 후 변동성':<20} {'-3% 충격 후 변동성':<20} {'차이(%p)'}")
    for name, (vol_up, vol_down) in results.items():
        diff = (vol_down - vol_up) * 100
        print(f"{name:<12} {vol_up:<20.4%} {vol_down:<20.4%} {diff:+.2f}")

    plot_asymmetry(results, shock_size)

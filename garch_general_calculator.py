"""General GARCH(p,q) volatility & VaR calculator: parameters estimated by maximum likelihood (MLE).

GARCH(1,1) is the special case p=1, q=1 (see garch_calculator.py). This module generalizes
to arbitrary (p, q) orders, letting the model look back further in time for the ARCH
(shock) and GARCH (variance) terms.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def _variance_path(returns, omega, alphas, betas):
    """GARCH(p,q) 파라미터가 주어졌을 때, 전체 기간의 분산 경로를 계산."""
    q, p = len(alphas), len(betas)
    n = len(returns)
    m = max(p, q)
    sigma2 = np.full(n, np.var(returns))  # 초기 m개 구간은 표본분산으로 채움
    for t in range(m, n):
        arch_term = sum(alphas[i] * returns[t - 1 - i] ** 2 for i in range(q))
        garch_term = sum(betas[j] * sigma2[t - 1 - j] for j in range(p))
        sigma2[t] = omega + arch_term + garch_term
    return sigma2


def _negative_log_likelihood(params, returns, p, q):
    """최적화 목적함수: 음의 로그우도 (이 값을 최소화 = 우도를 최대화)."""
    omega = params[0]
    alphas = params[1 : 1 + q]
    betas = params[1 + q : 1 + q + p]
    if omega <= 0 or np.any(alphas < 0) or np.any(betas < 0) or (alphas.sum() + betas.sum()) >= 1:
        return 1e10  # 정상성(stationarity) 조건 위반 시 큰 페널티
    sigma2 = _variance_path(returns, omega, alphas, betas)
    log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -np.sum(log_likelihood)


def fit_garch(returns, p=1, q=1):
    """과거 수익률에 가장 잘 맞는 GARCH(p,q) 파라미터(omega, alphas, betas)를 추정."""
    returns = np.asarray(returns)
    sample_var = np.var(returns)
    initial_guess = [sample_var * 0.05] + [0.05 / q] * q + [0.90 / p] * p
    bounds = [(1e-10, None)] + [(0, 1)] * (p + q)

    result = minimize(
        _negative_log_likelihood,
        initial_guess,
        args=(returns, p, q),
        method="L-BFGS-B",
        bounds=bounds,
    )
    omega = result.x[0]
    alphas = result.x[1 : 1 + q]
    betas = result.x[1 + q : 1 + q + p]
    return {
        "omega": omega,
        "alphas": alphas,
        "betas": betas,
        "p": p,
        "q": q,
        "converged": result.success,
        "log_likelihood": -result.fun,
    }


def garch_volatility(returns, params=None, p=1, q=1):
    """GARCH(p,q)로 추정한 다음 날(t+1)의 변동성."""
    returns = np.asarray(returns)
    if params is None:
        params = fit_garch(returns, p=p, q=q)
    omega, alphas, betas = params["omega"], params["alphas"], params["betas"]
    sigma2 = _variance_path(returns, omega, alphas, betas)
    arch_term = sum(alphas[i] * returns[-1 - i] ** 2 for i in range(len(alphas)))
    garch_term = sum(betas[j] * sigma2[-1 - j] for j in range(len(betas)))
    next_sigma2 = omega + arch_term + garch_term
    return np.sqrt(next_sigma2)


def garch_var(returns, confidence=0.95, portfolio_value=1.0, params=None, p=1, q=1):
    """GARCH(p,q) 기반 변동성을 이용한 모수적 VaR (평균은 0으로 가정)."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    sigma = garch_volatility(returns, params, p=p, q=q)
    z = Z_SCORES[confidence]
    return z * sigma * portfolio_value


def long_run_variance(params):
    """장기평균 분산: omega / (1 - sum(alphas) - sum(betas))."""
    persistence = params["alphas"].sum() + params["betas"].sum()
    return params["omega"] / (1 - persistence)


def plot_order_comparison(labels, persistences, long_run_vols, out_path="garch_order_comparison.png"):
    """차수(p,q)별 잔류율과 장기평균 변동성을 비교 (과적합 불안정성 시각화)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(labels, persistences, color="#4C72B0")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Stationarity limit (=1)")
    ax1.set_ylim(0.9, 1.01)
    ax1.set_title("Persistence (sum of alphas + betas)")
    ax1.legend()

    ax2.bar(labels, [v * 100 for v in long_run_vols], color="#C44E52")
    ax2.set_yscale("log")
    ax2.set_title("Implied Long-Run Volatility (%, log scale)")
    fig.suptitle("Higher-Order GARCH Becomes Unstable on Limited Data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[안내] 차트를 '{out_path}'에 저장했습니다.")


if __name__ == "__main__":
    # 예시: 평온한 구간(150일) 이후 변동성이 급등하는 구간(15일)이 이어지는 가짜 데이터
    calm = np.random.default_rng(1).normal(0, 0.005, 150)
    shock = np.random.default_rng(2).normal(0, 0.04, 15)
    sample_returns = np.concatenate([calm, shock])
    portfolio_value = 1_000_000

    print(f"{'모형':<12} {'잔류율(Σα+Σβ)':<16} {'다음날 변동성':<14} {'장기평균 변동성':<16} {'VaR(95%)'}")
    labels, persistences, long_run_vols = [], [], []
    for p, q in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        params = fit_garch(sample_returns, p=p, q=q)
        vol = garch_volatility(sample_returns, params, p=p, q=q)
        lr_vol = np.sqrt(long_run_variance(params))
        var_95 = garch_var(sample_returns, 0.95, portfolio_value, params, p=p, q=q)
        persistence = params["alphas"].sum() + params["betas"].sum()
        print(
            f"GARCH({p},{q})  {persistence:<16.4f} {vol:<14.4%} {lr_vol:<16.4%} {var_95:,.2f}"
        )
        labels.append(f"GARCH({p},{q})")
        persistences.append(persistence)
        long_run_vols.append(lr_vol)

    plot_order_comparison(labels, persistences, long_run_vols)

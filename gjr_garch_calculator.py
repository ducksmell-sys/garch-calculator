"""GJR-GARCH(1,1) volatility & VaR calculator: captures the leverage effect
(negative shocks increase volatility more than positive shocks of the same size).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from garch_calculator import fit_garch, garch_volatility

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def _variance_path(returns, omega, alpha, gamma, beta):
    """GJR-GARCH(1,1) 파라미터가 주어졌을 때, 전체 기간의 분산 경로를 계산."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        indicator = 1.0 if returns[t - 1] < 0 else 0.0  # 어제 하락했으면 1
        sigma2[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + gamma * indicator * returns[t - 1] ** 2
            + beta * sigma2[t - 1]
        )
    return sigma2


def _negative_log_likelihood(params, returns):
    """최적화 목적함수: 음의 로그우도."""
    omega, alpha, gamma, beta = params
    # 정상성 조건: 정규분포 가정하에 E[I]=0.5이므로 alpha + beta + gamma/2 < 1
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + gamma) < 0:
        return 1e10
    if alpha + beta + gamma * 0.5 >= 1:
        return 1e10
    sigma2 = _variance_path(returns, omega, alpha, gamma, beta)
    if np.any(sigma2 <= 0):
        return 1e10
    log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -np.sum(log_likelihood)


def fit_gjr_garch(returns):
    """과거 수익률에 가장 잘 맞는 GJR-GARCH(1,1) 파라미터를 추정."""
    returns = np.asarray(returns)
    sample_var = np.var(returns)
    initial_guess = [sample_var * 0.05, 0.03, 0.05, 0.85]
    bounds = [(1e-10, None), (0, 1), (-1, 1), (0, 1)]

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


def gjr_volatility(returns, params=None):
    """GJR-GARCH(1,1)로 추정한 다음 날(t+1)의 변동성."""
    returns = np.asarray(returns)
    if params is None:
        params = fit_gjr_garch(returns)
    sigma2 = _variance_path(returns, params["omega"], params["alpha"], params["gamma"], params["beta"])
    indicator = 1.0 if returns[-1] < 0 else 0.0
    next_sigma2 = (
        params["omega"]
        + params["alpha"] * returns[-1] ** 2
        + params["gamma"] * indicator * returns[-1] ** 2
        + params["beta"] * sigma2[-1]
    )
    return np.sqrt(next_sigma2)


def gjr_var(returns, confidence=0.95, portfolio_value=1.0, params=None):
    """GJR-GARCH(1,1) 기반 변동성을 이용한 모수적 VaR (평균은 0으로 가정)."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    sigma = gjr_volatility(returns, params)
    z = Z_SCORES[confidence]
    return z * sigma * portfolio_value


def plot_news_impact_curve(gjr_params, garch_params, sigma2_prev, out_path="gjr_news_impact_curve.png"):
    """뉴스 충격 곡선: 어제의 충격 크기(가로축)에 따른 내일의 변동성(세로축).

    어제까지의 변동성(sigma2_prev)을 고정하고 충격만 바꿔가며 두 모델을 비교한다.
    """
    shocks = np.linspace(-0.06, 0.06, 241)
    garch_vol = np.sqrt(
        garch_params["omega"] + garch_params["alpha"] * shocks**2 + garch_params["beta"] * sigma2_prev
    )
    indicator = (shocks < 0).astype(float)
    gjr_vol = np.sqrt(
        gjr_params["omega"]
        + (gjr_params["alpha"] + gjr_params["gamma"] * indicator) * shocks**2
        + gjr_params["beta"] * sigma2_prev
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(shocks * 100, garch_vol * 100, color="#4C72B0", linewidth=2, label="GARCH(1,1): symmetric")
    ax.plot(shocks * 100, gjr_vol * 100, color="#C44E52", linewidth=2, label="GJR-GARCH: asymmetric")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Yesterday's return shock (%)")
    ax.set_ylabel("Next-day volatility forecast (%)")
    ax.set_title("News Impact Curve: Down Shocks Raise Volatility More Than Up Shocks")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[안내] 차트를 '{out_path}'에 저장했습니다.")


if __name__ == "__main__":
    # 비대칭성이 내재된 가짜 데이터 생성: 하락일 다음날 변동성이 더 크게 반응하도록 설계
    rng = np.random.default_rng(7)
    n = 500
    returns = np.empty(n)
    sigma = 0.015
    for t in range(n):
        returns[t] = sigma * rng.normal()
        indicator = 1.0 if returns[t] < 0 else 0.0
        sigma = np.sqrt(0.00001 + (0.03 + 0.08 * indicator) * returns[t] ** 2 + 0.85 * sigma**2)

    gjr_params = fit_gjr_garch(returns)
    print(
        f"GJR-GARCH 추정 파라미터: omega={gjr_params['omega']:.8f}, "
        f"alpha={gjr_params['alpha']:.4f}, gamma={gjr_params['gamma']:.4f}, "
        f"beta={gjr_params['beta']:.4f}"
    )
    print(f"(gamma가 양수면 하락 충격이 변동성을 더 크게 키운다는 뜻 - 레버리지 효과 존재)\n")

    # 핵심 데모: 똑같은 크기(3%)의 상승/하락 충격이 "다음날 변동성 예측"에 미치는 영향 비교
    shock_size = 0.03
    history = returns[:-1]
    returns_after_up = np.append(history, shock_size)
    returns_after_down = np.append(history, -shock_size)

    gjr_vol_up = gjr_volatility(returns_after_up, gjr_params)
    gjr_vol_down = gjr_volatility(returns_after_down, gjr_params)

    garch_params = fit_garch(returns)
    garch_vol_up = garch_volatility(returns_after_up, garch_params)
    garch_vol_down = garch_volatility(returns_after_down, garch_params)

    print(f"{'모델':<12} {'+3% 충격 후 변동성':<20} {'-3% 충격 후 변동성':<20} {'차이'}")
    print(
        f"{'GARCH(1,1)':<12} {garch_vol_up:<20.4%} {garch_vol_down:<20.4%} "
        f"{'0.00%p (대칭이라 완전히 동일)'}"
    )
    print(
        f"{'GJR-GARCH':<12} {gjr_vol_up:<20.4%} {gjr_vol_down:<20.4%} "
        f"{(gjr_vol_down - gjr_vol_up) * 100:.2f}%p (하락이 변동성을 더 키움)"
    )

    sigma2_prev = _variance_path(
        returns, gjr_params["omega"], gjr_params["alpha"], gjr_params["gamma"], gjr_params["beta"]
    )[-1]
    plot_news_impact_curve(gjr_params, garch_params, sigma2_prev)

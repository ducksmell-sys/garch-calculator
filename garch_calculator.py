"""GARCH(1,1) volatility & VaR calculator: parameters estimated by maximum likelihood (MLE)."""

import numpy as np
from scipy.optimize import minimize

# 표준정규분포의 단측 z값 (신뢰수준별)
Z_SCORES = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}


def _variance_path(returns, omega, alpha, beta):
    """GARCH(1,1) 파라미터가 주어졌을 때, 전체 기간의 분산 경로를 계산."""
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)  # 초기값: 전체 표본 분산으로 시작
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _negative_log_likelihood(params, returns):
    """최적화 목적함수: 음의 로그우도 (이 값을 최소화 = 우도를 최대화)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10  # 정상성(stationarity) 조건 위반 시 큰 페널티
    sigma2 = _variance_path(returns, omega, alpha, beta)
    log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -np.sum(log_likelihood)


def fit_garch(returns):
    """과거 수익률에 가장 잘 맞는 GARCH(1,1) 파라미터(omega, alpha, beta)를 추정."""
    returns = np.asarray(returns)
    sample_var = np.var(returns)
    initial_guess = [sample_var * 0.05, 0.05, 0.90]  # 실무에서 흔히 쓰는 초기값
    bounds = [(1e-10, None), (0, 1), (0, 1)]

    result = minimize(
        _negative_log_likelihood,
        initial_guess,
        args=(returns,),
        method="L-BFGS-B",
        bounds=bounds,
    )
    omega, alpha, beta = result.x
    return {"omega": omega, "alpha": alpha, "beta": beta, "converged": result.success}


def garch_volatility(returns, params=None):
    """GARCH(1,1)로 추정한 다음 날(t+1)의 변동성."""
    returns = np.asarray(returns)
    if params is None:
        params = fit_garch(returns)
    sigma2 = _variance_path(returns, params["omega"], params["alpha"], params["beta"])
    next_sigma2 = (
        params["omega"] + params["alpha"] * returns[-1] ** 2 + params["beta"] * sigma2[-1]
    )
    return np.sqrt(next_sigma2)


def garch_var(returns, confidence=0.95, portfolio_value=1.0, params=None):
    """GARCH(1,1) 기반 변동성을 이용한 모수적 VaR (평균은 0으로 가정)."""
    if confidence not in Z_SCORES:
        raise ValueError(f"confidence must be one of {list(Z_SCORES)}")
    sigma = garch_volatility(returns, params)
    z = Z_SCORES[confidence]
    return z * sigma * portfolio_value


if __name__ == "__main__":
    # 예시: 평온한 구간(100일) 이후 변동성이 급등하는 구간(10일)이 이어지는 가짜 데이터
    calm = np.random.default_rng(1).normal(0, 0.005, 100)   # 평상시: 변동성 0.5%
    shock = np.random.default_rng(2).normal(0, 0.04, 10)    # 위기 발생: 변동성 4%
    sample_returns = np.concatenate([calm, shock])
    portfolio_value = 1_000_000

    params = fit_garch(sample_returns)
    print(f"추정된 GARCH(1,1) 파라미터: omega={params['omega']:.8f}, "
          f"alpha={params['alpha']:.4f}, beta={params['beta']:.4f}")
    print(f"alpha + beta = {params['alpha'] + params['beta']:.4f} (1에 가까울수록 충격이 오래 지속)\n")

    simple_std = sample_returns.std(ddof=1)
    g_std = garch_volatility(sample_returns, params)

    print(f"단순 표준편차 (전체 기간 동일 가중치): {simple_std:.4%}")
    print(f"GARCH(1,1) 다음날 예측 변동성:        {g_std:.4%}\n")

    for confidence in (0.90, 0.95, 0.99):
        simple_var = Z_SCORES[confidence] * simple_std * portfolio_value
        g_var = garch_var(sample_returns, confidence, portfolio_value, params)
        print(
            f"신뢰수준 {confidence:.0%} | 단순 VaR: {simple_var:,.2f} | "
            f"GARCH VaR: {g_var:,.2f}"
        )

# EGARCH(1,1) Calculator — Exponential GARCH

Another approach to asymmetric volatility, alongside
[GJR-GARCH](README_gjr.md): EGARCH (Nelson, 1991) models **log-variance** instead of
variance directly, and captures the leverage effect through a term that responds
smoothly to the sign of the shock, rather than GJR-GARCH's on/off indicator.

## The Model

```
ln(σ²_t) = ω + α·[|z_(t-1)| - E|z|] + γ·z_(t-1) + β·ln(σ²_(t-1))

z_(t-1) = ε_(t-1) / σ_(t-1)        (standardized shock)
E|z| = √(2/π)  ≈ 0.7979             (expected |z| under z ~ N(0,1))
```

## Why Model Log-Variance?

Standard GARCH and GJR-GARCH need constraints like `ω, α, β ≥ 0` just to guarantee
`σ²_t` stays positive. EGARCH sidesteps this entirely: whatever value `ln(σ²_t)` takes,
exponentiating it (`σ²_t = exp(ln σ²_t)`) is **always positive**, so no sign
restrictions are needed on the parameters at all.

## How the Asymmetry Term Works

```
γ · z_(t-1)
```

Unlike GJR-GARCH's indicator function (which switches discretely between two regimes),
this term responds **continuously** to the sign and size of the standardized shock:

- `z < 0` (a down day) and `γ < 0`  → `γ·z > 0` → log-variance pushed **up**
- `z > 0` (an up day) and `γ < 0`  → `γ·z < 0` → log-variance pushed **down**

A statistically significant negative `γ` is evidence of the leverage effect, same
interpretation as a positive `γ` in GJR-GARCH.

## Files

| File | Description |
|---|---|
| `egarch_calculator.py` | `fit_egarch`, `egarch_volatility`, `egarch_var` — demo compares GARCH(1,1), GJR-GARCH, and EGARCH side by side on identical shocks |

## Usage

```bash
pip install numpy scipy
python egarch_calculator.py
```

```python
from egarch_calculator import fit_egarch, egarch_volatility, egarch_var

params = fit_egarch(returns)                  # {"omega", "alpha", "gamma", "beta"}
egarch_volatility(returns, params)             # tomorrow's forecasted volatility
egarch_var(returns, confidence=0.95, portfolio_value=1_000_000, params=params)
```

## Sample Output

Using the same synthetic data (with a built-in asymmetric response) as the GJR-GARCH
demo, all three models are compared on an identical +3% vs. -3% shock:

```
EGARCH estimated parameters: omega=-1.640945, alpha=0.1413, gamma=-0.1225, beta=0.8221
(a negative gamma means down-shocks amplify log-variance more - leverage effect present)

모델           +3% 충격 후 변동성         -3% 충격 후 변동성         차이(%p)
GARCH(1,1)   1.4901%              1.4901%              +0.00
GJR-GARCH    1.4180%              1.5687%              +0.15
EGARCH       1.1946%              1.5861%              +0.39
```

All three models agree on the direction (down-shocks → higher forecasted volatility),
but EGARCH's smooth, continuous asymmetry term produces the most pronounced spread
between the two scenarios in this data — illustrating that GJR-GARCH and EGARCH, while
solving the same problem, don't always agree on *how much* asymmetry is present.

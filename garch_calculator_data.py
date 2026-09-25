"""Load real market data (ticker or CSV) and compute GARCH(1,1) volatility & VaR.

Usage:
    python garch_calculator_data.py --ticker AAPL
    python garch_calculator_data.py --ticker BTC-USD --confidence 0.99
    python garch_calculator_data.py --csv prices.csv --price-col Close
"""

import argparse
from datetime import date, timedelta

import pandas as pd

from garch_calculator import Z_SCORES, fit_garch, garch_var, garch_volatility


def load_returns_from_ticker(ticker, start=None, end=None):
    """yfinance로 티커의 주가를 받아와 일별 수익률(Series)로 변환."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"'{ticker}' 데이터를 받아오지 못했습니다. 티커를 확인해주세요.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # 티커를 여러 개 받은 경우 대비
        close = close.iloc[:, 0]
    returns = close.pct_change().dropna()
    returns.index.name = "Date"
    return returns


def load_returns_from_csv(csv_path, date_col="Date", price_col="Close"):
    """CSV에서 가격을 읽어 일별 수익률(Series)로 변환."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df[price_col].pct_change().dropna()


def main():
    parser = argparse.ArgumentParser(description="GARCH(1,1) volatility/VaR calculator on real market data")
    parser.add_argument("--ticker", help="야후 파이낸스 티커 (예: AAPL, 005930.KS, BTC-USD)")
    parser.add_argument("--start", default=str(date.today() - timedelta(days=365 * 2)), help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD, 기본값: 오늘)")
    parser.add_argument("--csv", help="주가 CSV 파일 경로 (티커 대신 사용)")
    parser.add_argument("--price-col", default="Close", help="가격 컬럼명")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    parser.add_argument("--portfolio-value", type=float, default=1_000_000)
    args = parser.parse_args()

    if args.ticker:
        print(f"[안내] yfinance로 '{args.ticker}' 데이터를 받아옵니다 ({args.start} ~ {args.end or '오늘'}).")
        returns = load_returns_from_ticker(args.ticker, args.start, args.end)
    elif args.csv:
        returns = load_returns_from_csv(args.csv, price_col=args.price_col)
    else:
        parser.error("--ticker 또는 --csv 중 하나는 반드시 지정해야 합니다.")

    print(f"데이터 기간: {returns.index.min().date()} ~ {returns.index.max().date()} "
          f"({len(returns)}개 거래일)")

    params = fit_garch(returns.values)
    print(f"추정된 GARCH(1,1) 파라미터: omega={params['omega']:.8f}, "
          f"alpha={params['alpha']:.4f}, beta={params['beta']:.4f}")
    print(f"alpha + beta = {params['alpha'] + params['beta']:.4f}\n")

    simple_std = returns.std()
    g_std = garch_volatility(returns.values, params)
    print(f"단순 표준편차: {simple_std:.4%} | GARCH(1,1) 다음날 예측 변동성: {g_std:.4%}\n")

    simple_var = Z_SCORES[args.confidence] * simple_std * args.portfolio_value
    g_var = garch_var(returns.values, args.confidence, args.portfolio_value, params)

    print(f"신뢰수준 {args.confidence:.0%}, 포트폴리오 {args.portfolio_value:,.0f} 기준 1일 VaR")
    print(f"  단순 VaR : {simple_var:,.2f}")
    print(f"  GARCH VaR: {g_var:,.2f}")


if __name__ == "__main__":
    main()

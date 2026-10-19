# scalperbot-mt5

ScalperBot — Python scalper for MetaTrader5 (Exness) implementing EMA+RSI+MACD signals, daily profit/loss enforcement, 5-minute break after losing trades, CSV logging, and improved pip/lot calculations.

## Requirements

- Python 3.8+
- MetaTrader 5 desktop terminal installed and running (same machine)
- MetaTrader5 Python package, pandas, numpy

Install

```
pip install MetaTrader5 pandas numpy
```

## Usage

1. Open MetaTrader5 terminal, enable algorithmic trading, and login to your Exness account (demo recommended).
2. Clone repo:

```
git clone https://github.com/TBAN009/scalperbot-mt5.git
cd scalperbot-mt5
```

3. Edit `ScalperBot.py` config at the top (SYMBOL, TARGET_DAILY_PROFIT, MIN_LOT, etc.) if needed.
4. Run:

```
python ScalperBot.py
```

## Notes & warnings

- Highly recommended to test on demo accounts only. Targets like 20–30 USD/day from a $10 balance are extremely high-risk.
- Broker specifics (min lot, volume step, filling type) can require parameter adjustments.
- The bot logs order placements and closed deals to `trades.csv`.

If you want, I can add a status overlay, unit tests, or a GitHub Actions workflow to lint the code.

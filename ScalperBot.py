#!/usr/bin/env python3
"""
ScalperBot.py

Improved Python scalper for MT5 (Exness) with:
- EMA crossover + RSI + MACD entries
- Daily profit target & daily loss percent stop
- Max trades per day
- 5-minute pause after any losing closed deal
- Dynamic lot sizing using precise pip value from symbol specs
- Logging of order placements and closed deals to CSV (trades.csv)
- Uses MetaTrader5 Python package
"""

import MetaTrader5 as mt5
import time
import math
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import csv
import os
import sys

# ----------------------------- CONFIG ---------------------------------
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
EMA_FAST = 5
EMA_SLOW = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

TP_PIPS = 10.0
SL_PIPS = 20.0

TARGET_DAILY_PROFIT = 25.0
DAILY_LOSS_PERCENT = 50.0
MAX_TRADES_PER_DAY = 15

USE_DYNAMIC_LOT = True
RISK_PERCENT_PER_TRADE = 2.0
MIN_LOT = 0.01
MAX_LOT = 1.0

SLIPPAGE = 5
MAGIC = 20261019

POLL_INTERVAL = 1.0
LOGFILE = "scalperbot.log"
TRADES_CSV = "trades.csv"

# ----------------------------- LOGGING --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ScalperBot")

# ----------------------------- STATE ----------------------------------
day_start_time = None
day_start_balance = None
trades_today = 0
pause_until = None
trading_paused_for_day = False
last_deal_time = None  # datetime for incremental deal scanning

# --------------------------- UTIL FUNCTIONS ----------------------------

def mt5_initialize():
    if not mt5.initialize():
        logger.error("mt5.initialize() failed. Error: %s", mt5.last_error())
        return False
    logger.info("MetaTrader5 initialized")
    return True


def mt5_shutdown():
    try:
        mt5.shutdown()
        logger.info("MetaTrader5 shutdown")
    except Exception:
        pass


def get_account_info():
    info = mt5.account_info()
    if info is None:
        logger.error("Failed to get account info: %s", mt5.last_error())
        return None
    return info


def get_symbol_info(sym):
    info = mt5.symbol_info(sym)
    if info is None:
        logger.error("Symbol %s not found", sym)
        return None
    if not info.visible:
        mt5.symbol_select(sym, True)
    return info


def get_rates(symbol, timeframe, bars=300):
    utc_from = datetime.now(timezone.utc) - timedelta(days=2)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)
    if rates is None or len(rates) == 0:
        logger.error("copy_rates_from returned empty for %s", symbol)
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# indicators
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def pip_price(symbol_info):
    """
    Returns price difference that corresponds to 1 pip (in price units) and pip value per 1 lot (money).
    Uses SYMBOL trade tick values when available for precise pip value.
    """
    point = symbol_info.point
    digits = symbol_info.digits
    pip_multiplier = 10 if digits > 3 else 1
    one_pip_in_price = point * pip_multiplier

    # try to compute pip value per 1 lot using trade_tick_value/trade_tick_size
    tick_value = getattr(symbol_info, "trade_tick_value", None)
    tick_size = getattr(symbol_info, "trade_tick_size", None)
    contract_size = getattr(symbol_info, "trade_contract_size", None)
    if tick_value and tick_size and tick_size != 0:
        value_per_price_unit_per_lot = tick_value / tick_size  # monetary per 1 price unit per 1 lot
        pip_value_per_lot = value_per_price_unit_per_lot * one_pip_in_price
        return one_pip_in_price, pip_value_per_lot
    # fallback: estimate pip value (very rough) for majors: assume $10 per pip per 1 lot
    fallback_pip_value = 10.0
    return one_pip_in_price, fallback_pip_value

def compute_lot(symbol_info, sl_pips):
    if not USE_DYNAMIC_LOT:
        return clamp(MIN_LOT, MIN_LOT, MAX_LOT)
    acc = get_account_info()
    if acc is None:
        return MIN_LOT
    balance = acc.balance
    money_risk = balance * (RISK_PERCENT_PER_TRADE / 100.0)
    _, pip_value_per_lot = pip_price(symbol_info)
    if pip_value_per_lot <= 0 or sl_pips <= 0:
        return MIN_LOT
    lot = money_risk / (sl_pips * pip_value_per_lot)
    # normalize to allowed step
    step = symbol_info.volume_step if symbol_info.volume_step and symbol_info.volume_step > 0 else 0.01
    minv = symbol_info.volume_min if symbol_info.volume_min and symbol_info.volume_min > 0 else MIN_LOT
    maxv = symbol_info.volume_max if symbol_info.volume_max and symbol_info.volume_max > 0 else MAX_LOT
    lot = max(minv, min(maxv, math.floor(lot / step) * step))
    lot = round(lot, 2)
    lot = clamp(lot, MIN_LOT, MAX_LOT)
    return lot

def clamp(x, a, b):
    return max(a, min(b, x))

def has_active_position_by_magic(symbol, magic):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return False
    for p in pos:
        try:
            if int(p.magic) == magic:
                return True
        except Exception:
            continue
    return False

def deals_since(ts, symbol=None):
    from_time = datetime.fromtimestamp(ts, tz=timezone.utc)
    to_time = datetime.now(timezone.utc)
    try:
        deals = mt5.history_deals_get(from_time, to_time, group=symbol)
        return deals if deals is not None else []
    except Exception:
        return []

def realized_profit_today(symbol, day_start_ts):
    deals = deals_since(day_start_ts, symbol)
    total = 0.0
    for d in deals:
        try:
            # d.order -> order ticket, get order to inspect magic
            order_ticket = int(d.order) if hasattr(d, "order") else None
            if order_ticket:
                order = mt5.order_get(ticket=order_ticket)
                if order and int(getattr(order, "magic", 0)) == MAGIC:
                    total += float(d.profit)
        except Exception:
            continue
    return float(total)

def append_trade_csv(row):
    is_new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp","event","ticket","order","symbol","type","volume","price","sl","tp","profit","balance","comment"])
        writer.writerow(row)

def send_market_order(symbol_info, buy, lot, sl_pips, tp_pips):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        logger.error("No tick for symbol %s", SYMBOL)
        return False, None
    price = tick.ask if buy else tick.bid
    one_pip_price, _ = pip_price(symbol_info)
    sl_price = price - sl_pips * one_pip_price if buy else price + sl_pips * one_pip_price
    tp_price = price + tp_pips * one_pip_price if buy else price - tp_pips * one_pip_price

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
        "price": float(price),
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": SLIPPAGE,
        "magic": MAGIC,
        "comment": "ScalperBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result is None:
        logger.error("order_send returned None")
        return False, None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.warning("Order failed retcode=%s comment=%s", result.retcode, result.comment)
        # still log attempt
        append_trade_csv([datetime.now(timezone.utc).isoformat(), "order_failed", result.order, getattr(result, "order", ""), SYMBOL, "BUY" if buy else "SELL",
                         lot, price, sl_price, tp_price, None, get_account_info().balance if get_account_info() else None, result.comment])
        return False, result
    logger.info("Order placed: %s %s lot=%.2f price=%.5f ticket=%s", "BUY" if buy else "SELL", SYMBOL, lot, price, result.order)
    append_trade_csv([datetime.now(timezone.utc).isoformat(), "order_placed", result.order, getattr(result, "order", ""), SYMBOL, "BUY" if buy else "SELL",
                     lot, price, sl_price, tp_price, None, get_account_info().balance if get_account_info() else None, result.comment])
    return True, result

# --------------------------- SIGNAL LOGIC ------------------------------

def get_signal(df):
    if df is None or len(df) < max(EMA_SLOW, RSI_PERIOD, MACD_SLOW) + 5:
        return 0
    close = df['close']
    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)
    rsi_v = rsi(close, RSI_PERIOD)
    _, _, macd_hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    idx_now = -1
    idx_prev = -2

    fast_prev = ema_fast.iloc[idx_prev]
    slow_prev = ema_slow.iloc[idx_prev]
    fast_now = ema_fast.iloc[idx_now]
    slow_now = ema_slow.iloc[idx_now]

    hist_prev = macd_hist.iloc[idx_prev]
    hist_now = macd_hist.iloc[idx_now]

    rsi_now = rsi_v.iloc[idx_now]

    if (fast_prev <= slow_prev) and (fast_now > slow_now) and (hist_now > hist_prev) and (rsi_now < 70.0):
        return 1
    if (fast_prev >= slow_prev) and (fast_now < slow_now) and (hist_now < hist_prev) and (rsi_now > 30.0):
        return -1
    return 0

# --------------------------- MAIN LOOP --------------------------------

def run():
    global day_start_time, day_start_balance, trades_today, pause_until, trading_paused_for_day, last_deal_time

    if not mt5_initialize():
        return
    acc = get_account_info()
    if acc is None:
        mt5_shutdown()
        return

    now = datetime.now(timezone.utc)
    day_start_time = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
    day_start_balance = acc.balance
    trades_today = 0
    pause_until = None
    trading_paused_for_day = False
    last_deal_time = datetime.now(timezone.utc) - timedelta(minutes=30)

    symbol_info = get_symbol_info(SYMBOL)
    if symbol_info is None:
        mt5_shutdown()
        return

    logger.info("Starting ScalperBot on %s (%s). Day start balance=%.2f", SYMBOL, TIMEFRAME, day_start_balance)
    logger.info("Targets: daily_profit=%.2f daily_loss_percent=%.1f%% max_trades=%d", TARGET_DAILY_PROFIT, DAILY_LOSS_PERCENT, MAX_TRADES_PER_DAY)

    try:
        while True:
            # day reset
            now = datetime.now(timezone.utc)
            today_start_ts = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
            if today_start_ts != day_start_time:
                day_start_time = today_start_ts
                acc_info = get_account_info()
                day_start_balance = acc_info.balance if acc_info else day_start_balance
                trades_today = 0
                pause_until = None
                trading_paused_for_day = False
                logger.info("New trading day. day_start_balance=%.2f", day_start_balance)

            # pauses
            if trading_paused_for_day:
                time.sleep(POLL_INTERVAL)
                continue
            if pause_until and datetime.now(timezone.utc) < pause_until:
                time.sleep(POLL_INTERVAL)
                continue
            else:
                pause_until = None

            acc_info = get_account_info()
            if acc_info is None:
                time.sleep(POLL_INTERVAL)
                continue
            current_balance = acc_info.balance

            # compute realized profit using order magic and deals
            realized = realized_profit_today(SYMBOL, day_start_time)

            # daily loss check
            loss_allowed_balance = day_start_balance * (1.0 - DAILY_LOSS_PERCENT/100.0)
            if current_balance <= loss_allowed_balance:
                trading_paused_for_day = True
                logger.warning("Daily loss threshold reached. Current balance=%.2f <= stop=%.2f. Pausing for the day.", current_balance, loss_allowed_balance)
                time.sleep(POLL_INTERVAL)
                continue
            if realized >= TARGET_DAILY_PROFIT:
                trading_paused_for_day = True
                logger.info("Daily profit target reached (realized=%.2f >= %.2f). Pausing for the day.", realized, TARGET_DAILY_PROFIT)
                time.sleep(POLL_INTERVAL)
                continue
            if trades_today >= MAX_TRADES_PER_DAY:
                trading_paused_for_day = True
                logger.info("Max trades per day reached (%d). Pausing for the day.", trades_today)
                time.sleep(POLL_INTERVAL)
                continue

            # check new deals since last_deal_time to detect losing closed deals and to log them
            try:
                deals = mt5.history_deals_get(last_deal_time, datetime.now(timezone.utc), group=SYMBOL)
                if deals:
                    for d in deals:
                        # log any deal associated to our orders (matching order.magic)
                        try:
                            order_ticket = int(d.order) if hasattr(d, "order") else None
                            if order_ticket:
                                order = mt5.order_get(ticket=order_ticket)
                                mag = int(getattr(order, "magic", 0)) if order else 0
                                if mag == MAGIC:
                                    append_trade_csv([datetime.now(timezone.utc).isoformat(), "deal", d.ticket if hasattr(d, "ticket") else "", d.order, d.symbol, d.type, d.volume, d.price, d.sl, d.tp, d.profit, get_account_info().balance if get_account_info() else None, getattr(d, "comment", "")])
                                    # if this deal had negative profit and is a closing deal, pause 5 minutes
                                    if float(d.profit) < 0:
                                        pause_until = datetime.now(timezone.utc) + timedelta(minutes=5)
                                        logger.warning("Detected losing deal (profit=%.2f). Pausing until %s", float(d.profit), pause_until.isoformat())
                        except Exception:
                            continue
                    last_deal_time = datetime.now(timezone.utc)
            except Exception:
                pass

            df = get_rates(SYMBOL, TIMEFRAME, bars=300)
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            signal = get_signal(df)
            if signal == 0:
                time.sleep(POLL_INTERVAL)
                continue

            if has_active_position_by_magic(SYMBOL, MAGIC):
                time.sleep(POLL_INTERVAL)
                continue

            lot = compute_lot(symbol_info, SL_PIPS)
            lot = clamp(lot, MIN_LOT, MAX_LOT)

            ok, res = send_market_order(symbol_info, buy=(signal==1), lot=lot, sl_pips=SL_PIPS, tp_pips=TP_PIPS)
            if ok:
                trades_today += 1
                logger.info("Trade executed. trades_today=%d", trades_today)
            else:
                logger.warning("Trade not executed. response=%s", res)

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
    except Exception as e:
        logger.exception("Error in main loop: %s", str(e))
    finally:
        mt5_shutdown()

if __name__ == "__main__":
    run()

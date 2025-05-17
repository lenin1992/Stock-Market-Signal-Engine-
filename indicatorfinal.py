import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
import vectorbt as vbt
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas_ta as ta
from rddit import fetch_reddit_posts, analyze_sentiment, aggregate_daily_sentiment
from ta.volatility import AverageTrueRange
from scipy.signal import argrelextrema

def download_data(symbol, start, end):
    file_path = f"{symbol}_data.csv"

    def clean_dataframe(df):
        df.columns = [str(col).strip().capitalize() for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        return df

    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Date' not in df.columns:
                raise ValueError("Missing 'Date' column in CSV. Likely corrupted.")
            df = clean_dataframe(df)
            return df
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"⚠️ Issue loading {file_path}: {e}")
        print("🔄 Redownloading fresh data...")
        if os.path.exists(file_path):
            os.remove(file_path)
        df = yf.download(symbol, start=start, end=end, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}.")
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        df.to_csv(file_path, index=False)
        df = clean_dataframe(df)
        return df

def calculate_rsi(data, period=14, use_ema=True):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    if use_ema:
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_supertrend(df, period=10, multiplier=3):
    # Convert data to numeric and round to 2 decimal places
    high = pd.to_numeric(df['High'], errors='coerce').round(2)
    low = pd.to_numeric(df['Low'], errors='coerce').round(2)
    close = pd.to_numeric(df['Close'], errors='coerce').round(2)

    # Calculate HL2 (average of high and low)
    hl2 = ((high + low) / 2).round(2)

    # Calculate ATR
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=period)
    atr = atr_indicator.average_true_range()  # This returns a pandas Series

    # Calculate the upper and lower bands
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Initialize SuperTrend and Direction series
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='object')

    # Initial values for the first row
    direction.iloc[0] = 'uptrend'
    supertrend.iloc[0] = lowerband.iloc[0]

    bullish_entries = 0
    bearish_entries = 0

    # Loop through the data to calculate the SuperTrend and Direction
    for i in range(1, len(df)):
        curr_close = close.iloc[i]
        prev_supertrend = supertrend.iloc[i - 1]
        prev_direction = direction.iloc[i - 1]
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]

        if prev_direction == 'uptrend':
            if curr_close < prev_supertrend:
                direction.iloc[i] = 'downtrend'
                supertrend.iloc[i] = curr_upper
                bearish_entries += 1
            else:
                direction.iloc[i] = 'uptrend'
                supertrend.iloc[i] = max(curr_lower, prev_supertrend)
        else:
            if curr_close > prev_supertrend:
                direction.iloc[i] = 'uptrend'
                supertrend.iloc[i] = curr_lower
                bullish_entries += 1
            else:
                direction.iloc[i] = 'downtrend'
                supertrend.iloc[i] = min(curr_upper, prev_supertrend)

    # Assign calculated values back to the DataFrame
    df['SuperTrend'] = supertrend.round(2)
    df['ST_Direction'] = direction.ffill()  # Forward fill the direction column
    df['UpperBand'] = upperband
    df['LowerBand'] = lowerband
    df['ATR'] = atr
    # Print summary of entries
    print(f"🔼 Bullish entries: {bullish_entries}")
    print(f"🔽 Bearish entries: {bearish_entries}")    
    return df

def calculate_obv(df):
    # Calculate On-Balance Volume (OBV)
    direction = np.sign(df['Close'].diff().fillna(0))
    volume_signed = df['Volume'] * direction
    obv = volume_signed.cumsum()
    obv.iloc[0] = 0  # Start from 0
    df['OBV'] = obv
    return df

def detect_obv_divergence(df, lookahead=5, stop_loss_pct=0.02):
    order = 5
    df['Price_Lows'] = np.nan
    df['Price_Highs'] = np.nan
    df['OBV_Lows'] = np.nan
    df['OBV_Highs'] = np.nan

    df['OBV Divergence'] = None
    df['OBV EntryPrice'] = np.nan
    df['OBV StopLoss'] = np.nan
    df['OBV Success'] = None

    # Identify local minima and maxima
    price_lows_idx = argrelextrema(df['Close'].values, np.less_equal, order=order)[0]
    price_highs_idx = argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]
    obv_lows_idx = argrelextrema(df['OBV'].values, np.less_equal, order=order)[0]
    obv_highs_idx = argrelextrema(df['OBV'].values, np.greater_equal, order=order)[0]

    # Fill extrema columns
    df.loc[df.index[price_lows_idx], 'Price_Lows'] = df['Close'].iloc[price_lows_idx]
    df.loc[df.index[price_highs_idx], 'Price_Highs'] = df['Close'].iloc[price_highs_idx]
    df.loc[df.index[obv_lows_idx], 'OBV_Lows'] = df['OBV'].iloc[obv_lows_idx]
    df.loc[df.index[obv_highs_idx], 'OBV_Highs'] = df['OBV'].iloc[obv_highs_idx]

    # --- Bullish Divergence ---
    for i in range(1, min(len(price_lows_idx), len(obv_lows_idx))):
        p1, p2 = price_lows_idx[i - 1], price_lows_idx[i]
        o1, o2 = obv_lows_idx[i - 1], obv_lows_idx[i]

        if df['Close'].iloc[p2] < df['Close'].iloc[p1] and df['OBV'].iloc[o2] > df['OBV'].iloc[o1]:
            entry_price = df['Close'].iloc[p2]
            stop_loss = entry_price * (1 - stop_loss_pct)
            future_prices = df['Close'].iloc[p2 + 1:p2 + 1 + lookahead]
            success = all(future_prices > entry_price)

            df.loc[df.index[p2], 'OBV Divergence'] = 'Bullish'
            df.loc[df.index[p2], 'OBV EntryPrice'] = entry_price
            df.loc[df.index[p2], 'OBV StopLoss'] = stop_loss
            df.loc[df.index[p2], 'OBV Success'] = success

    # --- Bearish Divergence ---
    for i in range(1, min(len(price_highs_idx), len(obv_highs_idx))):
        p1, p2 = price_highs_idx[i - 1], price_highs_idx[i]
        o1, o2 = obv_highs_idx[i - 1], obv_highs_idx[i]

        if df['Close'].iloc[p2] > df['Close'].iloc[p1] and df['OBV'].iloc[o2] < df['OBV'].iloc[o1]:
            entry_price = df['Close'].iloc[p2]
            stop_loss = entry_price * (1 + stop_loss_pct)
            future_prices = df['Close'].iloc[p2 + 1:p2 + 1 + lookahead]
            success = all(future_prices < entry_price)

            df.loc[df.index[p2], 'OBV Divergence'] = 'Bearish'
            df.loc[df.index[p2], 'OBV EntryPrice'] = entry_price
            df.loc[df.index[p2], 'OBV StopLoss'] = stop_loss
            df.loc[df.index[p2], 'OBV Success'] = success

    # Final counts
    print("✅ Bullish Divergences Detected:", (df['OBV Divergence'] == 'Bullish').sum())
    print("✅ Bearish Divergences Detected:", (df['OBV Divergence'] == 'Bearish').sum())
    print("🏁 Bullish Successes:", ((df['OBV Divergence'] == 'Bullish') & (df['OBV Success'] == True)).sum())
    print("🏁 Bearish Successes:", ((df['OBV Divergence'] == 'Bearish') & (df['OBV Success'] == True)).sum())

    print("Divergence signals updated in DataFrame.")

    return df
def validate_rsi_divergence_with_trailing_stop(
    df: pd.DataFrame, window: int = 5, lookahead: int = 3, stop_loss_pct: float = 0.02
) -> pd.DataFrame:
    
    pnl_data = []
    bullish_success = 0
    bearish_success = 0

    for i in range(window, len(df) - lookahead):
        price_now = df['Close'].iloc[i]
        price_past = df['Close'].iloc[i - window]
        rsi_now = df['RSI'].iloc[i]
        rsi_past = df['RSI'].iloc[i - window]
        date_now = df.index[i]

        entry_type = None
        stop_loss_price = None
        signal_validated = False
        entry_price = None
        exit_price = None
        pnl = 0
        stoploss_hit = False
        position_status = "Closed"

        # Bullish Divergence
        if price_now < price_past and rsi_now > rsi_past:
            entry_price = price_now
            stop_loss_price = round(entry_price * (1 - stop_loss_pct), 2)
            entry_type = 'bullish'

            for j in range(1, lookahead + 1):
                future_price = df['Close'].iloc[i + j]
                if future_price <= stop_loss_price:
                    stoploss_hit = True
                    exit_price = future_price
                    pnl = exit_price - entry_price
                    break
                if df['BullDiv'].iloc[i + j]:
                    break

            if not stoploss_hit:
                signal_validated = True
                exit_price = df['Close'].iloc[i + lookahead]
                pnl = exit_price - entry_price
                bullish_success += 1
                position_status = "Closed"

        # Bearish Divergence
        elif price_now > price_past and rsi_now < rsi_past:
            entry_price = price_now
            stop_loss_price = round(entry_price * (1 + stop_loss_pct), 2)
            entry_type = 'bearish'

            for j in range(1, lookahead + 1):
                future_price = df['Close'].iloc[i + j]
                if future_price >= stop_loss_price:
                    stoploss_hit = True
                    exit_price = future_price
                    pnl = entry_price - exit_price
                    break
                if df['BearDiv'].iloc[i + j]:
                    break

            if not stoploss_hit:
                signal_validated = True
                exit_price = df['Close'].iloc[i + lookahead]
                pnl = entry_price - exit_price
                bearish_success += 1
                position_status = "Closed"

        # Save row-wise data
        pnl_data.append({
            'RSI Entry Price': entry_price if signal_validated else None,
            'RSI Exit Price': exit_price if signal_validated else None,
            'RSI P&L': pnl if signal_validated else 0,
            'RSI Entry Type': entry_type if signal_validated else None,
            'RSI Stoploss Hit': stoploss_hit if signal_validated else None,
            'RSI Position Status': position_status if signal_validated else None,
            'RSI Signal Validated': signal_validated
        })

    # Pad beginning to match df length
    pad_rows = len(df) - len(pnl_data)
    for _ in range(pad_rows):
        pnl_data.insert(0, {
            'RSI Entry Price': None,
            'RSI Exit Price': None,
            'RSI P&L': 0,
            'RSI Entry Type': None,
            'RSI Stoploss Hit': None,
            'RSI Position Status': None,
            'RSI Signal Validated': False
        })

    pnl_df = pd.DataFrame(pnl_data, index=df.index)

    # Merge into df
    df = pd.concat([df, pnl_df], axis=1)

    print("✅ RSI Divergence signal validation completed.")
    print(f"📈 Bullish Validated: {bullish_success}")
    print(f"📉 Bearish Validated: {bearish_success}")
    return df
def calculate_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2
) -> pd.DataFrame:
    df = df.copy()  # avoid mutating the original
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Std'] = df['Close'].rolling(window=window).std()
    df['UpperBand'] = df['SMA'] + (df['Std'] * num_std)
    df['LowerBand'] = df['SMA'] - (df['Std'] * num_std)
    return df

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df

def calculate_rsi_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'RSI' not in df.columns:
        raise ValueError("RSI column not found in DataFrame. Please calculate RSI first.")
    df['RSI_diff'] = df['RSI'].diff()
    return df

def calculate_rsi_rolling_mean(
    df: pd.DataFrame, period: int = 14, rolling_window: int = 3
) -> pd.DataFrame:
    df = df.copy()
    df['RSI_rolling_mean'] = df['RSI'].rolling(window=rolling_window).mean()
    return df

def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df['Momentum'] = df['Close'].diff(periods=period)
    return df
def validate_bollinger_signals_with_sl(
    df: pd.DataFrame,
    lookahead=7,
    stop_loss_pct=0.03,
    target_pct=0.05
):
    df = df.copy()
    
    # Initialize columns
    df['Bollinger Signal'] = None
    df['Bollinger EntryPrice'] = np.nan
    df['Bollinger TargetPrice'] = np.nan
    df['Bollinger StopLoss'] = np.nan
    df['Bollinger Result'] = None

    # Signal Detection
    bullish_entries = df[
        (df['Low'] < df['LowerBand']) &
        (df['Close'] > df['LowerBand']) &
        (df['Close'] > df['Open'])
    ].index.tolist()

    bearish_entries = df[
        (df['High'] > df['UpperBand']) &
        (df['Close'] < df['UpperBand']) &
        (df['Close'] < df['Open'])
    ].index.tolist()

    print(f"Confirmed Bullish Entries: {len(bullish_entries)}")
    print(f"Confirmed Bearish Entries: {len(bearish_entries)}")

    bullish_success = bearish_success = bullish_fail = bearish_fail = 0

    for signal_dates, direction in [(bullish_entries, 'bullish'), (bearish_entries, 'bearish')]:
        for date in signal_dates:
            if date not in df.index:
                continue

            entry_idx = df.index.get_loc(date)
            if entry_idx + 1 >= len(df):
                continue

            future = df.iloc[entry_idx + 1: entry_idx + 1 + lookahead]
            if future.empty:
                continue

            entry_price = df.loc[date, 'Close']
            if direction == 'bullish':
                target_price = entry_price * (1 + target_pct)
                stop_price = entry_price * (1 - stop_loss_pct)
                hit_target = future['Close'][future['Close'] >= target_price].index.min()
                hit_stop = future['Close'][future['Close'] <= stop_price].index.min()
            else:
                target_price = entry_price * (1 - target_pct)
                stop_price = entry_price * (1 + stop_loss_pct)
                hit_target = future['Close'][future['Close'] <= target_price].index.min()
                hit_stop = future['Close'][future['Close'] >= stop_price].index.min()

            # Default result
            result = "No Trigger"
            if pd.isna(hit_stop) and not pd.isna(hit_target):
                result = "Success"
                if direction == 'bullish':
                    bullish_success += 1
                else:
                    bearish_success += 1
            elif not pd.isna(hit_stop) and (pd.isna(hit_target) or hit_stop <= hit_target):
                result = "Fail"
                if direction == 'bullish':
                    bullish_fail += 1
                else:
                    bearish_fail += 1

            df.loc[date, 'Bollinger Signal'] = direction.capitalize()
            df.loc[date, 'Bollinger EntryPrice'] = entry_price
            df.loc[date, 'Bollinger TargetPrice'] = target_price
            df.loc[date, 'Bollinger StopLoss'] = stop_price
            df.loc[date, 'Bollinger Result'] = result

    print(f"\nBollinger Band Bullish Success: {bullish_success} | Fail: {bullish_fail}")
    print(f"Bollinger Band Bearish Success: {bearish_success} | Fail: {bearish_fail}")
    print("✅ Bollinger Band signal validation completed and stored in DataFrame.")
    
    return df

def detect_rsi_divergence(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:    
    df = df.copy()
    if 'RSI' not in df.columns:
        raise ValueError("Missing 'RSI' column. Please compute RSI first.")
    df['BullDiv'] = False
    df['BearDiv'] = False
    # Loop through to detect divergences
    for i in range(window, len(df)):
        price_now = df['Close'].iloc[i]
        price_past = df['Close'].iloc[i - window]
        rsi_now = df['RSI'].iloc[i]
        rsi_past = df['RSI'].iloc[i - window]
        # Bullish Divergence: Price down, RSI up
        if price_now < price_past and rsi_now > rsi_past:
            df.at[df.index[i], 'BullDiv'] = True
        # Bearish Divergence: Price up, RSI down
        elif price_now > price_past and rsi_now < rsi_past:
            df.at[df.index[i], 'BearDiv'] = True
    # After loop, filter for detected divergences
    bullish_divergences = df[df['BullDiv'] == True]
    bearish_divergences = df[df['BearDiv'] == True]
    # Print results if any divergences are detected
    '''if not bullish_divergences.empty:
        print("\nBullish Divergence (BullDiv):")
        print(bullish_divergences[['Close', 'RSI', 'BullDiv']])

    if not bearish_divergences.empty:
        print("\nBearish Divergence (BearDiv):")
        print(bearish_divergences[['Close', 'RSI', 'BearDiv']])

    if bullish_divergences.empty and bearish_divergences.empty:
        print("\nNo RSI Divergences Detected.")'''
    print(f"\nBullish Divergences (BullDiv): {len(bullish_divergences)}")
    print(f"Bearish Divergences (BearDiv): {len(bearish_divergences)}")
    return df

def find_macd_divergence(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.001,
    lookahead: int = 5,
    stop_loss_pct: float = 0.02
) -> tuple[list, list, pd.DataFrame]:
    df = df.copy()

    bullish_divergences = []
    bearish_divergences = []

    divergence_labels = ['None'] * len(df)
    stop_losses = [None] * len(df)
    results = ['No Trigger'] * len(df)

    for i in range(window, len(df) - lookahead):
        price_now = df['Close'].iloc[i]
        macd_now = df['MACD'].iloc[i]
        future_prices = df['Close'].iloc[i + 1: i + 1 + lookahead]

        # --- Bullish Divergence ---
        price_prev = df['Close'].iloc[i - window:i].min()
        macd_prev = df['MACD'].iloc[i - window:i][df['Close'].iloc[i - window:i].idxmin()]
        if price_now < price_prev * (1 - threshold) and macd_now > macd_prev * (1 + threshold):
            bullish_divergences.append(df.index[i])
            divergence_labels[i] = 'Bullish'
            stop_losses[i] = round(price_now * (1 - stop_loss_pct), 2)

            returns = (future_prices - price_now) / price_now
            if (returns > stop_loss_pct).any():
                results[i] = 'Success'
            elif (returns < -stop_loss_pct).any():
                results[i] = 'Failure'

        # --- Bearish Divergence ---
        price_prev_high = df['Close'].iloc[i - window:i].max()
        macd_prev_high = df['MACD'].iloc[i - window:i][df['Close'].iloc[i - window:i].idxmax()]
        if price_now > price_prev_high * (1 + threshold) and macd_now < macd_prev_high * (1 - threshold):
            bearish_divergences.append(df.index[i])
            divergence_labels[i] = 'Bearish'
            stop_losses[i] = round(price_now * (1 + stop_loss_pct), 2)

            returns = (price_now - future_prices) / price_now
            if (returns > stop_loss_pct).any():
                results[i] = 'Success'
            elif (returns < -stop_loss_pct).any():
                results[i] = 'Failure'

    # Add to dataframe
    df['MACD Divergence'] = divergence_labels
    df['MACD StopLoss'] = stop_losses
    df['MACD Result'] = results

    return bullish_divergences, bearish_divergences, df

def validate_macd_signals(
    df: pd.DataFrame,
    stop_loss_percentage: float = 0.02,
    success_threshold: float = 0.02,
    lookahead_period: int = 10
) -> pd.DataFrame:

    if 'MACD_Hist' not in df.columns:
        df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['MACD_bullish_cross'] = (
        (df['MACD'] > df['Signal']) &
        (df['MACD'].shift(1) <= df['Signal'].shift(1))
    )

    df['MACD_bearish_cross'] = (
        (df['MACD'] < df['Signal']) &
        (df['MACD'].shift(1) >= df['Signal'].shift(1))
    )

    bullish_entries = df[df['MACD_bullish_cross']].index.tolist()
    bearish_entries = df[df['MACD_bearish_cross']].index.tolist()

    bullish_success = bearish_success = bullish_fail = bearish_fail = 0

    macd_signal_data = []

    for i in range(len(df)):
        signal = None
        stop_loss = None
        result = None

        if df.index[i] in bullish_entries:
            entry_price = df.iloc[i]['Close']
            stop_loss = round(entry_price * (1 - stop_loss_percentage), 2)
            max_future_price = df['Close'].iloc[i+1:i+1+lookahead_period].max() if i + lookahead_period < len(df) else None

            if max_future_price is not None:
                if max_future_price >= entry_price * (1 + success_threshold):
                    result = "Success"
                    bullish_success += 1
                else:
                    result = "Fail"
                    bullish_fail += 1
            else:
                result = "No Data"

            signal = "Bullish Crossover"

        elif df.index[i] in bearish_entries:
            entry_price = df.iloc[i]['Close']
            stop_loss = round(entry_price * (1 + stop_loss_percentage), 2)
            min_future_price = df['Close'].iloc[i+1:i+1+lookahead_period].min() if i + lookahead_period < len(df) else None

            if min_future_price is not None:
                if min_future_price <= entry_price * (1 - success_threshold):
                    result = "Success"
                    bearish_success += 1
                else:
                    result = "Fail"
                    bearish_fail += 1
            else:
                result = "No Data"

            signal = "Bearish Crossover"

        macd_signal_data.append({
            'MACD Signal': signal,
            'MACD StopLoss': stop_loss,
            'MACD Result': result
        })

    # Pad if needed
    while len(macd_signal_data) < len(df):
        macd_signal_data.append({
            'MACD Signal': None,
            'MACD StopLoss': None,
            'MACD Result': None
        })

    macd_signal_df = pd.DataFrame(macd_signal_data, index=df.index)
    df = pd.concat([df, macd_signal_df], axis=1)

    # Optional: Detect divergence (assuming this function exists in your code)
    bullish_divs, bearish_divs, df = find_macd_divergence(df)

    print(f"\n📊 MACD Crossover Stats:")
    print(f"  Bullish Crossovers: {len(bullish_entries)} | Success: {bullish_success} | Fail: {bullish_fail}")
    print(f"  Bearish Crossovers: {len(bearish_entries)} | Success: {bearish_success} | Fail: {bearish_fail}")

    print(f"\n🔍 MACD Divergences:")
    print(f"  Bullish Divergences: {len(bullish_divs)}")
    print(f"  Bearish Divergences: {len(bearish_divs)}")

    return df, bullish_entries, bearish_entries

def validate_rsi_diff(df):
    rsi_diff = df['RSI'].diff()
    return rsi_diff

def validate_rsi_rolling_signals_combined(
    df,
    lookahead=5,
    stop_loss_pct=0.02,
    signal_type="crossover"  # "crossover" or "simple"
):
    df = df.copy()
    
    # Initialize columns
    df['RSI Signal'] = None
    df['RSI EntryPrice'] = np.nan
    df['RSI StopLoss'] = np.nan
    df['RSI Result'] = None

    bullish_success = bearish_success = bullish_fail = bearish_fail = 0

    for i in range(1, len(df) - lookahead):
        rsi_now = df['RSI'].iloc[i]
        rsi_prev = df['RSI'].iloc[i - 1]
        mean_now = df['RSI_rolling_mean'].iloc[i]
        mean_prev = df['RSI_rolling_mean'].iloc[i - 1]
        price = df['Close'].iloc[i]
        future_prices = df['Close'].iloc[i + 1: i + 1 + lookahead]

        is_bullish = is_bearish = False
        if signal_type == "crossover":
            is_bullish = (rsi_prev <= mean_prev) and (rsi_now > mean_now)
            is_bearish = (rsi_prev >= mean_prev) and (rsi_now < mean_now)
        elif signal_type == "simple":
            is_bullish = rsi_now > mean_now
            is_bearish = rsi_now < mean_now

        if is_bullish:
            sl = price * (1 - stop_loss_pct)
            success = (future_prices > price).any() and (future_prices > sl).all()
            df.at[df.index[i], 'RSI Signal'] = 'Bullish'
            df.at[df.index[i], 'RSI EntryPrice'] = price
            df.at[df.index[i], 'RSI StopLoss'] = sl
            df.at[df.index[i], 'RSI Result'] = 'Success' if success else 'Failure'
            if success:
                bullish_success += 1
            else:
                bullish_fail += 1

        if is_bearish:
            sl = price * (1 + stop_loss_pct)
            success = (future_prices < price).any() and (future_prices < sl).all()
            df.at[df.index[i], 'RSI Signal'] = 'Bearish'
            df.at[df.index[i], 'RSI EntryPrice'] = price
            df.at[df.index[i], 'RSI StopLoss'] = sl
            df.at[df.index[i], 'RSI Result'] = 'Success' if success else 'Failure'
            if success:
                bearish_success += 1
            else:
                bearish_fail += 1

    print(f"Bullish - Success: {bullish_success}, Failure: {bullish_fail}")
    print(f"Bearish - Success: {bearish_success}, Failure: {bearish_fail}")
    print("✅ RSI signal validation completed and stored in DataFrame.")

    return df

def validate_momentum_crossover_signals(
    df: pd.DataFrame,
    lookahead: int = 5,
    stop_loss_pct: float = 0.02
) -> pd.DataFrame:

    # Identify bullish momentum crossovers
    df['Momentum_Crossover'] = (df['Momentum'] > 0) & (df['Momentum'].shift(1) <= 0)
    signal_dates = df.index[df['Momentum_Crossover']]
    print(f"📈 Momentum crossover signals generated: {len(signal_dates)}")

    # List to store row-wise data
    momentum_signal_data = []

    for i in range(len(df)):
        signal = None
        stop_loss = None
        success = None

        if df.index[i] in signal_dates:
            entry_price = df.iloc[i]['Close']
            stop_loss = round(entry_price * (1 - stop_loss_pct), 2)

            if i + lookahead < len(df):
                future_prices = df['Close'].iloc[i+1:i+1+lookahead]
                price_changes = (future_prices - entry_price) / entry_price
                success = (price_changes > stop_loss_pct).any()
            else:
                success = None  # Not enough data ahead

            signal = "Bullish Momentum Crossover"

        momentum_signal_data.append({
            'Momentum Signal': signal,
            'Momentum StopLoss': stop_loss,
            'Momentum Success': success
        })

    # Pad if needed
    while len(momentum_signal_data) < len(df):
        momentum_signal_data.append({
            'Momentum Signal': None,
            'Momentum StopLoss': None,
            'Momentum Success': None
        })

    momentum_df = pd.DataFrame(momentum_signal_data, index=df.index)
    df = pd.concat([df, momentum_df], axis=1)

    return df

def zscore_rsi_threshold(rsi_series, window=14, upper=1.0, lower=-1.0):
    if rsi_series.isnull().all():
        raise ValueError("RSI series is empty or all NaNs.")
    rolling_mean = rsi_series.rolling(window=window).mean()
    rolling_std = rsi_series.rolling(window=window).std()
    zscore = (rsi_series - rolling_mean) / rolling_std
    return zscore, upper, lower

def multi_timeframe_trend(df, period=10, multiplier=3):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    df_weekly = df.resample('W-FRI').agg({
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    if len(df_weekly) < period:
        raise ValueError("Not enough data to calculate the weekly trend.")
    
    df_weekly = calculate_supertrend(df_weekly, period, multiplier)

    # Extract trend and stoploss (Supertrend line) from weekly
    weekly_trend = df_weekly['ST_Direction']
    weekly_stoploss = df_weekly['SuperTrend']

    # Align weekly trend and stoploss into daily dataframe
    trend_daily = pd.Series(index=df.index, dtype='object')
    stoploss_daily = pd.Series(index=df.index, dtype='float')

    for i in range(1, len(weekly_trend)):
        start = weekly_trend.index[i - 1] + pd.Timedelta(days=1)
        end = weekly_trend.index[i]
        trend_daily.loc[start:end] = weekly_trend.iloc[i - 1]
        stoploss_daily.loc[start:end] = weekly_stoploss.iloc[i - 1]

    trend_daily = trend_daily.ffill().reindex(df.index, method='ffill').fillna('downtrend')
    stoploss_daily = stoploss_daily.ffill().reindex(df.index, method='ffill')

    # Add to dataframe
    df['Weekly_Trend'] = trend_daily.replace({'uptrend': 'uptrend', 'downtrend': 'downtrend'})
    df['Weekly_Stoploss'] = stoploss_daily

    return df

def validate_zscore_signals_with_stoploss(df, lookahead=5, stop_loss_pct=0.02):
    bullish_signal = (df['RSI_Z'] < -1)  # Bullish: Z-Score < -1 (oversold)
    bearish_signal = (df['RSI_Z'] > 1)   # Bearish: Z-Score > 1 (overbought)
    bull_dates = df.index[bullish_signal]
    bear_dates = df.index[bearish_signal]
    bull_success, bull_fail = [], []
    bear_success, bear_fail = [], []
    # Track crossovers
    total_crossovers = 0
    # Process bullish signals (Z-Score < -1)
    for date in bull_dates:
        try:
            idx = df.index.get_loc(date)
            entry_price = df['Close'].iloc[idx]
            future = df['Close'].iloc[idx + 1 : idx + 1 + lookahead]
            returns = (future - entry_price) / entry_price
            if (returns > stop_loss_pct).any():  # Success: Price rises above stop-loss threshold
                bull_success.append(date)
            elif (returns < -stop_loss_pct).any():  # Failure: Price drops below stop-loss threshold
                bull_fail.append(date)            
            # Count the crossover
            total_crossovers += 1
        except:
            continue
    # Process bearish signals (Z-Score > 1)
    for date in bear_dates:
        try:
            idx = df.index.get_loc(date)
            entry_price = df['Close'].iloc[idx]
            future = df['Close'].iloc[idx + 1 : idx + 1 + lookahead]
            returns = (entry_price - future) / entry_price
            if (returns > stop_loss_pct).any():  # Success: Price drops below stop-loss threshold
                bear_success.append(date)
            elif (returns < -stop_loss_pct).any():  # Failure: Price rises above stop-loss threshold
                bear_fail.append(date)            
            # Count the crossover
            total_crossovers += 1
        except:
            continue
    # Print results like TradingView signals
    print(f"Total Z-Score RSI Crossovers: {total_crossovers}")
    print(f"Z-Score RSI Bullish Success Count: {len(bull_success)} | Fail Count: {len(bull_fail)}")
    print(f"Z-Score RSI Bearish Success Count: {len(bear_success)} | Fail Count: {len(bear_fail)}")
    return bull_success, bull_fail, bear_success, bear_fail

def validate_supertrend_signals_with_trailing_stop(df):
    st_bull_success = st_bull_fail = 0
    st_bear_success = st_bear_fail = 0
    st_flip_count = 0
    total_pnl = 0
    in_position = False
    entry_type = None
    entry_price = None
    stoploss_price = None
    previous_direction = None

    pnl_data = []

    for i in range(1, len(df)):
        curr_close = df.iloc[i]['Close']
        curr_direction = df.iloc[i]['ST_Direction']
        curr_st = df.iloc[i]['SuperTrend']
        direction = 'uptrend' if curr_direction == 'uptrend' else 'downtrend'

        # Trend flip
        if previous_direction is not None and direction != previous_direction:
            st_flip_count += 1
            if in_position:
                pnl = curr_close - entry_price if entry_type == 'bullish' else entry_price - curr_close
                total_pnl += pnl
                st_bull_fail += int(entry_type == 'bullish')
                st_bear_fail += int(entry_type == 'bearish')
                in_position = False

            # New entry
            entry_price = curr_close
            stoploss_price = curr_st
            entry_type = 'bullish' if direction == 'uptrend' else 'bearish'
            in_position = True
            if entry_type == 'bullish':
                st_bull_success += 1
            else:
                st_bear_success += 1

        elif in_position:
            if entry_type == 'bullish':
                if curr_st > stoploss_price:
                    stoploss_price = curr_st
                if curr_close < stoploss_price:
                    pnl = curr_close - entry_price
                    total_pnl += pnl
                    st_bull_fail += 1
                    in_position = False
            elif entry_type == 'bearish':
                if curr_st < stoploss_price:
                    stoploss_price = curr_st
                if curr_close > stoploss_price:
                    pnl = entry_price - curr_close
                    total_pnl += pnl
                    st_bear_fail += 1
                    in_position = False

        # Save row-wise data
        pnl_data.append({
            'ST Entry Price': entry_price if in_position else None,
            'ST Exit Price': curr_close,
            'ST P&L': (curr_close - entry_price if entry_type == 'bullish' else entry_price - curr_close) if in_position else 0,
            'ST Entry Type': entry_type if in_position else None,
            'ST Stoploss Hit': False if in_position else None,
            'ST Position Status': 'Open' if in_position else 'Closed'
        })

        previous_direction = direction

    # Pad pnl_data to match df length
    pnl_data.insert(0, {
        'ST Entry Price': None,
        'ST Exit Price': None,
        'ST P&L': 0,
        'ST Entry Type': None,
        'ST Stoploss Hit': None,
        'ST  Position Status': None
    })

    pnl_df = pd.DataFrame(pnl_data, index=df.index)

    # Merge new columns into original df
    df = pd.concat([df, pnl_df], axis=1)

    print(f"\n🔼 Bullish entries: {st_bull_success}, Stoploss hits: {st_bull_fail}")
    print(f"🔽 Bearish entries: {st_bear_success}, Stoploss hits: {st_bear_fail}")
    print(f"↻ Trend flips: {st_flip_count}")
    print(f"💰 Total P&L: {total_pnl:.2f}")
    return df

def validate_weekly_trend_signals_with_stoploss(df):
    bull_success = bull_fail = bear_success = bear_fail = 0
    trend_flip_count = 0
    total_pnl = 0
    in_position = False
    entry_type = None
    entry_price = None
    stoploss_price = None
    previous_weekly_trend = None

    pnl_data = []

    for i in range(1, len(df)):
        curr_close = df.iloc[i]['Close']
        curr_weekly_trend = df.iloc[i]['Weekly_Trend']
        curr_weekly_stoploss = df.iloc[i]['Weekly_Stoploss']
        pnl = 0.0
        stoploss_hit = None
        position_status = 'Closed'
        entry_price_display = None
        entry_type_display = None

        # Check if in position
        if in_position:
            # Mark-to-market P&L
            pnl = curr_close - entry_price if entry_type == 'bullish' else entry_price - curr_close
            entry_price_display = entry_price
            entry_type_display = entry_type
            position_status = 'Open'
            stoploss_hit = False

            # Stoploss condition
            if entry_type == 'bullish' and curr_close < stoploss_price:
                pnl = curr_close - entry_price
                total_pnl += pnl
                bull_fail += 1
                in_position = False
                stoploss_hit = True
                position_status = 'Closed'
            elif entry_type == 'bearish' and curr_close > stoploss_price:
                pnl = entry_price - curr_close
                total_pnl += pnl
                bear_fail += 1
                in_position = False
                stoploss_hit = True
                position_status = 'Closed'

            # Update trailing stoploss
            if in_position:
                if entry_type == 'bullish' and curr_weekly_stoploss > stoploss_price:
                    stoploss_price = curr_weekly_stoploss
                elif entry_type == 'bearish' and curr_weekly_stoploss < stoploss_price:
                    stoploss_price = curr_weekly_stoploss

        # Detect trend flip (after stoploss handling above)
        if previous_weekly_trend is not None and curr_weekly_trend != previous_weekly_trend:
            trend_flip_count += 1

            # Close previous position if still open
            if in_position:
                pnl = curr_close - entry_price if entry_type == 'bullish' else entry_price - curr_close
                total_pnl += pnl
                position_status = 'Closed'
                stoploss_hit = False
                in_position = False

            # Start new position
            entry_price = curr_close
            stoploss_price = curr_weekly_stoploss
            entry_type = 'bullish' if curr_weekly_trend == 'uptrend' else 'bearish'
            in_position = True
            entry_type_display = entry_type
            entry_price_display = entry_price
            position_status = 'Open'
            stoploss_hit = False
            pnl = 0.0

            if entry_type == 'bullish':
                bull_success += 1
            else:
                bear_success += 1

        # Record daily row
        pnl_data.append({
            'Entry Price': entry_price_display,
            'Exit Price': curr_close,
            'P&L': pnl,
            'Entry Type': entry_type_display,
            'Stoploss Hit': stoploss_hit,
            'Position Status': position_status
        })

        previous_weekly_trend = curr_weekly_trend

    # Pad the beginning row (first candle not processed)
    first_row = {
        'Entry Price': None,
        'Exit Price': None,
        'P&L': 0.0,
        'Entry Type': None,
        'Stoploss Hit': None,
        'Position Status': None
    }
    pnl_data.insert(0, first_row)

    # Merge to df
    pnl_df = pd.DataFrame(pnl_data, index=df.index)
    df = pd.concat([df, pnl_df], axis=1)

    print(f"\n📈 Bullish entries: {bull_success}, Stoploss hits: {bull_fail}")
    print(f"📉 Bearish entries: {bear_success}, Stoploss hits: {bear_fail}")
    print(f"🔁 Weekly Trend flips: {trend_flip_count}")
    print(f"💰 Total P&L: {total_pnl:.2f}")

    return df
def validate_obv_signals(df, ma_period=14, lookahead=5, stop_loss_pct=0.02):
    if 'OBV' not in df.columns:
        raise ValueError("OBV column missing from DataFrame")

    df = df.copy()
    df['OBV_MA'] = df['OBV'].rolling(ma_period).mean()
    df['OBV_Crossover'] = df['OBV'] - df['OBV_MA']

    df['OBV Signal'] = None
    df['OBV EntryPrice'] = np.nan
    df['OBV StopLoss'] = np.nan
    df['OBV Success'] = None

    bullish_cross = (df['OBV_Crossover'] > 0) & (df['OBV_Crossover'].shift(1) <= 0)
    bearish_cross = (df['OBV_Crossover'] < 0) & (df['OBV_Crossover'].shift(1) >= 0)

    seen_dates = set()

    for signal_type, signal_mask in [('Bullish Crossover', bullish_cross), ('Bearish Crossover', bearish_cross)]:
        for signal_date in df.index[signal_mask]:
            pos = df.index.get_loc(signal_date)
            entry_price = df['Close'].iloc[pos]
            stop_loss = (
                entry_price * (1 - stop_loss_pct)
                if signal_type == 'Bullish Crossover'
                else entry_price * (1 + stop_loss_pct)
            )

            for offset in range(1, lookahead + 1):
                if pos + offset >= len(df):
                    break
                target_date = df.index[pos + offset]
                if target_date in seen_dates:
                    continue

                target_price = df['Close'].iloc[pos + offset]

                success = (
                    (target_price - entry_price) / entry_price > stop_loss_pct
                    if signal_type == 'Bullish Crossover'
                    else (entry_price - target_price) / entry_price > stop_loss_pct
                )

                df.loc[signal_date, 'OBV Signal'] = signal_type
                df.loc[signal_date, 'OBV EntryPrice'] = entry_price
                df.loc[signal_date, 'OBV StopLoss'] = stop_loss
                df.loc[signal_date, 'OBV Success'] = success

                seen_dates.add(target_date)
                break  # Only one validation per signal

    print("✅ OBV crossover signals validated and stored in DataFrame.")
    print("Bullish Crossovers:", (df['OBV Signal'] == 'Bullish Crossover').sum())
    print("Bearish Crossovers:", (df['OBV Signal'] == 'Bearish Crossover').sum())
    print("Bullish Successes:", ((df['OBV Signal'] == 'Bullish Crossover') & (df['OBV Success'] == True)).sum())
    print("Bearish Successes:", ((df['OBV Signal'] == 'Bearish Crossover') & (df['OBV Success'] == True)).sum())

    return df
def train_model(df, threshold=0.6, n_splits=5, verbose=True):    
    required_features = [
        'RSI', 'OBV', 'RSI_Overbought', 'RSI_Oversold', 'ST_Direction',
        'Weekly_Trend', 'Momentum', 'RSI_rolling_mean', 'RSI_diff',
        'MACD', 'Signal', 'UpperBand', 'LowerBand' , 'RSI_zscore', 'rsi_divergence',
        'daily_sentiment'
    ]    
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    df = df.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(subset=required_features + ['Target'], inplace=True)
    X = df[required_features].astype(float)
    y = df['Target']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)
    accs = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
    if verbose:
        print("\n[TimeSeriesSplit Accuracy]")
        print([f"Fold {i+1}: {acc:.4f}" for i, acc in enumerate(accs)])
        print(f"Average Accuracy: {np.mean(accs):.4f}")
    proba = model.predict_proba(X)[:, 1]
    signals = pd.Series((proba > threshold).astype(int), index=df.index)
    return model, np.mean(accs), signals

def process_stock(symbol, start, end, verbose=True):
    df = download_data(symbol, start, end)
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"No valid data for {symbol} between {start} and {end}")
    df.dropna(inplace=True)
    # Check if the CSV file exists
    if os.path.exists('reddit_stock_posts.csv'):
        print("Loading Reddit posts from 'reddit_stock_posts.csv'...")
        reddit_df = pd.read_csv('reddit_stock_posts.csv')
    else:
        try:
            # Fetch reddit data if CSV does not exist
            print("Fetching Reddit posts from r/stocks...")
            reddit_df = fetch_reddit_posts(subreddit="stocks", limit=100, days=7)
            if not reddit_df.empty:
                reddit_df.to_csv('reddit_stock_posts.csv', index=False)
                print("Reddit posts saved to 'reddit_stock_posts.csv'")
            else:
                print("Warning: No Reddit posts retrieved")
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            print("Continuing without sentiment analysis")
            reddit_df = pd.DataFrame()
    if not reddit_df.empty:
        # Calculate sentiment
        reddit_df = analyze_sentiment(reddit_df)
        sentiment_df = aggregate_daily_sentiment(reddit_df)        
        # Merge with your dataframe
        df.reset_index(inplace=True)
        df['date'] = pd.to_datetime(df['Date'].dt.date)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])        
        df = pd.merge(df, sentiment_df, on='date', how='left')
        df['daily_sentiment'] = df['daily_sentiment'].fillna(0)        
        df.set_index('Date', inplace=True)
        df.drop('date', axis=1, inplace=True)
# === Indicator Calculations ===
    df['RSI'] = calculate_rsi(df)
    df = calculate_rsi_rolling_mean(df)
    df = calculate_rsi_diff(df)
    df = calculate_momentum(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_supertrend(df)  # ✅ Assign whole updated df
    df = calculate_obv(df)
    df = validate_supertrend_signals_with_trailing_stop(df)
    df['ST_Direction'] = df['ST_Direction'].ffill()
    weekly_data = multi_timeframe_trend(df)
    df['Weekly_Trend'] = weekly_data['Weekly_Trend']
    df['Weekly_Stoploss'] = weekly_data['Weekly_Stoploss']

# Entry crossovers based on actual trend flips
    df['Bullish_ST_Entry'] = (df['ST_Direction'] == 'uptrend') & (df['ST_Direction'].shift(1) == 'downtrend')
    df['Bearish_ST_Entry'] = (df['ST_Direction'] == 'downtrend') & (df['ST_Direction'].shift(1) == 'uptrend')
    df['Bullish_Entry'] = df['Bullish_ST_Entry'] & (df['Weekly_Trend'] == 'uptrend')
    df['Bearish_Entry'] = df['Bearish_ST_Entry'] & (df['Weekly_Trend'] == 'downtrend')
# Trend context flags
    df['Daily_Trend'] = df['ST_Direction']  # keep as string
    df['MT_Bullish'] = (df['ST_Direction'] == 'uptrend') & (df['Weekly_Trend'] == 'uptrend')
    df['MT_Bearish'] = (df['ST_Direction'] == 'downtrend') & (df['Weekly_Trend'] == 'downtrend')
    df = validate_weekly_trend_signals_with_stoploss(df)
# === Z-score RSI ===
    df['RSI_Z'], upper_rsi, lower_rsi = zscore_rsi_threshold(df['RSI'])
    df['RSI_Overbought'] = df['RSI_Z'] > upper_rsi
    df['RSI_Oversold'] = df['RSI_Z'] < lower_rsi
# === RSI Divergence Detection ===
    df = detect_rsi_divergence(df)
    df = validate_rsi_divergence_with_trailing_stop(df, window=5)
    #bullish_hits, bearish_hits, bullish_entries, bearish_entries, all_signals_df, _ = validate_rsi_divergence(df, window=5)
# === OBV Signal Validation ===
    df=detect_obv_divergence(df, lookahead=5, stop_loss_pct=0.02)
    df= validate_obv_signals(df)
# === Other Indicator Validations ===
    df = validate_rsi_rolling_signals_combined(
    df,
    lookahead=5,
    stop_loss_pct=0.02,
    signal_type="crossover"  # or "simple"
)
    df = validate_momentum_crossover_signals(df)
    df, bullish_entries, bearish_entries = validate_macd_signals(df)
    signal_indices = bullish_entries + bearish_entries
    df_results = validate_bollinger_signals_with_sl(df)
    z_bull, z_bear, z_bull_dates, z_bear_dates = validate_zscore_signals_with_stoploss(df)
# === Stop-Loss Based Validations ===
    success_dates = df_results.loc[df_results['Bollinger Result'] == 'Success'].index
    df['bollinger_signal'] = df.index.isin(pd.to_datetime(success_dates))

    df['macd_signal'] = df.index.isin(signal_indices)
    #df['bollinger_signal'] = df.index.isin(bb_bull_success + bb_bear_success)
    df['momentum_signal'] = df['Momentum'] > 0
    df['rolling_mean_signal'] = df['RSI'] > df['RSI_rolling_mean']
# Divergence flags
    df['rsi_divergence'] = 0
    df.loc[df['BullDiv'], 'rsi_divergence'] = 1
    df.loc[df['BearDiv'], 'rsi_divergence'] = -1
    # === RSI Z-Score ===
    df['RSI_zscore'] = (df['RSI'] - df['RSI'].rolling(window=14).mean()) / df['RSI'].rolling(window=14).std()
    # === Composite Trading Signals ===
    # Define each individual buy condition
    # Define the conditions for buy signals
    buy_conditions = [
        df['ST_Direction'] == 'uptrend',  # or any condition based on your column
        df['Weekly_Trend'] == 'uptrend',  # replace as per actual data
        df['OBV'] > df['OBV'].rolling(14).mean(),
        df['momentum_signal'] == 1,  # Assuming 1 means True, adjust if needed
        df['macd_signal'] == 1,      # Assuming 1 means True, adjust if needed
        df['bollinger_signal'] == 1, # Assuming 1 means True, adjust if needed
        df['RSI_zscore'] < -1,
        df['rsi_divergence'] == 1,   # Uncomment if needed
        df['daily_sentiment'] > 0    # Uncomment if sentiment is being used
]
# Convert conditions to DataFrame
    buy_signals_df = pd.DataFrame(buy_conditions).T
# Note: This only applies if the columns have such values to replace.
    buy_signals_df = buy_signals_df.replace({'uptrend': 1, 'downtrend': 0})
# Ensure all values are converted to integers (True -> 1, False -> 0)
    buy_signals_df = buy_signals_df.astype(int)
# Now calculate the buy score as the sum of True conditions per row
    df['buy_score'] = buy_signals_df.sum(axis=1)
# Set threshold: how many conditions must be True to signal a Buy
    BUY_THRESHOLD = 6  # Change this to 5 or 7 depending on desired strictness
# Final composite buy signal
    df['Composite_Buy_Signal'] = df['buy_score'] >= BUY_THRESHOLD
# For example, if 'uptrend' means True and others mean False
    df['ST_Direction'] = df['ST_Direction'] == 'uptrend'  # True if 'uptrend', False otherwise
    df['Weekly_Trend'] = df['Weekly_Trend'] == 'uptrend'  # True if 'uptrend', False otherwise
# Define the sell conditions
    sell_conditions = [
        ~df['ST_Direction'],  # Negate the boolean value
    ~df['Weekly_Trend'],  # Negate the boolean value
    df['OBV'] < df['OBV'].rolling(14).mean(),
    ~df['momentum_signal'],  # Assuming 1 is True, negate it
    ~df['macd_signal'],      # Assuming 1 is True, negate it
    ~df['bollinger_signal'], # Assuming 1 is True, negate it
    df['RSI_zscore'] > 1,
    df['rsi_divergence'] == -1,  # Optional
    df['daily_sentiment'] < 0   # Optional
]
# Convert conditions into a DataFrame for easier row-wise operations
    sell_signals_df = pd.concat(sell_conditions, axis=1)
# Sum the True conditions to get the sell score
    df['sell_score'] = sell_signals_df.sum(axis=1)
    SELL_THRESHOLD = 6
    df['Composite_Sell_Signal'] = df['sell_score'] >= SELL_THRESHOLD
    # === Train ML Model ===
    model, accuracy, signals = train_model(df, threshold=0.6, verbose=verbose)
    df['ML_Buy_Prediction'] = signals
    summary = f"""
    Stock: {symbol}
    Period: {start} to {end}
    Model Accuracy: {accuracy:.4f}   
    === Strategy Summary ===
    Composite Buy Signals: {df['Composite_Buy_Signal'].sum()}
    Composite Sell Signals: {df['Composite_Sell_Signal'].sum()}
    ML-Based Buy Predictions (>60% Confidence): {df['ML_Buy_Prediction'].sum()}
    === Recent Snapshot ===
    MACD Histogram (Latest): {df['MACD_Hist'].iloc[-1]:.2f}
    RSI Daily Change: {df['RSI'].iloc[-1] - df['RSI'].iloc[-2]:.2f}
    Bollinger Band Width: {(df['UpperBand'].iloc[-1] - df['LowerBand'].iloc[-1]):.2f}
    20d vs 50d Rolling Mean Spread: {(df['Close'].rolling(20).mean().iloc[-1] - df['Close'].rolling(50).mean().iloc[-1]):.2f}
    Momentum (10-day): {(df['Close'].iloc[-1] - df['Close'].shift(10).iloc[-1]):.2f}
    """
    df.to_csv(f"{symbol}_analysis.csv")

    return summary.strip()

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., AAPL, SBIN.NS): ").strip().upper()
    start = input("Enter start date (YYYY-MM-DD): ").strip()
    end = input("Enter end date (YYYY-MM-DD or leave blank for today): ").strip()
    if not end:
        end = datetime.today().strftime('%Y-%m-%d')
    print("\n🔍 Running analysis...\n")
    result = process_stock(symbol, start, end)
    print(result)
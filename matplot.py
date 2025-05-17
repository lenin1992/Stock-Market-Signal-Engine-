import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from indicatorfinal import process_stock

def load_data(symbol):
    df = pd.read_csv(f"{symbol}_analysis.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

def plot_dashboard(df, symbol, indicators):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[[{"type": "candlestick"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )

    # === Row 1: Candlestick ===
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)

    # === SuperTrend ===
    if 'superrend' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name='SuperTrend',
                                 line=dict(color='magenta')), row=1, col=1)

    # === Bollinger Bands ===
    if 'bollinger' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='BB Upper',
                                 line=dict(color='cyan', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='BB Lower',
                                 line=dict(color='cyan', dash='dot')), row=1, col=1)

    # === Composite Buy/Sell ===
    if 'composite' in indicators:
        buy = df[df['Composite_Buy_Signal']]
        sell = df[df['Composite_Sell_Signal']]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers',
                                 marker=dict(symbol='triangle-up', color='green', size=10),
                                 name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers',
                                 marker=dict(symbol='triangle-down', color='red', size=10),
                                 name='Sell Signal'), row=1, col=1)

    # === RSI + Rolling Mean + Divergence ===
    if 'rsi' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                 line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_rolling_mean'], name='RSI Mean',
                                 line=dict(color='lightgreen', dash='dot')), row=2, col=1)

    if 'rsi_div' in indicators:
        bull_rsi = df[df['BullDiv'] == True]
        bear_rsi = df[df['BearDiv'] == True]
        fig.add_trace(go.Scatter(x=bull_rsi.index, y=bull_rsi['Close'], mode='markers',
                                 marker=dict(symbol='star', color='green', size=8),
                                 name='Bullish RSI Div'), row=2, col=1)
        fig.add_trace(go.Scatter(x=bear_rsi.index, y=bear_rsi['Close'], mode='markers',
                                 marker=dict(symbol='star', color='red', size=8),
                                 name='Bearish RSI Div'), row=2, col=1)

    # === MACD ===
    if 'macd' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                 line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='MACD Signal',
                                 line=dict(color='red', dash='dot')), row=3, col=1)

    if 'macd_div' in indicators:
        bull_macd = df[df['MACD Divergence'] == 'Bullish']
        bear_macd = df[df['MACD Divergence'] == 'Bearish']
        fig.add_trace(go.Scatter(x=bull_macd.index, y=bull_macd['Close'], mode='markers',
                                 marker=dict(symbol='circle', color='green', size=6),
                                 name='Bullish MACD Div'), row=3, col=1)
        fig.add_trace(go.Scatter(x=bear_macd.index, y=bear_macd['Close'], mode='markers',
                                 marker=dict(symbol='circle', color='red', size=6),
                                 name='Bearish MACD Div'), row=3, col=1)

    # === OBV + Momentum + RSI Z ===
    if 'obv' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV',
                                 line=dict(color='purple')), row=4, col=1)

    if 'momentum' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['Momentum'], name='Momentum',
                                 line=dict(color='skyblue')), row=4, col=1)

    if 'rsi_z' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_Z'], name='RSI Z-Score',
                                 line=dict(color='gray', dash='dot')), row=4, col=1)

    # === Layout ===
    fig.update_layout(
        title=f"{symbol} - Custom Indicator Dashboard",
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=1200
    )
    fig.show()


def main():
    symbol = input("Enter stock symbol (e.g., AAPL, SBIN.NS): ").strip().upper()
    start = input("Enter start date (YYYY-MM-DD): ").strip()
    end = input("Enter end date (YYYY-MM-DD): ").strip()

    print("\nAvailable indicators: supertrend, bollinger, composite, rsi, rsi_div, macd, macd_div, obv, momentum, rsi_z")
    raw_input = input("Enter indicators (comma-separated): ").strip().lower()
    indicators = [i.strip() for i in raw_input.split(',') if i.strip()]

    print("\n📈 Running full strategy pipeline...")
    summary = process_stock(symbol, start, end)  # No need to modify
    print(summary)

    df = load_data(symbol)
    plot_dashboard(df, symbol, indicators)

if __name__ == "__main__":
    main()

from fastapi import FastAPI

from src.dtos.ISayHelloDto import ISayHelloDto

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World(...)"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/hello")
async def hello_message(dto: ISayHelloDto):
    return {"message": f"Hello {dto.message}"}


from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import json
import akshare as ak

# 参数配置
params = {
    'ma_periods': {'short': 5, 'medium': 20, 'long': 60},
    'rsi_period': 14,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'volume_ma_period': 20,
    'atr_period': 14
}

# 鉴权
def verify_auth_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization Scheme")
    valid_tokens = ["your-api-key"]  # 替换为实际API密钥
    if token not in valid_tokens:
        raise HTTPException(status_code=403, detail="Invalid or Expired Token")
    return token

class StockAnalysisRequest(BaseModel):
    stock_code: str
    market_type: str = 'A'
    start_date: str = None
    end_date: str = None

# 计算技术指标（EMA、RSI、MACD、布林带、ATR等）
def calculate_indicators(df):
    df['MA5'] = df['close'].ewm(span=params['ma_periods']['short'], adjust=False).mean()
    df['MA20'] = df['close'].ewm(span=params['ma_periods']['medium'], adjust=False).mean()
    df['MA60'] = df['close'].ewm(span=params['ma_periods']['long'], adjust=False).mean()
    df['RSI'] = calculate_rsi(df['close'], params['rsi_period'])
    df['MACD'], df['Signal'], df['MACD_hist'] = calculate_macd(df['close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(
        df['close'], params['bollinger_period'], params['bollinger_std']
    )
    df['Volume_MA'] = df['volume'].rolling(window=params['volume_ma_period']).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    df['ATR'] = calculate_atr(df, params['atr_period'])
    df['Volatility'] = df['ATR'] / df['close'] * 100
    df['ROC'] = df['close'].pct_change(periods=10) * 100
    return df

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(series, period, std_dev):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(df, period):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_score(df):
    score = 0
    latest = df.iloc[-1]
    if latest['MA5'] > latest['MA20']:
        score += 15
    if latest['MA20'] > latest['MA60']:
        score += 15
    if 30 <= latest['RSI'] <= 70:
        score += 20
    elif latest['RSI'] < 30:
        score += 15
    if latest['MACD'] > latest['Signal']:
        score += 20
    if latest['Volume_Ratio'] > 1.5:
        score += 30
    elif latest['Volume_Ratio'] > 1:
        score += 15
    return score

def get_recommendation(score):
    if score >= 80:
        return '强烈推荐买入'
    elif score >= 60:
        return '建议买入'
    elif score >= 40:
        return '观望'
    elif score >= 20:
        return '建议卖出'
    else:
        return '强烈建议卖出'

def get_stock_data(stock_code, market_type='A', start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if market_type == 'A':
        valid_prefixes = ['0', '3', '6', '688', '8']
        if not any(stock_code.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"无效的A股代码: {stock_code}")
        df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
    else:
        raise ValueError(f"不支持的市场类型: {market_type}")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
    df['date'] = pd.to_datetime(df['date'])
    df[['open', 'close', 'high', 'low', 'volume']] = df[['open', 'close', 'high', 'low', 'volume']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df.sort_values('date')

@app.post("/analyze-stock/")
async def analyze_stock(request: StockAnalysisRequest, auth_token: str = Depends(verify_auth_token)):
    stock_data = get_stock_data(request.stock_code, request.market_type, request.start_date, request.end_date)
    stock_data = calculate_indicators(stock_data)
    score = calculate_score(stock_data)
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
    technical_summary = {
        "trend": "upward" if latest['MA5'] > latest['MA20'] else "downward",
        "volatility": f"{latest['Volatility']:.2f}%",
        "volume_trend": "increasing" if latest['Volume_Ratio'] > 1 else "decreasing",
        "rsi_level": latest['RSI']
    }
    recent_data = stock_data.tail(14).to_dict('records')
    report = {
        "stock_code": request.stock_code,
        "market_type": request.market_type,
        "analysis_date": datetime.now().strftime('%Y-%m-%d'),
        "score": score,
        "price": latest['close'],
        "price_change": (latest['close'] - prev['close']) / prev['close'] * 100,
        "ma_trend": 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
        "rsi": latest['RSI'] if not pd.isna(latest['RSI']) else None,
        "macd_signal": 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
        "volume_status": 'HIGH' if latest['Volume_Ratio'] > 1.5 else 'NORMAL',
        "recommendation": get_recommendation(score)
    }
    return {"technical_summary": technical_summary, "recent_data": recent_data, "report": report}
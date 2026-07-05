"""
技术分析辅助模块
RSI / MACD / 均线交叉 → 动量得分
作为择时系统的 momentum 维度输入
"""
import pandas as pd
import numpy as np


class TechnicalIndicator:
    """
    技术指标信号生成器
    输出统一映射到 [-1, 1] 得分，正值=看多，负值=看空
    """

    # -----------------------------------------------------------
    # 1. RSI 信号
    # -----------------------------------------------------------
    @staticmethod
    def rsi_score(close: pd.Series, period: int = 14) -> float:
        """
        RSI 超买超卖信号
        RSI<30 超卖 → 看多(正分), RSI>70 超买 → 看空(负分)
        Returns: [-1, 1]
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        if pd.isna(current_rsi):
            return 0

        # 分段映射
        score = (35 - current_rsi) / 35
        return float(np.clip(score, -1, 1))

    # -----------------------------------------------------------
    # 2. MACD 信号
    # -----------------------------------------------------------
    @staticmethod
    def macd_score(close: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """
        MACD 动量信号
        快线>慢线 + 柱状线为正 → 看多
        Returns: [-1, 1]
        """
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        if histogram.empty or macd_line.empty:
            return 0

        # 标准化: histogram / 价格 作为相对强度
        rel_hist = histogram.iloc[-1] / close.iloc[-1]
        macd_hist_norm = np.clip(rel_hist * 100, -1, 1)

        # 方向确认: MACD线在信号线上方 → 加分
        direction = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
        strength = abs(macd_hist_norm) * direction

        return float(np.clip(strength, -1, 1))

    # -----------------------------------------------------------
    # 3. 均线交叉信号
    # -----------------------------------------------------------
    @staticmethod
    def ma_crossover_score(close: pd.Series,
                            short: int = 20, long: int = 60) -> float:
        """
        均线交叉趋势信号
        短>长=多头(正分), 短<长=空头(负分)
        相对距离越大信号越强
        Returns: [-1, 1]
        """
        if len(close) < long:
            return 0

        ma_short = close.rolling(short).mean()
        ma_long = close.rolling(long).mean()

        short_val = ma_short.iloc[-1]
        long_val = ma_long.iloc[-1]

        if pd.isna(short_val) or pd.isna(long_val) or long_val == 0:
            return 0

        # 相对偏离度
        distance = (short_val / long_val - 1)
        # 再检查近期趋势方向: long MA 斜率
        ma_long_slope = (ma_long.iloc[-1] / ma_long.iloc[-min(len(ma_long), 20)] - 1)

        # 组合: 交叉方向 + 趋势强度
        score = distance * 5 + ma_long_slope * 5
        return float(np.clip(score, -1, 1))

    # -----------------------------------------------------------
    # 4. 综合动量得分
    # -----------------------------------------------------------
    @staticmethod
    def get_composite_momentum(close: pd.Series) -> dict:
        """
        综合三个技术指标生成动量得分

        Returns
        -------
        dict: momentum_score, rsi, macd, ma_cross, 明细
        """
        rsi = TechnicalIndicator.rsi_score(close)
        macd = TechnicalIndicator.macd_score(close)
        ma_cross = TechnicalIndicator.ma_crossover_score(close)

        # 等权综合 (动量信号置信度一般不高，取均值)
        momentum = (rsi + macd + ma_cross) / 3
        momentum = float(np.clip(momentum, -1, 1))

        # 信号解释
        parts = []
        if rsi > 0.3:
            parts.append("RSI偏多")
        elif rsi < -0.3:
            parts.append("RSI偏空")
        if macd > 0.3:
            parts.append("MACD偏多")
        elif macd < -0.3:
            parts.append("MACD偏空")
        if ma_cross > 0.3:
            parts.append("均线多头")
        elif ma_cross < -0.3:
            parts.append("均线空头")

        return {
            "momentum_score": round(momentum, 3),
            "rsi": round(rsi, 3),
            "macd": round(macd, 3),
            "ma_cross": round(ma_cross, 3),
            "explanation": "; ".join(parts) if parts else "中性",
        }

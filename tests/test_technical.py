"""
技术指标模块单元测试
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.technical import TechnicalIndicator


class TestRSI:
    """RSI 信号测试"""

    def test_oversold_returns_positive(self):
        """连续下跌（超卖区）→ RSI 应返回正分（看多）"""
        close = pd.Series([100, 98, 96, 94, 92, 90, 88, 86, 84, 82,
                           80, 78, 76, 74, 72, 70, 68, 66, 64, 62])
        score = TechnicalIndicator.rsi_score(close, period=5)
        assert -1 <= score <= 1, "RSI得分应在[-1,1]范围"
        assert score > 0, f"超卖区应返回正分，实际={score}"

    def test_overbought_returns_negative(self):
        """连续上涨（超买区）→ RSI 应返回负分（看空）"""
        # 加少量噪声确保 RSI 计算不因除零报错
        np.random.seed(0)
        noise = np.random.randn(100) * 0.5
        trend = np.linspace(0, 40, 100)
        close = pd.Series(60 + trend + noise)
        score = TechnicalIndicator.rsi_score(close, period=14)
        assert -1 <= score <= 1
        assert score < 0, f"超买区应返回负分，实际={score}"

    def test_constant_price_neutral(self):
        """价格不变 → RSI 应接近 0（中性）"""
        close = pd.Series([100] * 30)
        score = TechnicalIndicator.rsi_score(close, period=5)
        assert -0.5 <= score <= 0.5

    def test_insufficient_data(self):
        """数据不足时返回 0 不崩溃"""
        close = pd.Series([100, 101])
        score = TechnicalIndicator.rsi_score(close, period=14)
        assert score == 0


class TestMACD:
    """MACD 信号测试"""

    def test_uptrend_positive(self):
        """上涨趋势 → MACD 正分"""
        close = pd.Series(np.linspace(50, 100, 100))
        score = TechnicalIndicator.macd_score(close)
        assert score > 0, f"上涨趋势MACD应正，实际={score}"

    def test_downtrend_negative(self):
        """下跌趋势 → MACD 负分"""
        close = pd.Series(np.linspace(100, 50, 100))
        score = TechnicalIndicator.macd_score(close)
        assert score < 0, f"下跌趋势MACD应负，实际={score}"

    def test_macd_range(self):
        """MACD 得分应在 [-1, 1] 范围内"""
        close = pd.Series(np.linspace(50, 100, 100))
        score = TechnicalIndicator.macd_score(close)
        assert -1 <= score <= 1

    def test_flat_market_neutral(self):
        """震荡市场 → MACD 接近 0"""
        np.random.seed(42)
        close = pd.Series(100 + np.random.randn(100) * 2)
        score = TechnicalIndicator.macd_score(close)
        assert -0.8 <= score <= 0.8


class TestMACrossover:
    """均线交叉信号测试"""

    def test_uptrend_positive(self):
        """上涨 → 短均线 > 长均线 → 正分"""
        close = pd.Series(np.linspace(50, 100, 100))
        score = TechnicalIndicator.ma_crossover_score(close, short=5, long=20)
        assert score > 0, f"上涨趋势均线应正，实际={score}"

    def test_downtrend_negative(self):
        """下跌 → 短均线 < 长均线 → 负分"""
        close = pd.Series(np.linspace(100, 50, 100))
        score = TechnicalIndicator.ma_crossover_score(close, short=5, long=20)
        assert score < 0, f"下跌趋势均线应负，实际={score}"

    def test_range(self):
        """得分应在 [-1, 1]"""
        close = pd.Series(np.linspace(50, 100, 100))
        score = TechnicalIndicator.ma_crossover_score(close)
        assert -1 <= score <= 1

    def test_insufficient_data_returns_zero(self):
        """数据不足返回 0 不崩溃"""
        close = pd.Series([100, 101, 102])
        score = TechnicalIndicator.ma_crossover_score(close, short=5, long=20)
        assert score == 0


class TestCompositeMomentum:
    """综合动量信号测试"""

    def test_returns_all_fields(self):
        """返回所有必需的字段"""
        close = pd.Series(np.linspace(50, 100, 100))
        result = TechnicalIndicator.get_composite_momentum(close)
        assert "momentum_score" in result
        assert "rsi" in result
        assert "macd" in result
        assert "ma_cross" in result
        assert "explanation" in result

    def test_momentum_score_range(self):
        """综合得分在 [-1, 1]"""
        close = pd.Series(np.linspace(50, 100, 100))
        result = TechnicalIndicator.get_composite_momentum(close)
        assert -1 <= result["momentum_score"] <= 1

    def test_uptrend_signals(self):
        """上涨趋势中均线和综评为正"""
        close = pd.Series(np.linspace(50, 100, 200))
        result = TechnicalIndicator.get_composite_momentum(close)
        assert result["ma_cross"] > 0, f"ma_cross={result['ma_cross']}"
        assert result["momentum_score"] > 0, f"momentum={result['momentum_score']}"

    def test_short_data_no_error(self):
        """短序列不崩溃"""
        close = pd.Series([100, 101])
        result = TechnicalIndicator.get_composite_momentum(close)
        assert abs(result["momentum_score"]) < 0.01, f"短序列得分应接近0，实际={result['momentum_score']}"

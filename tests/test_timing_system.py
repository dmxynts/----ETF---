"""
综合择时系统单元测试
"""
import pytest
import pandas as pd
import numpy as np
from src.strategies.timing_system import (
    TimingSystem, SignalWeights, VolSubWeights, UnifiedSignal
)


# ---- 模块级别 fixture（所有类都可访问） ----

@pytest.fixture
def neutral_signals():
    return {
        "spread_signal": {"percentile": 0.5},
        "macro_signal": {"suggested_position": 0.5},
        "vol_signal": {"level_score": 0, "trend_score": 0,
                       "forecast_score": 0, "event_score": 0, "vol_zscore": 0},
        "momentum_score": 0.0,
    }


@pytest.fixture
def bullish_signals():
    return {
        "spread_signal": {"percentile": 0.95},
        "macro_signal": {"suggested_position": 0.8},
        "vol_signal": {"level_score": 1.0, "trend_score": 0.5,
                       "forecast_score": 0.5, "event_score": 1.0, "vol_zscore": 2.5},
        "momentum_score": 0.8,
    }


@pytest.fixture
def bearish_signals():
    return {
        "spread_signal": {"percentile": 0.05},
        "macro_signal": {"suggested_position": 0.2},
        "vol_signal": {"level_score": -0.8, "trend_score": -0.5,
                       "forecast_score": -0.5, "event_score": -0.8, "vol_zscore": -2.0},
        "momentum_score": -0.8,
    }


class TestWeights:
    """权重配置测试"""

    def test_default_weights_sum_to_one(self):
        w = SignalWeights()
        total = w.spread + w.macro + w.volatility + w.momentum
        assert abs(total - 1.0) < 0.01

    def test_vol_sub_weights_sum_to_one(self):
        w = VolSubWeights()
        total = w.level + w.trend + w.forecast + w.event
        assert abs(total - 1.0) < 0.01

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="权重之和"):
            TimingSystem(weights=SignalWeights(0.5, 0.5, 0.5, 0.5))

    def test_custom_weights_accepted(self):
        w = SignalWeights(0.5, 0.3, 0.1, 0.1)
        ts = TimingSystem(weights=w)
        assert ts.weights.spread == 0.5


class TestSignalEvaluation:
    """信号评估测试"""

    def test_neutral_returns_hold(self, neutral_signals):
        ts = TimingSystem()
        result = ts.evaluate(**neutral_signals)
        assert 0.3 <= result.position <= 0.7
        assert isinstance(result, UnifiedSignal)

    def test_bullish_high_position(self, bullish_signals):
        ts = TimingSystem()
        result = ts.evaluate(**bullish_signals)
        assert result.position > 0.5
        assert result.action in ("满仓/加仓", "增持")

    def test_bearish_low_position(self, bearish_signals):
        ts = TimingSystem()
        result = ts.evaluate(**bearish_signals)
        assert result.position < 0.5
        assert result.action in ("清仓/轻仓", "减仓")

    def test_score_symmetric(self):
        ts = TimingSystem()
        bullish = ts.evaluate(
            spread_signal={"percentile": 1.0},
            macro_signal={"suggested_position": 1.0},
            vol_signal={"level_score": 0, "trend_score": 0,
                        "forecast_score": 0, "event_score": 0, "vol_zscore": 0},
            momentum_score=1.0,
        )
        bearish = ts.evaluate(
            spread_signal={"percentile": 0.0},
            macro_signal={"suggested_position": 0.0},
            vol_signal={"level_score": 0, "trend_score": 0,
                        "forecast_score": 0, "event_score": 0, "vol_zscore": 0},
            momentum_score=-1.0,
        )
        assert abs(bullish.composite_score + bearish.composite_score) < 0.01

    def test_position_maps_correctly(self):
        ts = TimingSystem(weights=SignalWeights(1, 0, 0, 0))
        neg = ts.evaluate(
            spread_signal={"percentile": 0.0},
            macro_signal={"suggested_position": 0.5},
            vol_signal={"level_score": 0, "trend_score": 0,
                        "forecast_score": 0, "event_score": 0, "vol_zscore": 0},
        )
        assert neg.position == 0.0

        pos = ts.evaluate(
            spread_signal={"percentile": 1.0},
            macro_signal={"suggested_position": 0.5},
            vol_signal={"level_score": 0, "trend_score": 0,
                        "forecast_score": 0, "event_score": 0, "vol_zscore": 0},
        )
        assert pos.position == 1.0

    def test_details_contains_all_dims(self, neutral_signals):
        ts = TimingSystem()
        result = ts.evaluate(**neutral_signals)
        for key in ["spread_score", "macro_score", "vol_score",
                     "momentum_score", "vol_level", "vol_trend",
                     "vol_forecast", "vol_event"]:
            assert key in result.details


class TestETFAllocation:
    """ETF配置建议测试"""

    def test_target_etf_returns_single(self):
        result = TimingSystem.suggest_etf_allocation(0.8, target_etf="510880")
        assert len(result) == 1
        assert result[0]["代码"] == "510880"
        assert result[0]["权重"] == 0.8

    def test_empty_etf_list_returns_empty(self):
        result = TimingSystem.suggest_etf_allocation(0.8, etf_list=[])
        assert result == []

    def test_no_target_uses_etf_list(self):
        from types import SimpleNamespace
        etfs = [
            SimpleNamespace(code="510880", name="AA红利ETF", index_code="000922", index_name="中证红利指数"),
            SimpleNamespace(code="515180", name="BB中证红利ETF", index_code="000922", index_name="中证红利指数"),
        ]
        result = TimingSystem.suggest_etf_allocation(0.8, etf_list=etfs)
        assert len(result) == 2
        # BB中证红利 → 0.5 * 0.8 = 0.4, AA红利 → 0.2 * 0.8 = 0.16
        # 先遍历到第一个匹配"中证红利"的
        weights = [r["权重"] for r in result]
        assert 0.4 in weights
        assert 0.16 in weights

    def test_zero_position_all_zero_weights(self):
        from types import SimpleNamespace
        etfs = [SimpleNamespace(code="510880", name="红利ETF", index_code="000922", index_name="中证红利指数")]
        result = TimingSystem.suggest_etf_allocation(0.0, etf_list=etfs)
        assert all(r["权重"] == 0 for r in result)


class TestHistory:
    """信号历史记录测试"""

    def test_history_appended(self, neutral_signals):
        ts = TimingSystem()
        ts.evaluate(**neutral_signals)
        assert len(ts.history) == 1
        ts.evaluate(**neutral_signals)
        assert len(ts.history) == 2

    def test_get_summary_returns_dataframe(self, neutral_signals):
        ts = TimingSystem()
        for _ in range(5):
            ts.evaluate(**neutral_signals)
        summary = ts.get_summary()
        assert len(summary) == 5
        for col in ["综评", "仓位", "动作", "解释"]:
            assert col in summary.columns


class TestWeeklyChecklist:
    """周度检查清单测试"""

    def test_checklist_contains_sections(self, neutral_signals):
        ts = TimingSystem()
        checklist = ts.run_weekly_checklist(
            neutral_signals["spread_signal"],
            neutral_signals["macro_signal"],
            neutral_signals["vol_signal"],
        )
        assert "综合信号" in checklist
        assert "操作建议" in checklist


class TestCalibrate:
    """权重校准测试"""

    @pytest.fixture
    def calibration_data(self):
        """生成模拟的校准数据: 包含已知信号模式和价格"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")

        # 价格: 震荡上行
        trend = np.linspace(0, 0.2, 500)
        noise = np.random.randn(500) * 0.01
        close = 100 * (1 + trend + noise).cumprod() / (1 + trend + noise).cumprod()[0] * 100

        price = pd.DataFrame({"date": dates, "close": close})

        # spread: 0.2~0.8 之间摆动
        spread_df = pd.DataFrame({
            "percentile": 0.4 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, 500)),
        }, index=dates)

        # macro: 0.3~0.7
        macro_df = pd.DataFrame({
            "suggested_position": 0.5 + 0.2 * np.sin(np.linspace(0, 3 * np.pi, 500) + 1),
        }, index=dates)

        # vol: 四个子信号围绕 0 波动
        vol_df = pd.DataFrame({
            "level_score": 0.2 * np.sin(np.linspace(0, 5 * np.pi, 500)),
            "trend_score": 0.1 * np.cos(np.linspace(0, 4 * np.pi, 500)),
            "forecast_score": 0.15 * np.sin(np.linspace(0, 3 * np.pi, 500) + 0.5),
            "event_score": 0.05 * np.random.randn(500),
        }, index=dates)

        # momentum: [-0.5, 0.5]
        momentum_df = pd.DataFrame({
            "score": 0.3 * np.sin(np.linspace(0, 6 * np.pi, 500)),
        }, index=dates)

        return price, spread_df, macro_df, vol_df, momentum_df

    def test_calibrate_returns_expected_structure(self, calibration_data):
        """calibrate 返回预期结构"""
        price, spread_df, macro_df, vol_df, momentum_df = calibration_data
        ts = TimingSystem()
        result = ts.calibrate(price, spread_df, macro_df, vol_df, momentum_df, verbose=False)

        assert "best_weights" in result
        assert "best_metric" in result
        assert "previous_weights" in result
        assert "results" in result
        assert "n_trials" in result
        assert result["n_trials"] > 0

        bw = result["best_weights"]
        assert 0 < bw.spread < 1
        assert 0 < bw.macro < 1
        assert 0 < bw.volatility < 1
        assert 0 < bw.momentum < 1
        assert abs(bw.spread + bw.macro + bw.volatility + bw.momentum - 1.0) < 0.01

    def test_calibrate_results_sorted(self, calibration_data):
        """结果按 metric 降序排列"""
        price, spread_df, macro_df, vol_df, momentum_df = calibration_data
        ts = TimingSystem()
        result = ts.calibrate(price, spread_df, macro_df, vol_df, momentum_df, verbose=False)

        results = result["results"]
        assert results[result["metric"]].is_monotonic_decreasing

    def test_calibrate_updates_weights(self, calibration_data):
        """校准后 self.weights 被更新"""
        price, spread_df, macro_df, vol_df, momentum_df = calibration_data
        ts = TimingSystem()
        original_spread = ts.weights.spread
        result = ts.calibrate(price, spread_df, macro_df, vol_df, momentum_df, verbose=False)

        # 权重应已被更新
        assert ts.weights.spread == result["best_weights"].spread

    def test_calibrate_previous_metric_not_none(self, calibration_data):
        """校准返回中包含校准前指标"""
        price, spread_df, macro_df, vol_df, momentum_df = calibration_data
        ts = TimingSystem()
        result = ts.calibrate(price, spread_df, macro_df, vol_df, momentum_df, verbose=False)
        assert result["previous_metric"] is not None

    def test_calibrate_insufficient_data_raises(self):
        """不足20行数据应报错"""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        price = pd.DataFrame({"date": dates, "close": np.linspace(100, 110, 10)})
        empty = pd.DataFrame({"x": np.zeros(10)}, index=dates)

        ts = TimingSystem()
        with pytest.raises(ValueError, match="样本不足"):
            ts.calibrate(price, empty.rename(columns={"x": "percentile"}),
                         empty.rename(columns={"x": "suggested_position"}),
                         pd.DataFrame({"level_score": [0]*10, "trend_score": [0]*10,
                                       "forecast_score": [0]*10, "event_score": [0]*10}, index=dates),
                         empty.rename(columns={"x": "score"}),
                         verbose=False)

    def test_calibrate_best_not_worse_than_default(self, calibration_data):
        """最优 metric 应不低于默认权重"""
        price, spread_df, macro_df, vol_df, momentum_df = calibration_data
        ts = TimingSystem()
        result = ts.calibrate(price, spread_df, macro_df, vol_df, momentum_df,
                              metric="sharpe", verbose=False)
        assert result["best_metric"] >= result["previous_metric"] - 0.001  # 允许浮点误差

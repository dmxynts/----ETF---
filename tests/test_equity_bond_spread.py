"""
股债利差择时模块单元测试
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.equity_bond_spread import EquityBondSpread


@pytest.fixture
def sample_spread_data():
    """生成测试用的股债利差数据"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    div_yield = pd.DataFrame({"date": dates, "dividend_yield": 0.05 + np.random.randn(500) * 0.005})
    bond_yield = pd.DataFrame({"date": dates, "yield": 0.025 + np.random.randn(500) * 0.002})
    div_yield["dividend_yield"] = div_yield["dividend_yield"].clip(0.03, 0.08)
    bond_yield["yield"] = bond_yield["yield"].clip(0.01, 0.04)
    return div_yield, bond_yield


class TestSpreadComputation:
    """利差计算测试"""

    def test_compute_spread_basic(self, sample_spread_data):
        """计算出的利差 = 股息率 - 国债收益率"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        result = s.compute_spread(div_df, bond_df)
        assert "spread" in result.columns
        # 验证第一行的利差
        row = result.iloc[0]
        expected = row["dividend_yield"] - row["yield"]
        assert abs(row["spread"] - expected) < 1e-6

    def test_compute_spread_stores_data(self, sample_spread_data):
        """compute_spread 后 spread_data 应为非空"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        assert s.spread_data is not None
        assert not s.spread_data.empty

    def test_spread_positive_when_div_high(self):
        """股息率高时利差应为正"""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        div_df = pd.DataFrame({"date": dates, "dividend_yield": 0.08})
        bond_df = pd.DataFrame({"date": dates, "yield": 0.02})
        s = EquityBondSpread()
        result = s.compute_spread(div_df, bond_df)
        assert (result["spread"] > 0).all()

    def test_spread_negative_when_bond_high(self):
        """国债收益率高时利差应为负"""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        div_df = pd.DataFrame({"date": dates, "dividend_yield": 0.02})
        bond_df = pd.DataFrame({"date": dates, "yield": 0.06})
        s = EquityBondSpread()
        result = s.compute_spread(div_df, bond_df)
        assert (result["spread"] < 0).all()

    def test_auto_convert_percentage_to_decimal(self):
        """输入为百分比(>1)时自动转为小数"""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        div_df = pd.DataFrame({"date": dates, "dividend_yield": 5.0})  # 5%
        bond_df = pd.DataFrame({"date": dates, "yield": 2.5})  # 2.5%
        s = EquityBondSpread()
        result = s.compute_spread(div_df, bond_df)
        assert abs(result["spread"].iloc[0] - 0.025) < 1e-4


class TestPercentile:
    """利差分位计算测试"""

    def test_percentile_after_compute(self, sample_spread_data):
        """compute_percentile 返回需要的列"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        result = s.compute_percentile()
        for col in ["date", "spread", "percentile", "z_score"]:
            assert col in result.columns, f"缺少列: {col}"

    def test_percentile_between_0_and_1(self, sample_spread_data):
        """分位应在 [0, 1] 之间"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        result = s.compute_percentile()
        p = result["percentile"].dropna()
        assert (p >= 0).all(), "有分位 < 0"
        assert (p <= 1).all(), "有分位 > 1"


class TestSignal:
    """信号生成测试"""

    def test_signal_after_percentile(self, sample_spread_data):
        """generate_signal 返回需要的字段"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        result = s.generate_signal()
        for col in ["date", "spread", "percentile", "position", "signal"]:
            assert col in result.columns, f"缺少列: {col}"

    def test_signal_position_range(self, sample_spread_data):
        """仓位应在 [0, 1] 之间"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        result = s.generate_signal()
        assert (result["position"] >= 0).all()
        assert (result["position"] <= 1).all()

    def test_high_percentile_full_position(self):
        """利差分位高于阈值 → 满仓"""
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        # 构造利差持续走高的数据
        spreads = np.linspace(0.01, 0.05, 500)
        div_df = pd.DataFrame({"date": dates, "dividend_yield": 0.04 + spreads})
        bond_df = pd.DataFrame({"date": dates, "yield": 0.02})
        s = EquityBondSpread(high_percentile=0.70, low_percentile=0.30)
        s.compute_spread(div_df, bond_df)
        s.compute_percentile()
        result = s.generate_signal()
        # 最后阶段分位应很高 → 满仓
        assert result["position"].iloc[-1] >= 0.9

    def test_low_percentile_light_position(self):
        """利差分位低于阈值 → 轻仓"""
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        # 构造利差持续走低的数据
        spreads = np.linspace(0.05, 0.01, 500)
        div_df = pd.DataFrame({"date": dates, "dividend_yield": 0.04 + spreads})
        bond_df = pd.DataFrame({"date": dates, "yield": 0.02})
        s = EquityBondSpread(high_percentile=0.70, low_percentile=0.30,
                              light_position=0.2)
        s.compute_spread(div_df, bond_df)
        s.compute_percentile(window_years=1)
        result = s.generate_signal()
        assert result["position"].iloc[-1] <= 0.4

    def test_custom_thresholds(self, sample_spread_data):
        """自定义阈值应生效"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread(high_percentile=0.90, low_percentile=0.10,
                              full_position=0.8, light_position=0.2)
        s.compute_spread(div_df, bond_df)
        result = s.generate_signal()
        assert result["position"].max() <= 0.8
        assert result["position"].min() >= 0.2

    def test_signal_auto_calls_percentile(self, sample_spread_data):
        """未调用 compute_percentile 时 generate_signal 自动调用"""
        div_df, bond_df = sample_spread_data
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        # 直接调用 generate_signal，不先调 compute_percentile
        result = s.generate_signal()
        assert "position" in result.columns


class TestBacktest:
    """回测功能测试"""

    def test_backtest_returns_columns(self, sample_spread_data):
        """backtest 返回预期列"""
        div_df, bond_df = sample_spread_data
        index_df = pd.DataFrame({
            "date": div_df["date"],
            "close": np.linspace(100, 120, len(div_df))
        })
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        s.generate_signal()
        result = s.backtest(index_df)
        for col in ["date", "strategy_nav", "index_nav", "position"]:
            assert col in result.columns, f"回测结果缺少列: {col}"

    def test_backtest_metrics(self, sample_spread_data):
        """get_metrics 返回预期指标"""
        div_df, bond_df = sample_spread_data
        index_df = pd.DataFrame({
            "date": div_df["date"],
            "close": np.linspace(100, 120, len(div_df))
        })
        s = EquityBondSpread()
        s.compute_spread(div_df, bond_df)
        s.generate_signal()
        bt = s.backtest(index_df)
        metrics = s.get_metrics(bt)
        for key in ["策略总收益率", "年化收益率", "夏普比率", "最大回撤"]:
            assert key in metrics, f"指标缺少: {key}"


class TestCurrentJudgment:
    """当前判断测试"""

    def test_high_percentile_signal(self):
        """历史高分位 → 加仓建议"""
        spreads = pd.Series(np.random.randn(500) * 0.005 + 0.02)
        result = EquityBondSpread.current_judgment(0.06, 0.02, spreads)
        assert "加仓" in result["操作建议"] or "红利" in result["操作建议"]

    def test_current_judgment_keys(self):
        """判断结果包含必要字段"""
        spreads = pd.Series(np.random.randn(100) * 0.01 + 0.02)
        result = EquityBondSpread.current_judgment(0.05, 0.025, spreads)
        for key in ["当前利差", "历史分位", "Z-Score", "操作建议"]:
            assert key in result, f"缺少: {key}"

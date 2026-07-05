"""
回测引擎单元测试
"""
import pytest
import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine


def _make_price_signal(n_days=252, start_price=100, end_price=120, position=1.0):
    """生成测试用的价格和信号数据"""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    price = pd.DataFrame({
        "date": dates,
        "close": np.linspace(start_price, end_price, n_days)
    })
    signal = pd.DataFrame({
        "date": dates,
        "position": position
    })
    return price, signal


class TestBacktestEngine:
    """回测引擎核心功能测试"""

    def setup_method(self):
        self.engine = BacktestEngine(initial_capital=100000)

    def test_full_position_tracks_benchmark(self):
        """满仓策略收益应与基准接近"""
        price, signal = _make_price_signal(position=1.0)
        result = self.engine.run(price, signal)

        nav = result.nav
        total_ret = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
        bench_ret = price["close"].iloc[-1] / price["close"].iloc[0] - 1
        # 满仓扣除手续费后收益略低于基准，但差距应小于1%
        assert total_ret > bench_ret * 0.97, f"总收益={total_ret:.4%}, 基准={bench_ret:.4%}"

    def test_zero_position_no_return(self):
        """空仓策略总收益应接近 0"""
        price, signal = _make_price_signal(position=0.0)
        result = self.engine.run(price, signal)
        total_ret_str = result.metrics["总收益率"]
        total_ret = float(total_ret_str.strip("%")) / 100
        # 空仓可能有微量手续费影响，但不应有大额盈亏
        assert -0.02 < total_ret < 0.02, f"空仓总收益={total_ret:.4%}"

    def test_half_position_half_volatility(self):
        """半仓策略的波动率应低于满仓"""
        price, signal_full = _make_price_signal(
            n_days=1000, end_price=150, position=1.0
        )
        _, signal_half = _make_price_signal(
            n_days=1000, end_price=150, position=0.5
        )

        result_full = self.engine.run(price, signal_full)
        result_half = self.engine.run(price, signal_half)

        vol_full = float(result_full.metrics["年化波动率"].strip("%"))
        vol_half = float(result_half.metrics["年化波动率"].strip("%"))
        assert vol_half < vol_full, "半仓波动率应低于满仓"

    def test_metrics_contain_all_keys(self):
        """回测指标包含所有必需字段"""
        price, signal = _make_price_signal()
        result = self.engine.run(price, signal)
        expected_keys = [
            "总收益率", "年化收益率", "年化波动率",
            "夏普比率", "最大回撤", "卡尔玛比率", "日胜率", "盈亏比"
        ]
        for key in expected_keys:
            assert key in result.metrics, f"缺少指标: {key}"

    def test_nav_columns(self):
        """净值数据包含必要列"""
        price, signal = _make_price_signal()
        result = self.engine.run(price, signal)
        for col in ["date", "nav", "index_nav", "drawdown", "position"]:
            assert col in result.nav.columns, f"净值缺少列: {col}"

    def test_drawdown_never_positive(self):
        """回撤值应始终 ≤ 0（容忍浮点误差）"""
        price, signal = _make_price_signal(n_days=500, end_price=150)
        result = self.engine.run(price, signal)
        max_dd = result.nav["drawdown"].max()
        assert max_dd < 1e-10, f"回撤不应为正，最大={max_dd}"

    def test_max_drawdown_reasonable(self):
        """最大回撤应在合理范围"""
        price, signal = _make_price_signal(n_days=500, end_price=150)
        result = self.engine.run(price, signal)
        max_dd = float(result.metrics["最大回撤"].strip("%")) / 100
        assert -0.5 <= max_dd <= 0, f"最大回撤不合理: {max_dd:.2%}"

    def test_sharpe_positive_in_uptrend(self):
        """上涨趋势中夏普比率应为正"""
        price, signal = _make_price_signal(n_days=500, end_price=150)
        result = self.engine.run(price, signal)
        sharpe = float(result.metrics["夏普比率"])
        assert sharpe > 0, f"上涨趋势夏普应正，实际={sharpe}"

    def test_fee_reduces_return(self):
        """更高手续费导致更低收益"""
        # 用更长的序列和频繁变动的仓位放大手续费差异
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        price = pd.DataFrame({"date": dates, "close": np.linspace(100, 150, 500)})
        # 仓位频繁变动产生更多手续费
        signal = pd.DataFrame({"date": dates, "position": [0.5, 1.0, 0.3] * 166 + [0.5, 0.0]})

        result_low = BacktestEngine(100000).run(price, signal, fee_rate=0.0001)
        result_high = BacktestEngine(100000).run(price, signal, fee_rate=0.01)

        ret_low = float(result_low.metrics["总收益率"].strip("%"))
        ret_high = float(result_high.metrics["总收益率"].strip("%"))
        assert ret_low > ret_high, f"低手续费收益({ret_low:.2%})应高于高手续费({ret_high:.2%})"

    def test_cost_aligned_with_first_trade(self):
        """首次调仓成本基于 |position[0] - 0|（初始空仓→首次信号）"""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        price = pd.DataFrame({"date": dates, "close": np.linspace(100, 110, 50)})
        # 空仓10天，然后满仓
        signal = pd.DataFrame({
            "date": dates,
            "position": [0.0] * 10 + [1.0] * 40,
        })
        result = BacktestEngine(initial_capital=1.0).run(price, signal, fee_rate=0.01)
        nav = result.nav
        # 前10天空仓 → nav 应保持 1.0
        assert abs(nav["nav"].iloc[9] - 1.0) < 1e-6, f"空仓期净值应保持1.0，实际={nav['nav'].iloc[9]}"
        # 第11日起使用 position[9]=0.0 交易 → 仓位不变，仍为0，无成本
        # 第12日起使用 position[10]=1.0 交易 → 首笔调仓: |1.0-0|*0.01 = 0.01
        # strategy_return[11] = return[11] * 1.0 - 0.01
        # nav[11] = nav[10] * (1 + strategy_return[11])
        row11 = nav.iloc[11]
        expected_cost = 0.01  # |1.0-0| * 0.01
        # 因当天也有市场收益，净值的累计效应 ≈ -1%
        nav_loss = row11["nav"] / nav["nav"].iloc[10] - 1
        # 净值变化应接近 市场收益-成本
        price_ret = price["close"].iloc[11] / price["close"].iloc[10] - 1
        expected_nav_ret = price_ret * 1.0 - expected_cost
        assert abs(nav_loss - expected_nav_ret) < 0.001, (
            f"第12日净值变化={nav_loss:.4%}, 预期={expected_nav_ret:.4%}"
        )


class TestBenchmarkComparison:
    """基准比较测试"""

    def test_benchmark_comparison_columns(self):
        """基准比较返回必要字段"""
        price, signal = _make_price_signal()
        result = BacktestEngine(100000).run(price, signal)
        comp = BacktestEngine.benchmark_comparison(result.nav)
        expected = ["策略收益率", "基准收益率", "超额收益", "跟踪误差", "信息比率"]
        for key in expected:
            assert key in comp, f"缺少: {key}"

    def test_excess_return_near_zero(self):
        """满仓策略超额收益应接近0"""
        price, signal = _make_price_signal(position=1.0)
        result = BacktestEngine(100000).run(price, signal)
        comp = BacktestEngine.benchmark_comparison(result.nav)
        excess_str = comp["超额收益"]
        excess = float(excess_str.strip("%")) / 100
        # 满仓跟随基准，超额应接近0（手续费影响很小）
        assert -0.10 < excess < 0.10, f"超额收益={excess:.4%}"


class TestMultiAsset:
    """多资产组合测试"""

    def test_multi_asset_single_valid(self):
        """单资产组合回测应正常运行"""
        price, signal = _make_price_signal()
        prices = {"asset1": price}
        signals = {"asset1": signal}
        result = BacktestEngine().run_multi_asset(prices, signals, {"asset1": 1.0})
        assert result.metrics["总收益率"] is not None

    def test_multi_asset_empty_raises(self):
        """无有效资产应抛出异常"""
        with pytest.raises(ValueError, match="无有效资产"):
            BacktestEngine().run_multi_asset({}, {}, {})

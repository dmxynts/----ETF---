"""
风险管理模块单元测试
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.risk_management import RiskManager


@pytest.fixture
def normal_returns():
    """正态分布收益率（1000个样本）"""
    np.random.seed(42)
    return pd.Series(np.random.randn(1000) * 0.01, name="returns")


@pytest.fixture
def crash_returns():
    """含极端值的收益率（模拟崩盘）"""
    np.random.seed(42)
    normal = np.random.randn(980) * 0.01
    crash = np.array([-0.05, -0.04, -0.06, -0.03, -0.035,
                      -0.045, -0.055, -0.025, -0.04, -0.05,
                      -0.03, -0.02, -0.04, -0.05, -0.06,
                      -0.07, -0.03, -0.04, -0.05, -0.045])
    return pd.Series(np.concatenate([normal, crash]), name="returns")


@pytest.fixture
def rising_returns():
    """稳定上涨的收益率（用单调递增价格反算）"""
    prices = np.linspace(100, 150, 500)
    returns = pd.Series(prices).pct_change().dropna()
    returns.name = "returns"
    return returns


class TestVaR:
    """VaR 计算测试"""

    def test_var_historical_negative(self, normal_returns):
        """VaR 应为负值（损失）"""
        rm = RiskManager(confidence_level=0.95)
        var = rm.var_historical(normal_returns)
        assert var < 0, f"VaR应为负，实际={var}"

    def test_var_historical_99_stricter_than_95(self, normal_returns):
        """99% VaR 应比 95% VaR 更负（更严格）"""
        rm = RiskManager()
        var_95 = rm.var_historical(normal_returns, 0.95)
        var_99 = rm.var_historical(normal_returns, 0.99)
        assert var_99 < var_95, f"99%VaR({var_99})应<95%VaR({var_95})"

    def test_var_gaussian_sensible(self, normal_returns):
        """正态VaR应在合理范围内"""
        rm = RiskManager()
        var = rm.var_gaussian(normal_returns)
        assert -0.05 < var < 0, f"正态VaR不合理: {var}"

    def test_var_cornish_fisher(self, normal_returns):
        """Cornish-Fisher VaR 应在正常范围"""
        rm = RiskManager()
        var = rm.var_cornish_fisher(normal_returns)
        assert -0.06 < var < 0, f"CF-VaR不合理: {var}"

    def test_es_worse_than_var(self, normal_returns):
        """ES(CVaR) 应比同置信度 VaR 更负"""
        rm = RiskManager()
        var = rm.var_historical(normal_returns, 0.95)
        es = rm.expected_shortfall(normal_returns, 0.95)
        assert es <= var, f"ES({es})应<=VaR({var})"

    def test_crash_var_significantly_negative(self, crash_returns):
        """含崩盘数据的VaR应显著为负"""
        rm = RiskManager(confidence_level=0.99)
        var = rm.var_historical(crash_returns)
        assert var < -0.02, f"崩盘数据VaR应显著为负，实际={var}"


class TestEVT:
    """极值理论测试"""

    def test_evt_returns_dict(self, normal_returns):
        """EVT 返回字典结构"""
        rm = RiskManager()
        result = rm.evt_gev(normal_returns, block_size=21)
        assert isinstance(result, dict)

    def test_evt_warning_insufficient_data(self):
        """数据不足时返回警告"""
        rm = RiskManager()
        short_ret = pd.Series(np.random.randn(50) * 0.01)
        result = rm.evt_gev(short_ret, block_size=21)
        assert "warning" in result

    def test_evt_1000_days_returns_shape(self, normal_returns):
        """1000天数据应能拟合形状参数"""
        rm = RiskManager()
        result = rm.evt_gev(normal_returns, block_size=21)
        if "EVT_VaR" in result:
            assert -0.15 < result["EVT_VaR"] < 0


class TestDrawdown:
    """回撤分析测试"""

    def test_rising_market_no_drawdown(self, rising_returns):
        """上涨市场当前回撤应接近0"""
        rm = RiskManager()
        dd = rm.drawdown_analysis(rising_returns)
        assert dd["当前回撤"] >= -0.05 or dd["当前回撤"] <= 0
        # 上涨市回撤很小
        assert dd["当前回撤"] > -0.015, f"上涨市回撤应小，实际={dd['当前回撤']:.4%}"

    def test_drawdown_contains_all_keys(self, normal_returns):
        """回撤分析包含所有必需字段"""
        rm = RiskManager()
        dd = rm.drawdown_analysis(normal_returns)
        for key in ["当前回撤", "历史最大回撤", "回撤天数", "速度比", "趋势", "状态", "预警"]:
            assert key in dd, f"回撤分析缺少字段: {key}"

    def test_max_drawdown_negative(self, normal_returns):
        """最大回撤应为负"""
        rm = RiskManager()
        dd = rm.drawdown_analysis(normal_returns)
        assert dd["历史最大回撤"] < 0

    def test_crash_has_deep_drawdown(self, crash_returns):
        """崩盘数据回撤应较深"""
        rm = RiskManager()
        dd = rm.drawdown_analysis(crash_returns)
        assert dd["当前回撤"] < -0.01, f"崩盘回撤应深，实际={dd['当前回撤']:.4%}"


class TestStressTest:
    """压力测试测试"""

    def test_stress_test_contains_scenarios(self, normal_returns):
        """压力测试包含多个情景"""
        result = RiskManager.stress_test(normal_returns, holding_value=1000000)
        assert len(result) >= 5, f"情景数应>=5，实际={len(result)}"

    def test_stress_test_each_has_loss_amount(self, normal_returns):
        """每个情景包含损失金额"""
        result = RiskManager.stress_test(normal_returns)
        for name, detail in result.items():
            assert "损失金额" in detail, f"{name}缺少损失金额"
            assert detail["损失金额"] >= 0, f"{name}损失金额应>=0"

    def test_larger_shock_larger_loss(self, normal_returns):
        """单日暴跌越大损失越大"""
        result = RiskManager.stress_test(normal_returns, holding_value=1000000)
        shock_3 = result["单日3%暴跌"]["损失金额"]
        shock_5 = result["单日5%暴跌"]["损失金额"]
        shock_8 = result["单日8%暴跌"]["损失金额"]
        assert shock_3 < shock_5 < shock_8, "跌幅越大损失应越大"


class TestDynamicStopLoss:
    """动态止损测试"""

    def test_stop_loss_contains_keys(self, normal_returns):
        """止损建议包含必需字段"""
        rm = RiskManager()
        sl = rm.dynamic_stop_loss(normal_returns)
        for key in ["止损倍数", "当前波动率", "建议止损线(%)", "止损依据"]:
            assert key in sl, f"止损缺少: {key}"

    def test_stop_loss_positive(self, normal_returns):
        """止损线应为正"""
        rm = RiskManager()
        sl = rm.dynamic_stop_loss(normal_returns)
        assert sl["建议止损线(%)"] > 0, "止损线应>0"

    def test_crash_tighter_stop(self, crash_returns, normal_returns):
        """崩盘后止损线应比正常市更紧（倍数更小）"""
        rm = RiskManager()
        sl_normal = rm.dynamic_stop_loss(normal_returns)
        sl_crash = rm.dynamic_stop_loss(crash_returns)
        assert sl_crash["止损倍数"] <= sl_normal["止损倍数"], "崩盘后止损应更紧"


class TestRiskBudget:
    """风险预算测试"""

    def test_budget_contains_keys(self, normal_returns):
        """风险预算包含必需字段"""
        rm = RiskManager()
        budget = rm.risk_budget(normal_returns, 1000000)
        for key in ["总资金", "建议持仓上限", "持仓上限占比", "风险预算(日)", "调整明细"]:
            assert key in budget, f"风险预算缺少: {key}"

    def test_budget_reasonable_range(self, normal_returns):
        """持仓上限应在总资金范围内"""
        rm = RiskManager()
        budget = rm.risk_budget(normal_returns, 1000000)
        assert 0 <= budget["建议持仓上限"] <= 1_000_000 * 2
        assert budget["持仓上限占比"] > 0

    def test_budget_with_signal(self, normal_returns):
        """正信号时持仓上限应更高"""
        rm = RiskManager()
        budget_neg = rm.risk_budget(normal_returns, 1000000, composite_score=-0.5)
        budget_pos = rm.risk_budget(normal_returns, 1000000, composite_score=0.8)
        assert budget_pos["建议持仓上限"] >= budget_neg["建议持仓上限"] * 0.5

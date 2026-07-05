"""
因子归因模块测试
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.factor_attribution import FactorAttribution


@pytest.fixture
def sample_returns_and_factors():
    """生成模拟的ETF收益和因子收益数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")

    # ETF收益 = 0.8*市场 + 0.3*小盘 + 随机噪声
    market = np.random.randn(500) * 0.01
    size = np.random.randn(500) * 0.008
    noise = np.random.randn(500) * 0.005
    etf_ret = 0.8 * market + 0.3 * size + noise

    etf_returns = pd.Series(etf_ret, index=dates, name="ETF")
    market_ret = pd.Series(market, index=dates, name="market")
    size_ret = pd.Series(size, index=dates, name="size")

    return etf_returns, {"market": market_ret, "size": size_ret}


class TestFactorAttribution:
    """因子归因核心功能测试"""

    def test_prepare_factor_data(self, sample_returns_and_factors):
        """prepare_factor_data 返回预期结构"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        result = fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        assert fa.factor_data is not None
        assert "return" in result.columns
        for name in factor_dict:
            assert name in result.columns

    def test_run_regression_returns_keys(self, sample_returns_and_factors):
        """run_regression 返回完整的回归结果"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        result = fa.run_regression()
        for key in ["因子", "系数", "t值", "p值", "R²", "调整R²", "F值"]:
            assert key in result, f"缺少: {key}"

    def test_regression_known_beta(self, sample_returns_and_factors):
        """当ETF收益由已知因子生成时，回归系数应接近真实值"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        result = fa.run_regression()

        # 真实beta: market=0.8, size=0.3
        coefs = dict(zip(result["因子"], result["系数"]))
        assert abs(coefs.get("market", 0) - 0.8) < 0.15, f"market beta={coefs.get('market', 0):.3f}"
        assert abs(coefs.get("size", 0) - 0.3) < 0.15, f"size beta={coefs.get('size', 0):.3f}"

    def test_factor_contribution_sums_to_one(self, sample_returns_and_factors):
        """贡献度之和应≈1"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        total = contrib["贡献度"].sum()
        assert abs(total - 1.0) < 0.01, f"贡献度和={total:.4f}"

    def test_factor_contribution_columns(self, sample_returns_and_factors):
        """factor_contribution 返回正确的列"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        for col in ["因子", "系数", "t值", "p值", "贡献度"]:
            assert col in contrib.columns, f"缺少: {col}"

    def test_decompose_etf_return(self, sample_returns_and_factors):
        """收益分解应接近实际收益"""
        etf_ret, factor_dict = sample_returns_and_factors
        factor_returns = pd.DataFrame(factor_dict)
        fa = FactorAttribution()
        decomp = fa.decompose_etf_return(etf_ret, factor_returns)
        assert "实际收益" in decomp.columns
        assert "Alpha" in decomp.columns
        # 解释收益 + 残差 ≈ 实际收益
        residual = (decomp["实际收益"] - decomp["解释收益"]).abs()
        assert residual.mean() < 0.02, f"平均残差={residual.mean():.4f}"

    def test_rolling_factor_beta_shape(self, sample_returns_and_factors):
        """滚动Beta返回预期的形状"""
        etf_ret, factor_dict = sample_returns_and_factors
        factor_returns = pd.DataFrame(factor_dict)
        result = FactorAttribution.rolling_factor_beta(etf_ret, factor_returns, window=60)
        expected_rows = len(etf_ret) - 60
        assert len(result) == expected_rows, f"行数={len(result)}, 期望={expected_rows}"
        for name in factor_dict:
            assert f"{name}_beta" in result.columns, f"缺少{name}_beta"

    def test_summary_text_contains_r2(self, sample_returns_and_factors):
        """summary_text 包含R²（传入r_squared参数）"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        r2 = fa.regression_result["R²"]
        text = fa.summary_text(contrib, r_squared=r2)
        assert "R²" in text
        assert f"{r2:.4f}" in text

    def test_summary_text_no_r2_still_works(self, sample_returns_and_factors):
        """summary_text 不传r_squared也不崩溃"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        text = fa.summary_text(contrib)  # no r_squared
        assert "红利ETF因子归因分析" in text
        assert "market" in text or "market" in text

    def test_r2_not_sum_of_coefficients(self, sample_returns_and_factors):
        """验证R²≠系数之和（修复前的问题）"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        r2 = fa.regression_result["R²"]
        coef_sum = contrib["系数"].sum()
        # 系数之和与R²的差异应明显
        assert abs(r2 - coef_sum) > 0.01, f"R²={r2:.4f}, 系数和={coef_sum:.4f}（不应相等）"


class TestAnalyzeETFExposure:
    """ETF因子暴露分析测试"""

    def test_analyze_exposure_returns_keys(self):
        """analyze_etf_factor_exposure 返回必要字段"""
        holdings = pd.DataFrame({
            "stock_code": ["000001", "000002"],
            "stock_name": ["平安银行", "万科A"],
            "weight": [0.6, 0.4],
        })
        market_data = pd.DataFrame({
            "dividend_yield": [0.05, 0.04],
            "book_to_price": [0.8, 0.6],
            "volatility": [0.20, 0.25],
        })
        result = FactorAttribution.analyze_etf_factor_exposure(holdings, market_data)
        for key in ["股息率暴露", "价值因子暴露", "低波因子暴露", "解释"]:
            assert key in result, f"缺少: {key}"


class TestSummaryTextRegression:
    """R²修复的回归测试：确保旧问题不重现"""

    def test_factor_contrib_does_not_contain_intercept(self, sample_returns_and_factors):
        """factor_contribution 不应包含截距项"""
        etf_ret, factor_dict = sample_returns_and_factors
        fa = FactorAttribution()
        fa.prepare_factor_data(etf_ret.to_frame("ETF"), factor_dict)
        fa.run_regression()
        contrib = fa.factor_contribution()
        assert "截距" not in contrib["因子"].values, "贡献度中不应包含截距"

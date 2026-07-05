"""
因子归因分析 (Barra多因子模型思维)
拆解红利ETF的超额收益来源：纯红利因子 vs. 低波因子 vs. 价值因子
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


@dataclass
class FactorExposure:
    """因子暴露"""
    div_yield: float = 0.0     # 股息率因子
    book_to_price: float = 0.0  # 账面市值比（价值因子）
    volatility: float = 0.0     # 低波因子
    size: float = 0.0           # 规模因子
    momentum: float = 0.0       # 动量因子


class FactorAttribution:
    """
    对红利ETF进行因子归因分析
    模型: Y = α + β1*DivYield + β2*BP + β3*Volatility + β4*Size + β5*Momentum + ε
    """

    def __init__(self):
        self.factor_data: Optional[pd.DataFrame] = None
        self.regression_result: Optional[dict] = None

    def prepare_factor_data(self, stock_returns: pd.DataFrame,
                             factor_values: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        准备因子数据
        Parameters
        ----------
        stock_returns : DataFrame index=date, columns=stock_code
        factor_values : dict of {factor_name: Series(index=date, value)}
        """
        df = pd.DataFrame(index=stock_returns.index)
        for name, series in factor_values.items():
            df[name] = series
        # 平均收益作为Y
        df["return"] = stock_returns.mean(axis=1)
        self.factor_data = df.dropna()
        return self.factor_data

    def run_regression(self, y_col: str = "return") -> dict:
        """
        运行多元线性回归
        Returns
        -------
        dict: coefficients, t-stats, r_squared, etc.
        """
        if self.factor_data is None:
            raise ValueError("请先准备因子数据")

        X_cols = [c for c in self.factor_data.columns if c != y_col]
        X = sm.add_constant(self.factor_data[X_cols].values)
        y = self.factor_data[y_col].values

        model = sm.OLS(y, X).fit()

        self.regression_result = {
            "因子": ["截距"] + X_cols,
            "系数": model.params.tolist(),
            "t值": model.tvalues.tolist(),
            "p值": model.pvalues.tolist(),
            "R²": model.rsquared,
            "调整R²": model.rsquared_adj,
            "F值": model.fvalue,
        }
        return self.regression_result

    def factor_contribution(self) -> pd.DataFrame:
        """计算各因子对收益的贡献度"""
        if self.regression_result is None:
            self.run_regression()

        df = pd.DataFrame(self.regression_result)
        # 去掉截距
        contrib = df[df["因子"] != "截距"].copy()
        abs_coef = contrib["系数"].abs()
        contrib["贡献度"] = abs_coef / abs_coef.sum()
        contrib = contrib.sort_values("贡献度", ascending=False)
        return contrib[["因子", "系数", "t值", "p值", "贡献度"]]

    @staticmethod
    def analyze_etf_factor_exposure(etf_holdings: pd.DataFrame,
                                     market_data: pd.DataFrame) -> dict:
        """
        分析单只ETF的因子暴露
        Parameters
        ----------
        etf_holdings : DataFrame, columns=['stock_code','weight','stock_name']
        market_data : DataFrame with stock characteristics
        """
        # 加权平均计算ETF层面的因子暴露
        exposure = FactorExposure()
        total_weight = etf_holdings["weight"].sum()

        if "dividend_yield" in market_data.columns:
            exposure.div_yield = (
                etf_holdings["weight"] * market_data["dividend_yield"]
            ).sum() / total_weight

        if "book_to_price" in market_data.columns:
            exposure.book_to_price = (
                etf_holdings["weight"] * market_data["book_to_price"]
            ).sum() / total_weight

        if "volatility" in market_data.columns:
            exposure.volatility = (
                etf_holdings["weight"] * market_data["volatility"]
            ).sum() / total_weight

        return {
            "股息率暴露": f"{exposure.div_yield:.4f}",
            "价值因子暴露": f"{exposure.book_to_price:.4f}",
            "低波因子暴露": f"{exposure.volatility:.4f}",
            "解释": FactorAttribution._explain_exposure(exposure),
        }

    @staticmethod
    def _explain_exposure(exposure: FactorExposure) -> str:
        """解释因子暴露含义"""
        insights = []
        if exposure.div_yield > 0.03:
            insights.append("高股息暴露 → 利率敏感型，利率下行期受益")
        if exposure.book_to_price > 0.6:
            insights.append("深价值暴露 → 估值修复期表现好")
        if exposure.volatility < 0.25:
            insights.append("低波暴露 → 防御性强，下跌市中抗跌")
        return " | ".join(insights) if insights else "混合因子暴露"

    def decompose_etf_return(self, etf_returns: pd.Series,
                              factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        对ETF收益进行因子分解
        Y(t) = α + Σ βi * Fi(t) + ε(t)
        Parameters
        ----------
        etf_returns : Series of ETF daily returns
        factor_returns : DataFrame of factor returns (columns=factor names)
        """
        # 对齐
        df = factor_returns.copy()
        df["ETF_return"] = etf_returns
        df = df.dropna()

        X = sm.add_constant(df.drop(columns=["ETF_return"]))
        y = df["ETF_return"]
        model = sm.OLS(y, X).fit()

        # 分解收益
        decomp = pd.DataFrame(index=df.index)
        decomp["实际收益"] = y
        decomp["解释收益"] = model.predict(X)
        decomp["Alpha"] = model.params["const"]
        decomp["残差"] = y - decomp["解释收益"]

        for col in df.drop(columns=["ETF_return"]).columns:
            decomp[f"{col}_贡献"] = model.params[col] * df[col]

        return decomp

    @staticmethod
    def summary_text(factor_contrib: pd.DataFrame, r_squared: float = None) -> str:
        """生成因子归因的文字总结"""
        lines = ["=" * 50, "红利ETF因子归因分析", "=" * 50]
        for _, row in factor_contrib.iterrows():
            sig = "***" if row["p值"] < 0.01 else "**" if row["p值"] < 0.05 else "*" if row["p值"] < 0.1 else ""
            lines.append(
                f"{row['因子']:15s}  系数={row['系数']:+.4f}  t={row['t值']:+.2f}  "
                f"贡献度={row['贡献度']:.1%}{sig}"
            )
        if r_squared is not None:
            lines.append(f"\nR² = {r_squared:.4f}")
        lines.append("\n结论: 红利ETF的超额收益中")
        lines.append("  如果有显著的低波因子暴露 → 说明涨是因低波而非高股息")
        lines.append("  如果有显著的价值因子暴露 → 说明涨是因估值修复而非分红")
        return "\n".join(lines)

    @staticmethod
    def rolling_factor_beta(etf_returns: pd.Series,
                             factor_returns: pd.DataFrame,
                             window: int = 60) -> pd.DataFrame:
        """
        滚动计算因子Beta（时变因子暴露）
        Parameters
        ----------
        window : int  滚动窗口天数，默认60天
        """
        df = factor_returns.copy()
        df["ETF_return"] = etf_returns
        df = df.dropna()

        results = []
        for i in range(window, len(df)):
            chunk = df.iloc[i - window:i]
            X = sm.add_constant(chunk.drop(columns=["ETF_return"]))
            y = chunk["ETF_return"]
            model = sm.OLS(y, X).fit()
            row = {"date": chunk.index[-1]}
            for j, name in enumerate(X.columns):
                row[f"{name}_beta"] = model.params.iloc[j]
            results.append(row)

        return pd.DataFrame(results).set_index("date")

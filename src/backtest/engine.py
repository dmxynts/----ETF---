"""
回测引擎
支持策略回测、绩效评价、参数敏感性分析
"""
import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """回测结果"""
    nav: pd.DataFrame
    metrics: Dict[str, str]
    trades: Optional[pd.DataFrame] = None
    annual_returns: Optional[pd.Series] = None


class BacktestEngine:
    """
    回测引擎
    支持自定义策略函数、多标的、多周期
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.results: Dict[str, BacktestResult] = {}

    def run(self, price_data: pd.DataFrame, signal_data: pd.DataFrame,
             name: str = "策略", fee_rate: float = 0.0003,
             risk_free_rate: float = 0.02) -> BacktestResult:
        """
        运行回测
        Parameters
        ----------
        price_data : DataFrame with ['date', 'close']
        signal_data : DataFrame with ['date', 'position']  仓位信号
        fee_rate : 交易费率（默认万3）
        risk_free_rate : 年化无风险利率（默认2%）
        """
        # 对齐
        df = pd.merge(
            price_data[["date", "close"]],
            signal_data[["date", "position"]],
            on="date", how="inner",
        )
        df = df.sort_values("date").reset_index(drop=True)

        # 日收益计算
        df["return"] = df["close"].pct_change()
        df["strategy_return"] = df["return"] * df["position"].shift(1)
        df.loc[0, "strategy_return"] = 0

        # 交易成本（仓位变化时收取，基于 T-1 日信号的变化量）
        # position[t-1] 是当日使用的信号，相对 position[t-2] 的变化产生调仓成本
        df["position_shift"] = df["position"].shift(1).fillna(0).diff().fillna(0).abs()
        df["trade_cost"] = df["position_shift"] * fee_rate
        df["strategy_return"] -= df["trade_cost"]

        # 净值
        df["nav"] = (1 + df["strategy_return"]).cumprod() * self.initial_capital
        df["index_nav"] = (1 + df["return"].fillna(0)).cumprod() * self.initial_capital

        # 回撤
        cummax = df["nav"].cummax()
        df["drawdown"] = (df["nav"] - cummax) / cummax

        # 计算评价指标
        metrics = self._compute_metrics(df)

        result = BacktestResult(
            nav=df[["date", "nav", "index_nav", "drawdown", "position"]],
            metrics=metrics,
        )
        self.results[name] = result
        return result

    def run_multi_asset(self, price_dict: Dict[str, pd.DataFrame],
                         signal_dict: Dict[str, pd.DataFrame],
                         weights: Dict[str, float],
                         risk_free_rate: float = 0.02) -> BacktestResult:
        """
        多资产组合回测
        Parameters
        ----------
        price_dict : {asset_name: price_df}
        signal_dict : {asset_name: signal_df}
        weights : {asset_name: weight}
        risk_free_rate : 年化无风险利率（默认2%）
        """
        portfolio_returns = None
        combined_date = None

        for asset in price_dict:
            if asset not in signal_dict:
                continue
            w = weights.get(asset, 0)
            if w == 0:
                continue

            df = pd.merge(
                price_dict[asset][["date", "close"]],
                signal_dict[asset][["date", "position"]],
                on="date", how="inner",
            )
            df["return"] = df["close"].pct_change()
            df["strategy_return"] = df["return"] * df["position"].shift(1)
            df.loc[0, "strategy_return"] = 0

            if portfolio_returns is None:
                portfolio_returns = df[["date", "strategy_return"]].copy()
                portfolio_returns = portfolio_returns.rename(
                    columns={"strategy_return": "return"}
                )
                portfolio_returns["return"] *= w
            else:
                temp = df[["date", "strategy_return"]].rename(
                    columns={"strategy_return": "return"}
                )
                portfolio_returns = pd.merge(
                    portfolio_returns, temp, on="date", how="outer"
                )
                portfolio_returns["return"] = (
                    portfolio_returns["return_x"].fillna(0)
                    + portfolio_returns["return_y"].fillna(0) * w
                )

        if portfolio_returns is None:
            raise ValueError("无有效资产")

        portfolio_returns = portfolio_returns.sort_values("date").reset_index(drop=True)
        portfolio_returns["nav"] = (1 + portfolio_returns["return"]).cumprod() * self.initial_capital
        cummax = portfolio_returns["nav"].cummax()
        portfolio_returns["drawdown"] = (portfolio_returns["nav"] - cummax) / cummax

        metrics = self._compute_metrics(portfolio_returns, risk_free_rate)
        return BacktestResult(
            nav=portfolio_returns[["date", "nav", "drawdown"]],
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, str]:
        """计算绩效指标"""
        ret = df["strategy_return"].dropna() if "strategy_return" in df.columns else df["return"].dropna()

        total_ret = df["nav"].iloc[-1] / df["nav"].iloc[0] - 1 if "nav" in df.columns else 0
        n_years = len(ret) / 252
        annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = ret.std() * np.sqrt(252)
        daily_rf = risk_free_rate / 252
        sharpe = (ret.mean() - daily_rf) / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        max_dd = df["drawdown"].min() if "drawdown" in df.columns else 0

        # 卡尔玛比率
        calmar = abs(annual_ret / max_dd) if max_dd != 0 else 0

        # 胜率
        win_rate = (ret > 0).mean()

        # 盈亏比
        avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0
        avg_loss = abs(ret[ret < 0].mean()) if (ret < 0).any() else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        return {
            "总收益率": f"{total_ret:.2%}",
            "年化收益率": f"{annual_ret:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "卡尔玛比率": f"{calmar:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "日胜率": f"{win_rate:.2%}",
            "盈亏比": f"{profit_loss_ratio:.2f}",
        }

    @staticmethod
    def benchmark_comparison(result_df: pd.DataFrame, benchmark_col: str = "index_nav") -> dict:
        """与基准比较"""
        if "nav" not in result_df.columns:
            return {}

        strategy_ret = result_df["nav"].iloc[-1] / result_df["nav"].iloc[0] - 1
        benchmark_ret = result_df[benchmark_col].iloc[-1] / result_df[benchmark_col].iloc[0] - 1

        # 超额
        excess = strategy_ret - benchmark_ret

        # 跟踪误差
        strategy_daily = result_df["nav"].pct_change()
        benchmark_daily = result_df[benchmark_col].pct_change()
        te = (strategy_daily - benchmark_daily).std() * np.sqrt(252)

        # 信息比率
        ir = excess / te if te > 0 else 0

        return {
            "策略收益率": f"{strategy_ret:.2%}",
            "基准收益率": f"{benchmark_ret:.2%}",
            "超额收益": f"{excess:.2%}",
            "跟踪误差": f"{te:.2%}",
            "信息比率": f"{ir:.2f}",
        }

    @staticmethod
    def sensitivity_analysis(price_df: pd.DataFrame,
                              signal_func: Callable,
                              param_grid: Dict[str, list]) -> pd.DataFrame:
        """
        参数敏感性分析
        Parameters
        ----------
        price_df : DataFrame
        signal_func : Callable   signal_func(param_dict) -> signal_df
        param_grid : {'param_name': [values]}
        """
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            try:
                signals = signal_func(params)
                engine = BacktestEngine()
                result = engine.run(price_df, signals, name="test")
                metrics = result.metrics
                results.append({
                    **params,
                    "年化收益": metrics["年化收益率"],
                    "夏普比率": metrics["夏普比率"],
                    "最大回撤": metrics["最大回撤"],
                })
            except Exception as e:
                results.append({**params, "error": str(e)})

        return pd.DataFrame(results)

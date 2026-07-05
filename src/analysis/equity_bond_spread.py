"""
股债利差择时系统
核心逻辑: 红利ETF股息率 - 十年期国债收益率 → 利差滚动分位 → 仓位信号
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class EquityBondSpread:
    """
    股债利差模型 (FED Model 改良版)
    - 计算: 股息率(DivYield) - 无风险利率(BondYield)
    - 滚动百分位判断当前利差处于历史什么位置
    - 高于阈值 → 满仓, 低于阈值 → 轻仓
    """

    def __init__(self, lookback_years: int = 10,
                 high_percentile: float = 0.80, low_percentile: float = 0.20,
                 full_position: float = 1.0, light_position: float = 0.3):
        self.lookback_years = lookback_years
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        self.full_position = full_position
        self.light_position = light_position
        self.spread_data: Optional[pd.DataFrame] = None
        self.signal_data: Optional[pd.DataFrame] = None

    def compute_spread(self, div_yield_df: pd.DataFrame, bond_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算股债利差 = 股息率 - 国债收益率
        Parameters
        ----------
        div_yield_df : DataFrame with columns ['date','dividend_yield']
        bond_df : DataFrame with columns ['date','yield']

        Returns
        -------
        DataFrame with columns: date, div_yield, bond_yield, spread
        """
        # 合并数据
        df = pd.merge(
            div_yield_df[["date", "dividend_yield"]],
            bond_df[["date", "yield"]],
            on="date",
            how="inner",
        )
        df = df.dropna().sort_values("date").reset_index(drop=True)
        # 统一为小数
        if df["dividend_yield"].max() > 1:
            df["dividend_yield"] = df["dividend_yield"] / 100.0
        if df["yield"].max() > 1:
            df["yield"] = df["yield"] / 100.0
        df["spread"] = df["dividend_yield"] - df["yield"]
        self.spread_data = df
        return df

    def compute_percentile(self, window_years: int = None) -> pd.DataFrame:
        """
        计算利差的滚动百分位
        如果数据不足指定窗口，自动使用可用数据长度
        Returns
        -------
        DataFrame: date, spread, percentile, z_score
        """
        if self.spread_data is None:
            raise ValueError("请先调用 compute_spread()")

        window = window_years or self.lookback_years
        window_days = int(window * 252)

        df = self.spread_data.copy()
        n = len(df)

        # 如果数据少于窗口，用全部数据作为窗口
        if n < window_days:
            window_days = min(n, 60)  # 至少60天

        df["rolling_rank"] = df["spread"].rolling(window=window_days).apply(
            lambda x: pd.Series(x.values[:-1]).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5
        )
        df["z_score"] = df["spread"].rolling(window=window_days).apply(
            lambda x: (x.values[-1] - np.mean(x.values[:-1])) / np.std(x.values[:-1]) if len(x) > 1 else 0
        )
        # 前向填充开头的NaN
        df["rolling_rank"] = df["rolling_rank"].ffill()
        df["percentile"] = df["rolling_rank"].fillna(0.5)  # 实在没有就中性
        self.spread_data = df
        return df[["date", "spread", "percentile", "z_score"]]

    def generate_signal(self, percentile_col: str = "percentile") -> pd.DataFrame:
        """
        根据利差分位生成仓位信号
        - 百分位 > high_percentile → 满仓 (1.0)
        - 百分位 < low_percentile  → 轻仓 (0.3)
        - 中间 → 线性插值
        """
        if self.spread_data is None or percentile_col not in self.spread_data.columns:
            self.compute_percentile()

        df = self.spread_data.copy()
        p = df[percentile_col]

        # 分段线性仓位
        df["position"] = np.where(
            p >= self.high_percentile, self.full_position,
            np.where(
                p <= self.low_percentile, self.light_position,
                self.light_position +  # 线性插值
                (p - self.low_percentile) / (self.high_percentile - self.low_percentile) * (self.full_position - self.light_position)
            )
        )

        # 信号变化记录
        df["signal"] = "持有"
        df.loc[p >= self.high_percentile, "signal"] = "加仓/满仓"
        df.loc[p <= self.low_percentile, "signal"] = "减仓/轻仓"
        self.signal_data = df
        return df[["date", "spread", "percentile", "position", "signal"]]

    def backtest(self, index_df: pd.DataFrame, fee_rate: float = 0.0003) -> pd.DataFrame:
        """
        回测算净值（委托 BacktestEngine 执行，含交易成本）

        Parameters
        ----------
        index_df : DataFrame with columns ['date','close']
        fee_rate : float  交易费率（默认万3）

        Returns
        -------
        DataFrame: date, strategy_nav, index_nav, strategy_return,
                   index_return, position, excess_return
        """
        if self.signal_data is None:
            self.generate_signal()

        from src.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=1.0)
        result = engine.run(index_df, self.signal_data, name="spread", fee_rate=fee_rate)

        nav = result.nav
        m = pd.DataFrame({
            "date": nav["date"],
            "strategy_nav": nav["nav"],
            "index_nav": nav["index_nav"],
            "position": nav["position"],
        })
        m["strategy_return"] = m["strategy_nav"].pct_change().fillna(0)
        m["index_return"] = m["index_nav"].pct_change().fillna(0)
        m["excess_return"] = m["strategy_nav"] - m["index_nav"]

        return m[["date", "strategy_nav", "index_nav", "strategy_return",
                  "index_return", "position", "excess_return"]]

    def get_metrics(self, backtest_df: pd.DataFrame) -> dict:
        """
        计算策略评价指标
        """
        ret = backtest_df["strategy_return"].dropna()
        idx_ret = backtest_df["index_return"].dropna()

        total_return = backtest_df["strategy_nav"].iloc[-1] - 1
        idx_total = backtest_df["index_nav"].iloc[-1] - 1

        # 年化收益率
        n_years = len(ret) / 252
        annual_ret = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 年化波动率
        annual_vol = ret.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率2%）
        sharpe = (ret.mean() - 0.02 / 252) / ret.std() * np.sqrt(252) if ret.std() > 0 else 0

        # 最大回撤
        cummax = backtest_df["strategy_nav"].cummax()
        drawdown = (backtest_df["strategy_nav"] - cummax) / cummax
        max_dd = drawdown.min()

        # 胜率
        win_rate = (ret > 0).mean()

        # 年化超额
        excess_annual = total_return - idx_total

        return {
            "策略总收益率": f"{total_return:.2%}",
            "基准总收益率": f"{idx_total:.2%}",
            "年化收益率": f"{annual_ret:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "日胜率": f"{win_rate:.2%}",
            "年化超额收益": f"{excess_annual:.2%}",
        }

    @staticmethod
    def current_judgment(div_yield: float, bond_yield: float,
                         historical_spreads: pd.Series) -> dict:
        """
        实时判断当前股债利差水平
        Parameters
        ----------
        div_yield : float  当前股息率（小数）
        bond_yield : float 当前国债收益率（小数）
        historical_spreads : Series 历史利差序列
        """
        spread = div_yield - bond_yield
        percentile = (historical_spreads < spread).mean()
        z = (spread - historical_spreads.mean()) / historical_spreads.std()

        if percentile >= 0.8:
            signal = "🔴 利差处于历史高分位 → 积极加仓"
        elif percentile <= 0.2:
            signal = "🟢 利差处于历史低分位 → 注意减仓"
        else:
            signal = "🟡 利差处于正常区间 → 持有"

        return {
            "当前利差": f"{spread:.2%}",
            "历史分位": f"{percentile:.1%}",
            "Z-Score": f"{z:.2f}",
            "操作建议": signal,
        }

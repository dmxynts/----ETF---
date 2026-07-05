"""
网格交易优化模块
结合卡尔曼滤波动态估计均衡价格，用优化算法确定最优网格参数
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize


class KalmanFilter:
    """
    卡尔曼滤波估计短期均衡价格
    状态: 均衡价格 (random walk)
    观测: 实际价格
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        """
        Parameters
        ----------
        delta : float  状态转移方差（越小越平滑）
        R : float      观测噪声方差
        """
        self.delta = delta
        self.R = R

    def fit(self, prices: np.ndarray) -> np.ndarray:
        """
        估计均衡价格序列
        Returns
        -------
        filtered_state_means : 均衡价格序列
        """
        import pykalman

        n = len(prices)
        kf = pykalman.KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=prices[0],
            initial_state_covariance=1.0,
            transition_covariance=self.delta,
            observation_covariance=self.R,
        )
        filtered_state_means, _ = kf.filter(prices)
        return filtered_state_means.flatten()

    def estimate_noise(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        估计最优的delta和R参数（EM算法）
        """
        from pykalman import KalmanFilter

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=prices[0],
            n_dim_obs=1,
        )
        kf = kf.em(prices, n_iter=10)
        return kf.transition_covariance[0, 0], kf.observation_covariance[0, 0]


class GridOptimizer:
    """
    网格交易参数优化器
    使用历史数据计算最优网格间距和每格仓位
    目标: 最大化夏普比率 / 卡尔玛比率
    """

    def __init__(self, num_grids: int = 10, atr_period: int = 20,
                 slippage: float = 0.0001, fee_rate: float = 0.0003):
        self.num_grids = num_grids
        self.atr_period = atr_period
        self.slippage = slippage
        self.fee_rate = fee_rate
        self.optimal_params: Optional[dict] = None

    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """计算平均真实波幅 (ATR)"""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        # 简单移动平均
        atr = np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period else np.mean(tr)
        return float(atr)

    def calibrate_grid_levels(self, prices: pd.DataFrame,
                               kalman_delta: float = 1e-4) -> dict:
        """
        基于卡尔曼滤波和ATR校准网格
        Parameters
        ----------
        prices : DataFrame with columns ['close','high','low']

        Returns
        -------
        dict: grid_center, grid_spacing, grid_levels
        """
        close = prices["close"].values
        high = prices["high"].values
        low = prices["low"].values

        # 卡尔曼滤波求均衡价格
        kf = KalmanFilter(delta=kalman_delta)
        equilibrium = kf.fit(close)

        # ATR作为波动率度量
        atr = self._compute_atr(high, low, close)

        # 网格间距 = ATR × 系数（防止过密）
        grid_spacing = atr * 1.5

        # 网格中心 = 最新均衡价
        grid_center = float(equilibrium[-1])

        # 生成网格层级
        grid_levels = []
        for i in range(self.num_grids // 2, 0, -1):
            grid_levels.append(grid_center - i * grid_spacing)
        grid_levels.append(grid_center)
        for i in range(1, self.num_grids // 2 + 1):
            grid_levels.append(grid_center + i * grid_spacing)

        self.optimal_params = {
            "网格中心": grid_center,
            "网格间距": grid_spacing,
            "ATR": atr,
            "网格层数": self.num_grids,
            "网格价格": [round(p, 3) for p in grid_levels],
        }
        return self.optimal_params

    def simulate_grid(self, price_series: pd.Series, grid_levels: list,
                       capital: float = 100000, grid_ratio: float = 0.1) -> pd.DataFrame:
        """
        模拟网格交易（含滑点和交易费用）
        Parameters
        ----------
        price_series : 价格序列
        grid_levels : 网格价格列表（从小到大）
        capital : 初始资金
        grid_ratio : 每格资金比例
        """
        cash = capital
        position = 0.0
        total_slippage = 0.0
        total_fee = 0.0
        n_levels = len(grid_levels)
        trade_records = []
        nav = []

        # 映射价格→格子索引
        level_to_idx = {round(lvl, 4): j for j, lvl in enumerate(grid_levels)}

        prev_level = -1  # 上次所在格子

        for t, price in enumerate(price_series):
            # 找到当前价格在哪个网格区间
            current_level = -1
            for j, lvl in enumerate(grid_levels):
                if price >= lvl:
                    current_level = j

            # 网格触发: 穿过了某个格子边界
            if prev_level != -1 and current_level != prev_level:
                # 上穿（价格上涨）→ 卖出
                if current_level > prev_level:
                    for lvl_idx in range(prev_level + 1, current_level + 1):
                        sell_price = grid_levels[lvl_idx]
                        exec_price = sell_price * (1 - self.slippage)  # 滑点: 卖价变低
                        sell_amount = capital * grid_ratio / n_levels
                        shares = sell_amount / exec_price
                        actual_sell = min(shares, position)
                        position -= actual_sell
                        trade_value = actual_sell * exec_price
                        fee = trade_value * self.fee_rate
                        slippage_cost = actual_sell * (sell_price - exec_price)
                        total_slippage += slippage_cost
                        total_fee += fee
                        cash += trade_value - fee
                        trade_records.append({
                            "date": t, "type": "sell", "price": sell_price,
                            "exec_price": exec_price, "shares": actual_sell,
                            "fee": fee, "slippage_cost": slippage_cost, "cash": cash,
                        })
                # 下穿（价格下跌）→ 买入
                else:
                    for lvl_idx in range(prev_level - 1, current_level - 1, -1):
                        buy_price = grid_levels[lvl_idx]
                        exec_price = buy_price * (1 + self.slippage)  # 滑点: 买价变高
                        buy_amount = capital * grid_ratio / n_levels
                        actual_buy = min(buy_amount / exec_price, cash / exec_price)
                        position += actual_buy
                        trade_value = actual_buy * exec_price
                        fee = trade_value * self.fee_rate
                        slippage_cost = actual_buy * (exec_price - buy_price)
                        total_slippage += slippage_cost
                        total_fee += fee
                        cash -= trade_value + fee
                        trade_records.append({
                            "date": t, "type": "buy", "price": buy_price,
                            "exec_price": exec_price, "shares": actual_buy,
                            "fee": fee, "slippage_cost": slippage_cost, "cash": cash,
                        })

            # 记录净资产
            total_value = cash + position * price
            nav.append({
                "date": t, "nav": total_value, "position": position, "cash": cash,
                "cumulative_slippage_cost": total_slippage,
                "cumulative_fee_cost": total_fee,
            })
            prev_level = current_level

        return pd.DataFrame(nav)

    def optimize_grid_by_sharpe(self, price_series: pd.Series,
                                  grid_levels: list) -> dict:
        """
        用优化算法搜索最优网格间距（最大化夏普比率）
        """
        from scipy.optimize import minimize_scalar

        def objective(spacing_mult):
            # 重建网格
            center = grid_levels[len(grid_levels) // 2] if grid_levels else price_series.iloc[-1]
            levels = []
            half = self.num_grids // 2
            atr_est = price_series.diff().abs().rolling(20).mean().iloc[-1]
            spacing = atr_est * spacing_mult
            for i in range(half, 0, -1):
                levels.append(center - i * spacing)
            levels.append(center)
            for i in range(1, half + 1):
                levels.append(center + i * spacing)

            nav_df = self.simulate_grid(price_series, levels)
            if nav_df.empty or len(nav_df) < 10:
                return 999

            returns = nav_df["nav"].pct_change().dropna()
            if returns.std() == 0:
                return 999

            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            return -sharpe  # 最小化负夏普

        result = minimize_scalar(objective, bounds=(0.5, 4.0), method="bounded")
        return {"最优间距乘数": result.x, "最优夏普": -result.fun}

    @staticmethod
    def generate_grid_report(optimal_params: dict) -> str:
        """生成网格交易参数报告"""
        lines = [
            "=" * 50,
            "网格交易优化参数",
            "=" * 50,
            f"网格中心价格: {optimal_params.get('网格中心', 'N/A'):.3f}",
            f"网格间距: {optimal_params.get('网格间距', 'N/A'):.4f}",
            f"网格层数: {optimal_params.get('网格层数', 'N/A')}",
            f"ATR(20): {optimal_params.get('ATR', 'N/A'):.4f}",
        ]

        levels = optimal_params.get("网格价格", [])
        if levels:
            lines.append("\n网格价格层级:")
            for i, p in enumerate(levels):
                direction = "↓买" if p <= optimal_params.get("网格中心", 0) else "↑卖"
                lines.append(f"  [{i + 1:2d}] {direction}  {p:.3f}")

        return "\n".join(lines)

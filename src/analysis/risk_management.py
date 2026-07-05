"""
风险管理模块
VaR (风险价值)、ES (条件风险价值)、极值理论(EVT)
行业集中度风险、尾部风险防范
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy import stats
from scipy.stats import genpareto


class RiskManager:
    """
    风险管理器
    提供VaR、ES、EVT等风险度量方法
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.var_results: Optional[dict] = None

    # -----------------------------------------------------------
    # 1. VaR 计算
    # -----------------------------------------------------------
    def var_historical(self, returns: pd.Series, confidence: float = None) -> float:
        """
        历史模拟法 VaR
        """
        conf = confidence or self.confidence_level
        return float(np.percentile(returns, (1 - conf) * 100))

    def var_gaussian(self, returns: pd.Series, confidence: float = None) -> float:
        """
        参数法 VaR（假设正态分布）
        """
        conf = confidence or self.confidence_level
        mu, sigma = returns.mean(), returns.std()
        z = stats.norm.ppf(1 - conf)
        return float(mu + z * sigma)

    def var_cornish_fisher(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Cornish-Fisher 扩展 VaR（修正偏度和峰度）
        比正态VaR更准确，考虑了红利ETF的偏峰特征
        """
        conf = confidence or self.confidence_level
        mu, sigma = returns.mean(), returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # 超额峰度

        z = stats.norm.ppf(1 - conf)
        # Cornish-Fisher 展开
        z_cf = (z + (z**2 - 1) * skew / 6
                + (z**3 - 3 * z) * kurt / 24
                - (2 * z**3 - 5 * z) * skew**2 / 36)
        return float(mu + z_cf * sigma)

    # -----------------------------------------------------------
    # 2. Expected Shortfall (ES)
    # -----------------------------------------------------------
    def expected_shortfall(self, returns: pd.Series, confidence: float = None) -> float:
        """
        条件风险价值 ES = E[损失 | 损失 > VaR]
        """
        conf = confidence or self.confidence_level
        var = self.var_historical(returns, conf)
        tail_losses = returns[returns <= var]
        return float(tail_losses.mean()) if len(tail_losses) > 0 else var

    # -----------------------------------------------------------
    # 3. 极值理论 (EVT)
    # -----------------------------------------------------------
    def evt_gev(self, returns: pd.Series, block_size: int = 21) -> dict:
        """
        极值理论 - 广义极值分布 (GEV)
        对收益率的月极大/极小值建模

        Parameters
        ----------
        block_size : int  块大小（天数），默认21个交易日约=1个月
        """
        # 取极小值（损失）
        n_blocks = len(returns) // block_size
        block_mins = []
        for i in range(n_blocks):
            block = returns.iloc[i * block_size:(i + 1) * block_size]
            block_mins.append(block.min())

        block_mins = np.array(block_mins) * -1  # 取正

        if len(block_mins) < 10:
            return {"warning": "数据不足，需要至少10个块"}

        # 拟合 GEV
        try:
            params = genpareto.fit(block_mins - 0.01)  # 平移保证正数
            shape, loc, scale = params

            # 计算极端 VaR
            conf = self.confidence_level
            var_evt = -stats.genpareto.ppf(conf, shape, loc=loc, scale=scale)
            return {
                "EVT_VaR": var_evt,
                "形状参数": shape,
                "位置参数": loc,
                "尺度参数": scale,
                "肥尾程度": "显著肥尾" if shape > 0 else "近似正态" if abs(shape) < 0.1 else "薄尾",
            }
        except Exception as e:
            return {"error": str(e)}

    # -----------------------------------------------------------
    # 4. 综合风险报告
    # -----------------------------------------------------------
    def full_risk_report(self, returns: pd.Series, holding_value: float = 1_000_000) -> dict:
        """
        生成完整的风险管理报告
        Parameters
        ----------
        returns : pd.Series  日收益率
        holding_value : float  持仓市值（默认100万）
        """
        confs = [0.95, 0.99]

        report = {"持仓市值": f"{holding_value:,.0f} 元"}

        for conf in confs:
            var_h = self.var_historical(returns, conf)
            var_g = self.var_gaussian(returns, conf)
            var_cf = self.var_cornish_fisher(returns, conf)
            es = self.expected_shortfall(returns, conf)

            report[f"VaR(历史法,{conf:.0%})"] = f"{var_h:.2%}"
            report[f"VaR(正态,{conf:.0%})"] = f"{var_g:.2%}"
            report[f"VaR(CF修正,{conf:.0%})"] = f"{var_cf:.2%}"
            report[f"CVaR/ES({conf:.0%})"] = f"{es:.2%}"
            report[f"日最大预期损失({conf:.0%})"] = f"{abs(es * holding_value):,.0f} 元"

        # 回撤监控 (深度+时长+速度)
        dd_info = self.drawdown_analysis(returns)
        report["当前回撤"] = f"{dd_info['当前回撤']:.2%}"
        report["历史最大回撤"] = f"{dd_info['历史最大回撤']:.2%}"
        report["回撤天数"] = f"{dd_info['回撤天数']}天"
        report["回撤趋势"] = f"{dd_info['趋势']} (速度比{dd_info['速度比']:+.2f})"
        report["回撤状态"] = dd_info["状态"]
        report["预警等级"] = dd_info["预警"]

        # 动态止损
        sl = self.dynamic_stop_loss(returns)
        report["建议止损线"] = f"{sl['建议止损线(%)']:.1f}%"
        report["止损依据"] = sl["止损依据"]

        # 分布特征
        report["日收益率偏度"] = f"{stats.skew(returns):.3f}"
        report["日收益率超额峰度"] = f"{stats.kurtosis(returns):.3f}"
        report["分布特征"] = (
            "右偏厚尾（走楼梯式上涨，适合长期持有）"
            if stats.skew(returns) > 0
            else "左偏厚尾（需要注意尾部风险）"
        )

        # 风险解读
        report["风险提示"] = self._risk_comment(report)

        self.var_results = report
        return report

    @staticmethod
    def _risk_comment(report: dict) -> str:
        """生成风险解读"""
        try:
            cvar_95 = float(report.get("CVaR/ES(95%)", "0%").strip("%")) / 100
            if abs(cvar_95) > 0.03:
                return "[注意] 尾部风险较大，建议考虑对冲（如做空股指期货或买入认沽期权）"
            elif abs(cvar_95) > 0.02:
                return "[正常] 风险可控，正常持有"
            else:
                return "[安全] 风险较低，适合作为底仓配置"
        except (ValueError, KeyError):
            return ""

    # -----------------------------------------------------------
    # 5. 动态回撤监控
    # -----------------------------------------------------------
    def drawdown_analysis(self, returns: pd.Series) -> dict:
        """
        回撤监控: 深度 / 时长 / 速度 / 预警

        Parameters
        ----------
        returns : pd.Series  日收益率（小数）

        Returns
        -------
        dict: 当前回撤、最大回撤、天数、速度、状态、预警等级
        """
        nav = (1 + returns).cumprod()
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax

        current_dd = drawdown.iloc[-1]
        max_dd = drawdown.min()

        # --- 回撤持续天数 ---
        peak_idx = drawdown[drawdown == 0].index
        if len(peak_idx) > 0:
            last_peak = peak_idx[-1]
            dd_duration = len(drawdown.loc[last_peak:]) - 1
        else:
            dd_duration = len(drawdown)

        # --- 回撤速度: 近5日 vs 近20日 平均收益 ---
        recent = returns.tail(20).mean()
        very_recent = returns.tail(5).mean() if len(returns) >= 5 else recent
        # 速度比 < -1 加速跌, 负值持续跌, 正值企稳/回升
        speed_ratio = very_recent / recent if recent < 0 else very_recent / max(recent, 0.001)

        # --- 状态分类 ---
        abs_dd = abs(current_dd)
        if current_dd == 0:
            status, alert = "创新高", "none"
        elif abs_dd < 0.03:
            status, alert = "小幅波动", "none"
        elif abs_dd < 0.05:
            status, alert = "正常回撤", "info"
        elif abs_dd < 0.10:
            status, alert = "中等回撤", "watch"
        elif abs_dd < 0.20:
            status, alert = "深度回撤", "warning"
        else:
            status, alert = "极端回撤", "danger"

        # --- 速度趋势 ---
        if speed_ratio < -1:
            trend = "加速下跌 ↑"
        elif speed_ratio < 0:
            trend = "持续下跌 →"
        elif speed_ratio < 0.5:
            trend = "减速企稳 ↓"
        else:
            trend = "震荡/回升 ←"

        return {
            "当前回撤": round(current_dd, 4),
            "历史最大回撤": round(max_dd, 4),
            "回撤天数": dd_duration,
            "速度比": round(speed_ratio, 3),
            "趋势": trend,
            "状态": status,
            "预警": alert,
        }

    def dynamic_stop_loss(self, returns: pd.Series,
                          conditional_vol: pd.Series = None) -> dict:
        """
        基于波动率的动态止损建议

        Parameters
        ----------
        returns : pd.Series  日收益率（小数）
        conditional_vol : pd.Series, optional  GARCH条件波动率（小数）

        Returns
        -------
        dict: 止损线、倍数、依据
        """
        # 当前波动率
        if conditional_vol is not None and len(conditional_vol) > 0:
            vol = conditional_vol.iloc[-1]
            vol_source = "GARCH"
        else:
            vol = returns.tail(60).std()
            vol_source = "历史60日"

        # 回撤越深，止损越紧
        dd_info = self.drawdown_analysis(returns)
        abs_dd = abs(dd_info["当前回撤"])

        if abs_dd < 0.03:
            mult = 3.0
        elif abs_dd < 0.08:
            mult = 2.5
        elif abs_dd < 0.15:
            mult = 2.0
        else:
            mult = 1.5

        stop_loss = vol * mult

        # 转换为持仓金额
        return {
            "止损倍数": mult,
            "当前波动率": round(vol, 4),
            "波动率来源": vol_source,
            "建议止损线(%)": round(stop_loss * 100, 2),
            "止损依据": f"{vol_source}波动率 {vol:.2%} × {mult:.0f}倍 (回撤越深越紧)",
        }

    # -----------------------------------------------------------
    # 6. 情景压力测试
    # -----------------------------------------------------------
    @staticmethod
    def stress_test(returns: pd.Series, holding_value: float = 1_000_000) -> dict:
        """
        情景压力测试: 历史最差区间 / 单日冲击 / 连续下跌

        Parameters
        ----------
        returns : pd.Series  日收益率（小数）
        holding_value : float  持仓市值

        Returns
        -------
        dict: 各情景下的预计损失
        """
        results = {}

        # --- 1. 历史最差 N 日区间 ---
        for label, days in [("5日最大跌幅", 5), ("10日最大跌幅", 10),
                            ("20日最大跌幅", 20), ("60日最大跌幅", 60)]:
            if len(returns) < days:
                continue
            # 滚动求和（累计收益 = 最差情况）
            rolling_ret = returns.rolling(days).sum()
            worst = rolling_ret.min()
            # 安全提取起始日
            try:
                idx = rolling_ret.idxmin()
                start_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
            except Exception:
                start_date = "N/A"
            results[label] = {
                "跌幅": round(worst, 4),
                "损失金额": int(abs(worst) * holding_value),
                "起始日": start_date,
            }

        # --- 2. 单日冲击 ---
        for shock in [-0.03, -0.05, -0.08]:
            label = f"单日{abs(shock)*100:.0f}%暴跌"
            results[label] = {
                "跌幅": shock,
                "损失金额": int(abs(shock) * holding_value),
            }

        # --- 3. 连续下跌 ---
        for label, daily_drop, duration in [
            ("连续5日跌1%", -0.01, 5),
            ("连续10日跌1%", -0.01, 10),
            ("连续5日跌2%", -0.02, 5),
        ]:
            total = daily_drop * duration
            results[label] = {
                "跌幅": round(total, 4),
                "损失金额": int(abs(total) * holding_value),
                "说明": f"每日{daily_drop:.0%} × {duration}天",
            }

        # --- 4. 最大回撤比例映射 ---
        nav = (1 + returns).cumprod()
        max_dd = nav.div(nav.cummax()).min() - 1
        results["历史最差回撤重演"] = {
            "跌幅": round(max_dd, 4),
            "损失金额": int(abs(max_dd) * holding_value),
            "说明": "你持有的ETF历史上最惨的时候",
        }

        return results

    # -----------------------------------------------------------
    # 7. 风险预算动态分配
    # -----------------------------------------------------------
    def risk_budget(self, returns: pd.Series, total_capital: float,
                    composite_score: float = 0,
                    conditional_vol: pd.Series = None) -> dict:
        """
        动态风险预算: 结合波动率、回撤、综合信号，计算建议持仓上限

        Parameters
        ----------
        returns : pd.Series  日收益率（小数）
        total_capital : float  总可投资金
        composite_score : float  综合择时得分 [-1,1], 来自TimingSystem
        conditional_vol : pd.Series, optional  GARCH条件波动率

        Returns
        -------
        dict: 建议持仓上限、风险预算、调整明细
        """
        # --- 1. 日 VaR (Cornish-Fisher 最准确) ---
        daily_var = abs(self.var_cornish_fisher(returns, 0.95))
        if daily_var < 0.001:
            daily_var = 0.01  # 安全下限

        # --- 2. 综合信号调整 ---
        signal_factor = 1.0 + composite_score * 0.5  # [-0.5, 1.5]

        # --- 3. 波动率调整 ---
        if conditional_vol is not None and len(conditional_vol) > 0:
            ann_vol = conditional_vol.iloc[-1] * np.sqrt(252)
            vol_factor = 0.15 / max(ann_vol, 0.05)  # 目标年化波动率 15%
        else:
            ann_vol = returns.tail(60).std() * np.sqrt(252)
            vol_factor = 0.15 / max(ann_vol, 0.05)
        vol_factor = np.clip(vol_factor, 0.3, 2.0)

        # --- 4. 回撤调整 ---
        dd_info = self.drawdown_analysis(returns)
        abs_dd = abs(dd_info["当前回撤"])
        if abs_dd < 0.03:
            dd_factor = 1.0
        elif abs_dd < 0.08:
            dd_factor = 0.85
        elif abs_dd < 0.15:
            dd_factor = 0.65
        else:
            dd_factor = 0.40

        # --- 5. 综合调整系数 ---
        adj = np.clip(signal_factor * vol_factor * dd_factor, 0.15, 2.0)

        # --- 6. 风险预算 = 总资金 × 1% × 调整系数 ---
        base_risk_pct = 0.01  # 基准: 每日最多亏总资金的1%
        risk_budget_rmb = total_capital * base_risk_pct * adj

        # --- 7. 建议持仓上限 = 风险预算 / 日VaR ---
        max_position = min(risk_budget_rmb / daily_var, total_capital)
        max_position = max(max_position, 0)

        return {
            "总资金": total_capital,
            "建议持仓上限": round(max_position, 0),
            "持仓上限占比": round(max_position / total_capital, 3),
            "风险预算(日)": round(risk_budget_rmb, 0),
            "日VaR(95%)": round(daily_var, 4),
            "年化波动率": round(ann_vol, 4),
            "调整明细": {
                "信号因子": round(signal_factor, 3),
                "波动率因子": round(vol_factor, 3),
                "回撤因子": round(dd_factor, 3),
                "综合调整系数": round(adj, 3),
            },
            "回撤状态": dd_info["状态"],
        }

    # -----------------------------------------------------------
    # 8. 行业集中度分析
    # -----------------------------------------------------------
    @staticmethod
    def industry_concentration(holdings: pd.DataFrame) -> pd.DataFrame:
        """
        分析ETF持仓的行业集中度
        Parameters
        ----------
        holdings : DataFrame with columns ['stock_code', 'weight', 'industry']

        Returns
        -------
        DataFrame: industry, total_weight, cumulative_weight
        """
        if "industry" not in holdings.columns:
            return pd.DataFrame({"warning": ["请提供行业信息"]})

        industry_weight = (
            holdings.groupby("industry")["weight"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        industry_weight["cumulative_weight"] = industry_weight["weight"].cumsum()
        return industry_weight

    @staticmethod
    def concentration_risk_assessment(industry_weight: pd.DataFrame,
                                       max_single_industry: float = 0.4) -> str:
        """
        评估行业集中度风险
        """
        if industry_weight.empty:
            return "无行业数据"

        top_industry = industry_weight.iloc[0]
        if top_industry["weight"] > max_single_industry:
            return (
                f"[注意] 行业集中度过高: {top_industry['industry']} "
                f"占比{top_industry['weight']:.1%}，超过上限{max_single_industry:.0%}"
            )
        return f"[正常] 行业分布合理，最大行业{top_industry['industry']}占比{top_industry['weight']:.1%}"

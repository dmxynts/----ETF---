"""
择时系统：整合多维度信号的综合择时框架
将股债利差、宏观状态、波动率信号融合为统一的仓位建议
"""
import itertools
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class VolSubWeights:
    """波动率子信号权重配置 (细分四个维度)"""
    level: float = 0.35      # 波动率水平: z-score高低
    trend: float = 0.30      # 波动率趋势: 短期vs长期均线方向
    forecast: float = 0.20   # GARCH预测: 未来波动预期方向
    event: float = 0.15      # 极端事件: 3-sigma恐慌/平静信号


@dataclass
class SignalWeights:
    """各信号权重配置"""
    spread: float = 0.40      # 股债利差权重
    macro: float = 0.25       # 宏观状态权重
    volatility: float = 0.20  # 波动率信号权重
    momentum: float = 0.15    # 动量信号权重
    vol_sub: VolSubWeights = field(default_factory=VolSubWeights)  # 波动率子信号权重


@dataclass
class UnifiedSignal:
    """统一信号"""
    date: str = ""
    composite_score: float = 0.0
    position: float = 0.5
    action: str = "持有"
    details: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class TimingSystem:
    """
    综合择时系统
    将多个维度的信号合并为单一仓位建议
    """

    def __init__(self, weights: Optional[SignalWeights] = None):
        self.weights = weights or SignalWeights()
        self.history: list = []
        self._validate_weights()

    def _validate_weights(self):
        total = sum([
            self.weights.spread,
            self.weights.macro,
            self.weights.volatility,
            self.weights.momentum,
        ])
        if not abs(total - 1.0) < 0.01:
            raise ValueError(f"权重之和必须为1 (当前: {total})")
        vs = self.weights.vol_sub
        vol_sub_total = sum([vs.level, vs.trend, vs.forecast, vs.event])
        if not abs(vol_sub_total - 1.0) < 0.01:
            raise ValueError(f"波动率子信号权重之和必须为1 (当前: {vol_sub_total})")

    def evaluate(self, spread_signal: dict, macro_signal: dict,
                  vol_signal: dict, momentum_score: float = 0.0) -> UnifiedSignal:
        """
        综合评价所有信号，生成最终仓位建议
        Parameters
        ----------
        spread_signal : 来自EquityBondSpread的信号
        macro_signal : 来自MacroStateModel的信号
        vol_signal : 来自VolatilityModel/综合信号
        momentum_score : 动量得分 [-1, 1]
        """
        # 1. 股债利差得分
        spread_percentile = spread_signal.get("percentile", 0.5)
        spread_score = spread_percentile * 2 - 1  # [-1, 1]

        # 2. 宏观状态得分
        macro_pos = macro_signal.get("suggested_position", 0.5)
        macro_score = macro_pos * 2 - 1  # [-1, 1]

        # 3. 波动率得分 (四维细分化)
        vol_level = vol_signal.get("level_score", 0)
        vol_trend = vol_signal.get("trend_score", 0)
        vol_forecast = vol_signal.get("forecast_score", 0)
        vol_event = vol_signal.get("event_score", 0)

        vw = self.weights.vol_sub
        vol_score = (vw.level * vol_level + vw.trend * vol_trend
                     + vw.forecast * vol_forecast + vw.event * vol_event)
        vol_score = np.clip(vol_score, -1, 1)

        # 保留原始z-score用于展示
        vol_zscore = vol_signal.get("vol_zscore", 0)

        # 4. 动量得分（外部传入）

        # 加权综合得分
        composite = (
            self.weights.spread * spread_score
            + self.weights.macro * macro_score
            + self.weights.volatility * vol_score
            + self.weights.momentum * momentum_score
        )

        # 映射为仓位 [0, 1]
        position = np.clip((composite + 1) / 2, 0, 1)

        # 动作判断
        if position >= 0.8:
            action = "满仓/加仓"
        elif position >= 0.6:
            action = "增持"
        elif position >= 0.4:
            action = "持有"
        elif position >= 0.2:
            action = "减仓"
        else:
            action = "清仓/轻仓"

        # 解释
        parts = []
        if spread_score > 0.3:
            parts.append(f"利差有利(得分{spread_score:.2f})")
        elif spread_score < -0.3:
            parts.append(f"利差不利(得分{spread_score:.2f})")
        if macro_score > 0.3:
            parts.append(f"宏观支持(得分{macro_score:.2f})")
        elif macro_score < -0.3:
            parts.append(f"宏观不利(得分{macro_score:.2f})")
        if vol_level > 0.3:
            parts.append(f"波动率偏高(恐慌加仓)")
        elif vol_level < -0.3:
            parts.append(f"波动率偏低(警惕拥挤)")
        if vol_trend > 0.3:
            parts.append(f"波动率下降(风险释放)")
        elif vol_trend < -0.3:
            parts.append(f"波动率上升(风险积累)")
        if vol_forecast > 0.3:
            parts.append(f"GARCH预测波动回落")
        elif vol_forecast < -0.3:
            parts.append(f"GARCH预测波动升高")
        if vol_event > 0.5:
            parts.append(f"极端事件:恐慌加仓信号!")
        elif vol_event < -0.5:
            parts.append(f"极端事件:平静减仓信号")
        if momentum_score > 0.3:
            parts.append(f"动量偏多(得分{momentum_score:.2f})")

        signal = UnifiedSignal(
            composite_score=round(composite, 3),
            position=round(position, 3),
            action=action,
            details={
                "spread_score": round(spread_score, 3),
                "macro_score": round(macro_score, 3),
                "vol_score": round(vol_score, 3),
                "momentum_score": round(momentum_score, 3),
                "vol_level": round(vol_level, 3),
                "vol_trend": round(vol_trend, 3),
                "vol_forecast": round(vol_forecast, 3),
                "vol_event": round(vol_event, 3),
            },
            explanation="; ".join(parts) if parts else "信号中性",
        )
        self.history.append(signal)
        return signal

    def get_summary(self, n_last: int = 5) -> pd.DataFrame:
        """获取最近N次信号摘要"""
        if not self.history:
            return pd.DataFrame()
        recent = self.history[-n_last:]
        records = []
        for s in recent:
            records.append({
                "综评": s.composite_score,
                "仓位": s.position,
                "动作": s.action,
                "解释": s.explanation,
            })
        return pd.DataFrame(records)

    @staticmethod
    def suggest_etf_allocation(total_position: float, target_etf: str = None,
                               etf_list: list = None) -> list:
        """
        根据总仓位，返回ETF配置建议

        Parameters
        ----------
        total_position : float  建议仓位 [0, 1]
        target_etf : str, optional  目标ETF代码，指定则只返回该ETF的权重
        etf_list : list, optional  ETF列表，用于多ETF分配时不依赖外部配置
        """
        # 指定单只ETF → 简化输出
        if target_etf:
            name = target_etf
            if etf_list:
                for etf in etf_list:
                    if etf.code == target_etf:
                        name = etf.name
                        break
            return [{"ETF": name, "代码": target_etf, "权重": round(total_position, 3)}]

        if not etf_list:
            return []

        # 多ETF组合分配（默认）
        if total_position <= 0:
            return [{"ETF": etf.name, "代码": etf.code, "权重": 0} for etf in etf_list]

        allocation = []
        for etf in etf_list:
            if "中证红利" in etf.name and "低波" not in etf.name:
                allocation.append({"ETF": etf.name, "代码": etf.code, "权重": round(total_position * 0.5, 3)})

        for etf in etf_list:
            if "低波" in etf.name:
                allocation.append({"ETF": etf.name, "代码": etf.code, "权重": round(total_position * 0.3, 3)})

        for etf in etf_list:
            if not any(a["ETF"] == etf.name for a in allocation):
                allocation.append({"ETF": etf.name, "代码": etf.code, "权重": round(total_position * 0.2, 3)})

        return allocation

    def calibrate(self, price_data: pd.DataFrame,
                  spread_df: pd.DataFrame,
                  macro_df: pd.DataFrame,
                  vol_df: pd.DataFrame,
                  momentum_df: pd.DataFrame,
                  metric: str = "sharpe",
                  fee_rate: float = 0.0003,
                  verbose: bool = True) -> dict:
        """
        网格搜索校准各信号权重，最大化历史回测指标

        遍历 (spread, macro, volatility, momentum) 的权重组合，用 BacktestEngine
        评估每组权重在历史上的表现，返回最优配置。

        Parameters
        ----------
        price_data : DataFrame   columns=['date','close'], 用于计算收益和回测
        spread_df : DataFrame    index=date, column='percentile' [0,1]
        macro_df : DataFrame     index=date, column='suggested_position' [0,1]
        vol_df : DataFrame       index=date, columns=['level_score','trend_score',
                                                      'forecast_score','event_score']
        momentum_df : DataFrame  index=date, column='score' [-1,1]
        metric : str             优化目标: 'sharpe' | 'calmar' | 'return'
        fee_rate : float         交易费率, 默认万三
        verbose : bool           是否打印进度

        Returns
        -------
        dict: best_weights, best_metric, previous_*, results DataFrame
        """
        from src.backtest.engine import BacktestEngine

        # 1. 对齐信号
        aligned = spread_df.join(macro_df, how="inner").join(vol_df, how="inner")
        aligned = aligned.join(momentum_df, how="inner")
        aligned.columns = ["percentile", "suggested_position",
                           "level_score", "trend_score", "forecast_score", "event_score",
                           "momentum"]
        aligned = aligned.dropna()
        if len(aligned) < 20:
            raise ValueError(f"对齐后样本不足 ({len(aligned)} 行), 至少需要 20 行")

        # 2. 生成权重网格
        spread_r = np.arange(0.10, 0.56, 0.05)
        macro_r = np.arange(0.10, 0.46, 0.05)
        vol_r = np.arange(0.05, 0.36, 0.05)
        momentum_r = np.arange(0.05, 0.26, 0.05)

        combos = []
        for s, m, v, mo in itertools.product(spread_r, macro_r, vol_r, momentum_r):
            if abs(s + m + v + mo - 1.0) < 0.01:
                combos.append(dict(spread=s, macro=m, volatility=v, momentum=mo))

        # 预计算各信号得分 (避免循环内重复计算)
        spread_score = aligned["percentile"] * 2 - 1
        macro_score = aligned["suggested_position"] * 2 - 1
        vw = self.weights.vol_sub
        vol_score = np.clip(
            vw.level * aligned["level_score"]
            + vw.trend * aligned["trend_score"]
            + vw.forecast * aligned["forecast_score"]
            + vw.event * aligned["event_score"],
            -1, 1,
        )
        momentum_score = aligned["momentum"]

        # 3. 遍历评估
        engine = BacktestEngine(initial_capital=1.0)
        previous_weights = self.weights
        previous_metric = None
        results = []
        n = len(combos)

        for i, w in enumerate(combos):
            composite = (w["spread"] * spread_score + w["macro"] * macro_score
                         + w["volatility"] * vol_score + w["momentum"] * momentum_score)
            position = np.clip((composite + 1) / 2, 0, 1)

            signal_df = pd.DataFrame({
                "date": aligned.index,
                "position": position.values,
            })

            result = engine.run(price_data, signal_df, name=f"cal_{i}", fee_rate=fee_rate)
            m_value = self._metric_value(result.metrics, metric)
            results.append({**w, metric: m_value})

            if previous_metric is None and previous_weights is not None:
                pw = previous_weights
                prev_composite = (pw.spread * spread_score + pw.macro * macro_score
                                  + pw.volatility * vol_score + pw.momentum * momentum_score)
                prev_position = np.clip((prev_composite + 1) / 2, 0, 1)
                prev_signal = pd.DataFrame({"date": aligned.index, "position": prev_position.values})
                prev_result = engine.run(price_data, prev_signal, name="prev", fee_rate=fee_rate)
                previous_metric = self._metric_value(prev_result.metrics, metric)

            if verbose and (i + 1) % 20 == 0:
                print(f"  校准进度: {i+1}/{n}")

        results_df = pd.DataFrame(results).sort_values(metric, ascending=False).reset_index(drop=True)
        best_row = results_df.iloc[0]

        best_weights = SignalWeights(
            spread=best_row["spread"],
            macro=best_row["macro"],
            volatility=best_row["volatility"],
            momentum=best_row["momentum"],
        )
        self.weights = best_weights

        return {
            "best_weights": best_weights,
            "best_metric": best_row[metric],
            "previous_weights": previous_weights,
            "previous_metric": previous_metric,
            "results": results_df,
            "n_trials": n,
            "metric": metric,
        }

    @staticmethod
    def _metric_value(metrics: dict, metric: str) -> float:
        """从 BacktestEngine 的 metrics dict 中提取数值指标"""
        key_map = {"sharpe": "夏普比率", "calmar": "卡尔玛比率", "return": "总收益率"}
        raw = metrics.get(key_map.get(metric, "夏普比率"), "0")
        if isinstance(raw, str) and raw.endswith("%"):
            return float(raw.strip("%")) / 100
        return float(raw)

    def run_weekly_checklist(self, spread_data: dict, macro_data: dict,
                              vol_data: dict) -> str:
        """
        生成每周操作检查清单
        """
        signal = self.evaluate(spread_data, macro_data, vol_data)

        lines = [
            "=" * 55,
            "  红利ETF 周度操作检查清单",
            "=" * 55,
            f"  综合信号: {signal.action} (仓位: {signal.position:.0%})",
            f"  综合得分: {signal.composite_score:.3f}",
            "",
            "  各维度信号:",
        ]
        for name, score in signal.details.items():
            lines.append(f"    {name:20s}: {score:+.3f}")

        lines.extend([
            "",
            f"  核心判断: {signal.explanation}",
            "",
            "  操作建议:",
        ])

        if signal.position >= 0.7:
            lines.extend([
                "  1. 当前适合重仓持有红利ETF",
                "  2. 若遇恐慌下跌（波动率骤升），可适当加仓",
                "  3. 关注国债收益率变化，利差收窄需警惕",
            ])
        elif signal.position >= 0.4:
            lines.extend([
                "  1. 正常持有，保持中性仓位",
                "  2. 关注PMI、M1-M2剪刀差变化",
                "  3. 若有显著回调，可小步加仓",
            ])
        else:
            lines.extend([
                "  1. 建议降低红利ETF仓位",
                "  2. 转向短债/货币基金等防御资产",
                "  3. 等待利差或宏观信号好转再入场",
            ])

        lines.append("=" * 55)
        self._last_checklist = "\n".join(lines)
        return self._last_checklist

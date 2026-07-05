#!/usr/bin/env python3
"""
红利ETF量化分析系统 - 主入口
股债利差择时 + 宏观状态识别 + 因子归因 + 波动率分析 + 风险管理
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from config import CFG, DIVIDEND_ETFS
from src.data.fetcher import DataFetcher
from src.analysis.equity_bond_spread import EquityBondSpread
from src.analysis.macro_state import MacroStateModel
from src.analysis.factor_attribution import FactorAttribution
from src.analysis.volatility import VolatilityModel
from src.analysis.technical import TechnicalIndicator
from src.analysis.grid_optimizer import GridOptimizer
from src.analysis.risk_management import RiskManager
from src.strategies.timing_system import TimingSystem
from src.backtest.engine import BacktestEngine
from src.utils.helpers import (
    print_header, print_metrics, plot_spread_history,
    plot_backtest_result, plot_volatility_analysis, plot_risk_heatmap,
    analyze_return_distribution,
)


class DividendETFQuantSystem:
    """红利ETF量化分析系统主类"""

    def __init__(self):
        proxy_cfg = CFG.proxy
        proxy = {"http": proxy_cfg.http, "https": proxy_cfg.https, "no_proxy": proxy_cfg.no_proxy} \
            if proxy_cfg.http or proxy_cfg.https else None

        # ETF映射表（用于数据回退）
        etf_index_map = {etf.code: etf.index_code for etf in DIVIDEND_ETFS}
        self._etf_list = DIVIDEND_ETFS

        self.fetcher = DataFetcher(proxy=proxy, etf_index_map=etf_index_map)
        self.spread_analyzer = EquityBondSpread(
            lookback_years=CFG.spread_timing.window_years,
            high_percentile=CFG.spread_timing.high_percentile,
            low_percentile=CFG.spread_timing.low_percentile,
            full_position=CFG.spread_timing.full_position,
            light_position=CFG.spread_timing.light_position,
        )
        self.macro_model = MacroStateModel(
            n_states=CFG.hmm.n_states,
            n_iter=CFG.hmm.n_iter,
        )
        self.factor_model = FactorAttribution()
        self.vol_model = VolatilityModel(
            p=CFG.garch.p,
            q=CFG.garch.q,
            forecast_days=CFG.garch.forecast_days,
            dist=CFG.garch.dist,
            model_type=CFG.garch.model_type,
        )
        self.grid_optimizer = GridOptimizer(num_grids=CFG.grid.num_grids)
        self.risk_manager = RiskManager(confidence_level=CFG.risk.confidence_level)
        self.timing_system = TimingSystem()
        self.backtest_engine = BacktestEngine()

        # 缓存数据
        self._data_cache = {}

    # -----------------------------------------------------------
    # 缓存交互
    # -----------------------------------------------------------
    def _check_cache_and_prompt(self):
        """检查缓存是否存在，统一询问是否更新"""
        info = self.fetcher.get_cache_info()
        if not info:
            return

        logger.info("\n检测到缓存数据:")
        cat_list = list(info.keys())
        for i, cat in enumerate(cat_list, 1):
            fnames = ", ".join(info[cat].keys())
            logger.info(f"  [{i}] {cat}: {fnames}")

        ans = input("\n是否更新数据? [y/N] (或输入序号如 1 3 5 单独更新): ").strip().lower()
        if not ans or ans == "n":
            return

        if ans == "y":
            self.fetcher.clear_cache()
            logger.info("  已清除全部缓存")
        else:
            # 尝试按序号解析
            try:
                indices = [int(x) for x in ans.split()]
                for idx in indices:
                    if 1 <= idx <= len(cat_list):
                        cat = cat_list[idx - 1]
                        self.fetcher.clear_cache(cat)
                        logger.info(f"  已清除: {cat}")
            except ValueError:
                logger.warning("  输入无法识别，跳过更新")

    # -----------------------------------------------------------
    # 模块1: 股债利差择时
    # -----------------------------------------------------------
    def run_spread_timing(self, index_code: str = DIVIDEND_ETFS[0].index_code,
                           years: int = 10, plot: bool = False):
        """运行股债利差择时分析"""
        print_header("股债利差择时分析")

        start_date = DataFetcher.get_start_date(years)

        logger.info(f"获取中证红利({index_code})股息率数据...")
        div_df = self.fetcher.get_index_dividend_yield(index_code, start_date)

        logger.info("获取十年期国债收益率数据...")
        bond_df = self.fetcher.get_bond_yield(start_date)

        logger.info("计算股债利差...")
        self.spread_analyzer.compute_spread(div_df, bond_df)
        self.spread_analyzer.compute_percentile()
        signal = self.spread_analyzer.generate_signal()

        # 当前判断
        latest = signal.iloc[-1]
        logger.info(f"\n最新信号 ({latest['date'].strftime('%Y-%m-%d')}):")
        logger.info(f"  利差: {latest['spread']:.2%}")
        logger.info(f"  历史分位: {latest['percentile']:.1%}")
        logger.info(f"  建议仓位: {latest['position']:.0%}")
        logger.info(f"  操作信号: {latest['signal']}")

        # 回测
        logger.info("\n运行策略回测...")
        index_df = self.fetcher.get_index_daily(index_code, start_date)
        bt_result = self.spread_analyzer.backtest(index_df)
        metrics = self.spread_analyzer.get_metrics(bt_result)
        print_metrics(metrics)

        if plot:
            self._data_cache["spread_bt"] = bt_result
            plot_spread_history(
                signal.merge(div_df, on="date", how="left")
                .merge(bond_df, on="date", how="left"),
                save_path=str(CFG.output_dir / "spread_history.png"),
            )
            plot_backtest_result(bt_result, save_path=str(CFG.output_dir / "backtest_result.png"))

        return {"signal": signal, "backtest": bt_result, "metrics": metrics}

    # -----------------------------------------------------------
    # 模块2: 宏观状态识别
    # -----------------------------------------------------------
    def run_macro_analysis(self, years: int = 10, plot: bool = False, tune: bool = False):
        """运行宏观状态分析 (HMM)"""
        print_header("宏观状态转移分析 (HMM)")

        start_year = datetime.now().year - years

        logger.info("获取M1-M2剪刀差...")
        m1m2 = self.fetcher.get_m1_m2_gap(start_year)

        logger.info("获取PMI数据...")
        pmi = self.fetcher.get_macro_data("PMI", start_year)

        logger.info("获取CPI数据...")
        cpi = self.fetcher.get_macro_data("CPI", start_year)

        logger.info("获取PPI数据...")
        ppi = self.fetcher.get_macro_data("PPI", start_year)

        logger.info("获取工业增加值数据...")
        industry = self.fetcher.get_macro_data("INDUSTRY", start_year)

        logger.info("获取国债收益率...")
        bond = self.fetcher.get_bond_yield(
            f"{start_year}0101", datetime.now().strftime("%Y%m%d")
        )

        logger.info("准备特征并训练HMM模型...")
        features, features_norm = self.macro_model.prepare_features(m1m2, pmi, cpi, ppi, industry, bond)

        # 参数调优
        if tune:
            logger.info("\n>>> HMM参数对比调优 <<<")
            logger.info("尝试不同 n_states / covariance_type / tol 组合...\n")
            result = MacroStateModel.compare_params(features_norm)
            logger.info("\n参数对比结果 (按推荐排序):")
            logger.info(result.sort_values("推荐排序").to_string(index=False))

            best = result.dropna(subset=["AIC"]).sort_values("推荐排序").iloc[0]
            logger.info(f"\n推荐参数: n_states={int(best['n_states'])}, "
                  f"cov_type={best['cov_type']}, tol={best['tol']}")
            logger.info(f"(AIC={best['AIC']}, 均匀度={best['balance_ratio']})")
            logger.info()

        self.macro_model.train(features_norm)
        labeled = self.macro_model.label_states(features)

        enhanced = self.macro_model.get_enhanced_score(features_norm)
        state_label = enhanced["state"]

        logger.info("\n最新状态:")
        logger.info(f"  当前状态: {state_label}")
        logger.info(f"  宏观评分: {enhanced['macro_score']:+.3f}")
        logger.info(f"  建议仓位: {enhanced['suggested_position']:.0%}")
        logger.info(f"  状态概率: {enhanced['state_probs']}")
        logger.info(f"  状态持续: {enhanced['persistence_months']}个月")
        logger.info(f"  下期预期: {enhanced['expected_next_pos']:.0%}仓位")

        logger.info("\n状态转移概率矩阵:")
        logger.info(self.macro_model.state_transition_matrix())

        # 各状态分布
        logger.info(f"\n各状态出现频率:")
        state_counts = labeled["state_label"].value_counts()
        for label, count in state_counts.items():
            logger.info(f"  {label}: {count}次 ({count / len(labeled):.1%})")

        return {
            "features": features,
            "labeled": labeled,
            "transition_matrix": self.macro_model.state_transition_matrix(),
            "current_state": state_label,
            "enhanced_score": enhanced,
        }

    # -----------------------------------------------------------
    # 模块3: 波动率分析 (GARCH)
    # -----------------------------------------------------------
    def run_volatility_analysis(self, etf_code: str = DIVIDEND_ETFS[0].code,
                                 years: int = 5, plot: bool = False):
        """运行波动率分析"""
        print_header("波动率建模分析 (GARCH)")

        start_date = DataFetcher.get_start_date(years)

        logger.info(f"获取ETF {etf_code} 数据...")
        etf_data = self.fetcher.get_etf_daily(etf_code, start_date)

        # 计算收益率
        returns = etf_data["close"].pct_change().dropna()
        ret_dates = etf_data["date"].iloc[-len(returns):].reset_index(drop=True)

        logger.info("拟合GARCH(1,1)模型...")
        params = self.vol_model.fit_garch(returns, dates=ret_dates)
        print_metrics(params)

        logger.info(f"\n预测未来{self.vol_model.forecast_days}日波动率（90%置信区间）...")
        vol_forecast = self.vol_model.forecast_volatility()
        for h, row in vol_forecast.iterrows():
            logger.info(f"  第{h+1}天: {row['point']:.2f}% [{row['lower']:.2f}% ~ {row['upper']:.2f}%]")

        logger.info("\n检测极端波动率事件...")
        events = self.vol_model.detect_extreme_events()
        extreme = events[events["signal"] != "正常"]
        if len(extreme) > 0:
            last = extreme.iloc[-1]
            last_date = last.get("date", extreme.index[-1])
            last_date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)
            logger.info(f"  最近一次极端事件: {last_date_str}")
            logger.info(f"  Z-Score: {last['vol_zscore']:.2f}")
            logger.info(f"  信号: {last['signal']}")
        else:
            logger.info("  当前无极端波动事件")

        # 波动率体制
        logger.info("\n波动率体制分析...")
        regime = VolatilityModel.analyze_vol_regime(returns)
        latest_regime = regime.iloc[-1]
        logger.info(f"  当前体制: {latest_regime['体制']}")
        logger.info(f"  年化波动率: {latest_regime['年化波动率']:.2%}")
        logger.info(f"  建议: {latest_regime['建议']}")

        if plot:
            plot_volatility_analysis(
                self.vol_model.vol_data, save_path=str(CFG.output_dir / "volatility_analysis.png")
            )

        return {
            "garch_params": params,
            "vol_forecast": vol_forecast,
            "extreme_events": events,
            "vol_regime": regime,
        }

    # -----------------------------------------------------------
    # 模块4: 风险管理
    # -----------------------------------------------------------
    def run_risk_analysis(self, etf_code: str = DIVIDEND_ETFS[0].code,
                           years: int = 10, holding: float = 1_000_000):
        """运行风险管理分析"""
        print_header("风险管理分析")

        start_date = DataFetcher.get_start_date(years)
        etf_data = self.fetcher.get_etf_daily(etf_code, start_date)
        returns = etf_data["close"].pct_change().dropna()
        if "date" in etf_data.columns:
            returns.index = etf_data["date"].iloc[-len(returns):]

        logger.info(f"持仓市值: {holding:,.0f} 元")
        logger.info("\n# VaR / ES 分析")
        report = self.risk_manager.full_risk_report(returns, holding)
        print_metrics(report)

        logger.info("\n# 动态回撤监控")
        dd_info = self.risk_manager.drawdown_analysis(returns)
        sl_info = self.risk_manager.dynamic_stop_loss(returns)
        logger.info(f"  当前回撤: {dd_info['当前回撤']:.2%}")
        logger.info(f"  历史最大回撤: {dd_info['历史最大回撤']:.2%}")
        logger.info(f"  回撤天数: {dd_info['回撤天数']}天")
        logger.info(f"  趋势: {dd_info['趋势']} (速度比{dd_info['速度比']:+.2f})")
        logger.info(f"  状态/预警: {dd_info['状态']} | {dd_info['预警']}")
        logger.info(f"  ─────────────────────")
        logger.info(f"  建议止损线: {sl_info['建议止损线(%)']:.1f}%")
        logger.info(f"  ({sl_info['止损依据']})")

        logger.info("\n# 收益率分布分析")
        dist = analyze_return_distribution(returns)
        print_metrics(dist)

        logger.info("\n# 极值理论 (EVT) 分析")
        evt_result = self.risk_manager.evt_gev(returns)
        if "EVT_VaR" in evt_result:
            logger.info(f"  EVT极端VaR: {evt_result['EVT_VaR']:.2%}")
            logger.info(f"  肥尾程度: {evt_result['肥尾程度']}")
            logger.info(f"  形状参数(>0=肥尾): {evt_result['形状参数']:.4f}")

        # 情景压力测试
        logger.info("\n# 情景压力测试")
        scenarios = RiskManager.stress_test(returns, holding)
        for name, detail in scenarios.items():
            loss_str = f"损失 {detail['损失金额']:,} 元"
            drop_str = f"{abs(detail['跌幅']):.1%}"
            if "起始日" in detail:
                logger.info(f"  {name}: {drop_str} ({loss_str}, 始于{detail['起始日']})")
            elif "说明" in detail:
                logger.info(f"  {name}: {drop_str} ({loss_str}, {detail['说明']})")
            else:
                logger.info(f"  {name}: {drop_str} ({loss_str})")

        # 风险预算
        logger.info("\n# 风险预算动态分配")
        budget = self.risk_manager.risk_budget(returns, holding, composite_score=0)
        pct = budget["持仓上限占比"]
        logger.info(f"  总资金: {budget['总资金']:,.0f} 元")
        logger.info(f"  建议持仓上限: {budget['建议持仓上限']:,.0f} 元 ({pct:.0%})")
        logger.info(f"  风险预算(日): {budget['风险预算(日)']:,.0f} 元")
        det = budget["调整明细"]
        logger.info(f"  调整: 信号{det['信号因子']:.2f} × 波动率{det['波动率因子']:.2f} × 回撤{det['回撤因子']:.2f}")
        logger.info(f"  综合调整系数: {det['综合调整系数']:.2f}")
        logger.info(f"  回撤状态: {budget['回撤状态']}")

        # 行业集中度
        logger.info("\n# ETF持仓行业集中度")
        try:
            holdings = self.fetcher.get_etf_holdings(etf_code)
            if not holdings.empty:
                # 假设持有全部股票，查看前10大
                logger.info(f"  前10大持仓:")
                for _, row in holdings.head(10).iterrows():
                    logger.info(f"    {row.get('stock_name', 'N/A')} "
                          f"({row.get('stock_code', 'N/A')}): "
                          f"{row.get('weight', 0):.2%}")
        except Exception as e:
            logger.warning(f"  (获取持仓失败: {e})")

        return {"risk_report": report, "distribution": dist, "evt": evt_result}

    # -----------------------------------------------------------
    # 模块5: 网格交易优化
    # -----------------------------------------------------------
    def run_grid_optimization(self, etf_code: str = DIVIDEND_ETFS[0].code, years: int = 3):
        """运行网格交易优化"""
        print_header("网格交易参数优化")

        start_date = DataFetcher.get_start_date(years)
        etf_data = self.fetcher.get_etf_daily(etf_code, start_date)

        logger.info("校准网格参数（卡尔曼滤波 + ATR）...")
        params = self.grid_optimizer.calibrate_grid_levels(etf_data)

        logger.info(f"  网格中心: {params['网格中心']:.3f}")
        logger.info(f"  网格间距: {params['网格间距']:.4f}")
        logger.info(f"  ATR(20): {params['ATR']:.4f}")
        logger.info(f"  网格层数: {params['网格层数']}")
        logger.info(f"  滑点假设: {self.grid_optimizer.slippage:.4%}")
        logger.info(f"  交易费率: {self.grid_optimizer.fee_rate:.4%}")

        logger.info("\n网格价格层级:")
        levels = params["网格价格"]
        for i, p in enumerate(levels):
            direction = "↓买入" if p <= params["网格中心"] else "↑卖出"
            logger.info(f"  [{i+1:2d}] {direction}  {p:.3f}")

        # 模拟回测
        logger.info("\n模拟网格交易...")
        nav_df = self.grid_optimizer.simulate_grid(
            etf_data["close"], levels, capital=100000, grid_ratio=0.1
        )
        if not nav_df.empty:
            total_return = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
            logger.info(f"  模拟总收益: {total_return:.2%}")
            logger.info(f"  最终净值: ¥{nav_df['nav'].iloc[-1]:,.2f}")
            total_slippage = nav_df["cumulative_slippage_cost"].iloc[-1]
            total_fee = nav_df["cumulative_fee_cost"].iloc[-1]
            logger.info(f"  累计滑点损失: ¥{total_slippage:,.2f}")
            logger.info(f"  累计交易费用: ¥{total_fee:,.2f}")

        return {"grid_params": params, "simulation": nav_df}

    # -----------------------------------------------------------
    # 模块6: 因子归因
    # -----------------------------------------------------------
    def run_factor_attribution(self, etf_code: str = DIVIDEND_ETFS[0].code, years: int = 5):
        """运行因子归因分析（时间序列法：ETF收益对因子收益回归）"""
        print_header("因子归因分析 (时间序列多因子)")

        start_date = DataFetcher.get_start_date(years)

        # 获取ETF数据，收益率带日期索引
        etf_data = self.fetcher.get_etf_daily(etf_code, start_date)
        returns = etf_data.set_index("date")["close"].pct_change().dropna()
        returns.name = "ETF_return"

        # 构造代理因子
        logger.info("获取因子数据...")
        factor_data_ok = False
        factor_series = {}

        try:
            # 沪深300（市场因子）
            hs300 = self.fetcher.get_index_daily("000300", start_date)
            factor_series["market"] = hs300.set_index("date")["close"].pct_change().dropna()

            # 国证2000 → 小盘因子
            try:
                gz2000 = self.fetcher.get_index_daily("399303", start_date)
                factor_series["size"] = gz2000.set_index("date")["close"].pct_change().dropna()
            except Exception as e:
                logger.debug("小盘因子获取失败（可选）: %s", e)

            # 红利低波 / 中证红利 → 低波因子（低波指数收益 - 红利指数收益）
            try:
                hlw = self.fetcher.get_index_daily("930846", start_date)
                hlw_ret = hlw.set_index("date")["close"].pct_change().dropna()
            except Exception:
                hlw_ret = None

            if hlw_ret is not None:
                try:
                    hs_div = self.fetcher.get_index_daily("000922", start_date)
                    div_ret = hs_div.set_index("date")["close"].pct_change().dropna()
                    # 低波因子 = 红利低波超额（相对中证红利）
                    aligned = pd.concat([hlw_ret, div_ret], axis=1, join="inner")
                    factor_series["low_vol"] = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                except Exception:
                    factor_series["low_vol"] = hlw_ret
            else:
                # 用中证红利波动率代理
                try:
                    hs_div = self.fetcher.get_index_daily("000922", start_date)
                    div_ret = hs_div.set_index("date")["close"].pct_change().dropna()
                    vol_20 = div_ret.rolling(20).std()
                    factor_series["low_vol"] = -vol_20  # 高波动→负信号
                except Exception as e:
                    logger.debug("低波波动率代理计算失败: %s", e)

            factor_data_ok = len(factor_series) >= 2

        except Exception as e:
            logger.warning("  因子数据获取受限: %s", e)

        if not factor_data_ok:
            logger.warning("  因子数据不足（至少需要市场因子和一个小盘/低波因子）")
            logger.warning("  提示: 请检查网络连接或数据源可用性")
            return {}

        logger.info(f"  已构造 {len(factor_series)} 个因子: {', '.join(factor_series.keys())}")

        # 对齐所有序列到相同日期
        all_series = {"ETF": returns, **factor_series}
        aligned = pd.concat(all_series, axis=1).dropna()
        etf_aligned = aligned["ETF"]
        factor_dict = {name: aligned[name] for name in factor_series.keys()}

        if len(aligned) < 60:
            logger.warning(f"  对齐后数据不足（{len(aligned)}天），至少需要60天")
            return {}

        # 准备因子数据并运行回归
        etf_returns_df = etf_aligned.to_frame("ETF")
        self.factor_model.prepare_factor_data(etf_returns_df, factor_dict)
        reg_result = self.factor_model.run_regression()
        factor_contrib = self.factor_model.factor_contribution()

        r2 = reg_result.get("R²", 0)
        logger.info(self.factor_model.summary_text(factor_contrib, r_squared=r2))

        # 收益分解
        logger.info("\n收益分解（最近5日平均因子贡献）:")
        try:
            factor_returns = pd.DataFrame(factor_dict)
            decomp = self.factor_model.decompose_etf_return(etf_aligned, factor_returns)
            recent = decomp.tail(5)
            for date, row in recent.iterrows():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                parts = [f"  {date_str}  实际={row['实际收益']:.4%}"]
                for col in factor_series.keys():
                    key = f"{col}_贡献"
                    if key in row:
                        parts.append(f"{col}={row[key]:+.4%}")
                parts.append(f"Alpha={row['Alpha']:+.4%}")
                logger.info("  ".join(parts))
        except Exception as e:
            logger.warning(f"  收益分解失败: {e}")

        # 滚动Beta
        logger.info("\n滚动因子Beta (120日窗口):")
        try:
            factor_returns = pd.DataFrame(factor_dict)
            rolling_beta = self.factor_model.rolling_factor_beta(
                etf_aligned, factor_returns, window=120
            )
            if not rolling_beta.empty:
                latest_beta = rolling_beta.iloc[-1]
                for name in factor_series.keys():
                    beta_key = f"{name}_beta"
                    beta_val = latest_beta.get(beta_key, "N/A")
                    logger.info(f"  {name} Beta: {beta_val:.3f}" if isinstance(beta_val, float) else f"  {name} Beta: {beta_val}")
        except Exception as e:
            logger.warning(f"  滚动Beta计算失败: {e}")

        return {
            "factor_contribution": factor_contrib,
            "regression": reg_result,
        }

    # -----------------------------------------------------------
    # 模块7: 综合择时系统
    # -----------------------------------------------------------
    def run_comprehensive_timing(self, index_code: str = DIVIDEND_ETFS[0].index_code,
                                  etf_code: str = DIVIDEND_ETFS[0].code):
        """运行综合择时系统"""
        print_header("综合择时系统")

        start_date = DataFetcher.get_start_date(10)
        today_str = DataFetcher.get_today_str()

        logger.info("获取各维度数据...")

        # 1. 股债利差
        div_df = self.fetcher.get_index_dividend_yield(index_code, start_date)
        bond_df = self.fetcher.get_bond_yield(start_date)
        self.spread_analyzer.compute_spread(div_df, bond_df)
        self.spread_analyzer.compute_percentile()
        signal = self.spread_analyzer.generate_signal()

        latest = signal.iloc[-1]
        spread_data = {
            "percentile": latest["percentile"],
            "spread": latest["spread"],
        }

        # 2. 宏观状态 (增强评分)
        try:
            start_year = datetime.now().year - 10
            m1m2 = self.fetcher.get_m1_m2_gap(start_year)
            pmi = self.fetcher.get_macro_data("PMI", start_year)
            cpi = self.fetcher.get_macro_data("CPI", start_year)
            ppi = self.fetcher.get_macro_data("PPI", start_year)
            industry = self.fetcher.get_macro_data("INDUSTRY", start_year)
            features, features_norm = self.macro_model.prepare_features(m1m2, pmi, cpi, ppi, industry, bond_df)
            self.macro_model.train(features_norm)
            self.macro_model.label_states(features)

            # 使用增强评分代替简单 suggest_position
            enhanced = self.macro_model.get_enhanced_score(features_norm)
            macro_data = {
                "state": enhanced["state"],
                "suggested_position": enhanced["suggested_position"],
            }
            macro_enhanced = enhanced  # 保存供展示
        except Exception as e:
            logger.warning(f"  宏观数据获取受限: {e}")
            macro_data = {"state": "未知", "suggested_position": 0.5}
            macro_enhanced = None

        # 3. 波动率 (四维细分化信号)
        try:
            etf_data = self.fetcher.get_etf_daily(etf_code, start_date)
            returns = etf_data["close"].pct_change().dropna()
            vol_data = self.vol_model.get_composite_signal(returns)
        except Exception as e:
            logger.warning(f"  波动率数据获取受限: {e}")
            vol_data = {"level_score": 0, "trend_score": 0,
                        "forecast_score": 0, "event_score": 0, "vol_zscore": 0}

        # 3.5 技术分析动量 (从 etf_data 取收盘价)
        try:
            tech_signal = TechnicalIndicator.get_composite_momentum(etf_data["close"])
            momentum_score = tech_signal["momentum_score"]
        except Exception as e:
            logger.warning(f"  技术信号计算受限: {e}")
            tech_signal = {"momentum_score": 0, "rsi": 0, "macd": 0, "ma_cross": 0, "explanation": "无"}
            momentum_score = 0

        # 4. 综合打分
        signal_result = self.timing_system.evaluate(spread_data, macro_data, vol_data, momentum_score)

        logger.info(f"\n综合信号分析结果:")
        logger.info(f"  {'综评':20s}: {signal_result.composite_score:+.3f}")
        logger.info(f"  {'建议仓位':20s}: {signal_result.position:.0%}")
        logger.info(f"  {'操作':20s}: {signal_result.action}")
        logger.info(f"  {'解释':20s}: {signal_result.explanation}")

        logger.info("\n各维度得分:")
        for name, score in signal_result.details.items():
            logger.info(f"  {name:20s}: {score:+.3f}")

        # 波动率子信号明细
        if "level_score" in signal_result.details:
            logger.info("\n波动率子信号:")
            sub_names = {"level_score": "水平(恐慌/平静)", "trend_score": "趋势(上升/下降)",
                         "forecast_score": "预测(GARCH方向)", "event_score": "事件(极端信号)"}
            for key, label in sub_names.items():
                val = signal_result.details[key]
                logger.info(f"  {label:20s}: {val:+.3f}")
            vol_z = vol_data.get("vol_zscore", 0)
            logger.info(f"  (原始Z-Score: {vol_z:+.2f})")

        # 动量子信号明细
        if tech_signal["momentum_score"] != 0 or tech_signal["rsi"] != 0:
            logger.info("\n动量子信号 (RSI+MACD+均线):")
            logger.info(f"  RSI:       {tech_signal['rsi']:+.3f}")
            logger.info(f"  MACD:      {tech_signal['macd']:+.3f}")
            logger.info(f"  均线交叉:   {tech_signal['ma_cross']:+.3f}")
            logger.info(f"  动量综评:   {tech_signal['momentum_score']:+.3f}")
            logger.info(f"  ({tech_signal['explanation']})")

        # 宏观增强评分明细
        if macro_enhanced:
            logger.info("\n宏观评分明细:")
            det = macro_enhanced["details"]
            logger.info(f"  状态概率分布: {det['state_probs']}")
            logger.info(f"  概率加权仓位: {det['加权仓位']:.3f}")
            logger.info(f"  下期预期仓位: {det['下期预期仓位']:.3f}")
            logger.info(f"  当前状态持续: {det['持续月数']}个月")
            logger.info(f"  特征极端程度: {det['极端程度']:.2f}")
            logger.info(f"  最终建议仓位: {macro_enhanced['suggested_position']:.1%}")
            logger.info(f"  宏观得分: {macro_enhanced['macro_score']:+.3f}")

        # ETF配置建议
        logger.info("\nETF配置建议:")
        allocation = TimingSystem.suggest_etf_allocation(
            signal_result.position, target_etf=etf_code, etf_list=self._etf_list
        )
        for a in allocation:
            if a["权重"] > 0:
                logger.info(f"  {a['ETF']}({a['代码']}): {a['权重']:.0%}")

        # 风险预算 (综合考虑择时信号)
        try:
            budget = self.risk_manager.risk_budget(
                returns, holding=1_000_000,
                composite_score=signal_result.composite_score,
                conditional_vol=self.vol_model.vol_data["conditional_vol"]
                if self.vol_model.vol_data is not None else None,
            )
            pct = budget["持仓上限占比"]
            logger.info(f"\n风险预算:")
            logger.info(f"  建议持仓上限: {budget['建议持仓上限']:,.0f} 元 ({pct:.0%})")
            det = budget["调整明细"]
            logger.info(f"  调整: 信号{det['信号因子']:.2f} × 波动率{det['波动率因子']:.2f} × 回撤{det['回撤因子']:.2f}")
            logger.info(f"  综合调整系数: {det['综合调整系数']:.2f}")
        except Exception as e:
            logger.debug("风险预算计算跳过: %s", e)

        return {
            "signal": signal_result,
            "allocation": allocation,
        }

    # -----------------------------------------------------------
    # 一键全运行
    # -----------------------------------------------------------
    def run_all(self, index_code: str = DIVIDEND_ETFS[0].index_code, etf_code: str = DIVIDEND_ETFS[0].code,
                 plot: bool = False):
        """运行全部分析模块"""
        logger.info("=" * 55)
        logger.info("  红利ETF量化分析系统 - 全面分析报告")
        logger.info(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 55)

        # 询问是否更新数据
        self._check_cache_and_prompt()

        results = {}

        logger.info("\n—— 模块1: 股债利差择时 ——")
        results["spread"] = self.run_spread_timing(index_code, plot=False)

        logger.info("\n—— 模块2: 宏观状态识别 ——")
        try:
            results["macro"] = self.run_macro_analysis(plot=False)
        except Exception as e:
            logger.warning(f"  跳过(数据获取异常): {e}")
            results["macro"] = None

        logger.info("\n—— 模块3: 波动率分析 ——")
        try:
            results["volatility"] = self.run_volatility_analysis(etf_code, plot=False)
        except Exception as e:
            logger.warning(f"  跳过(数据获取异常): {e}")
            results["volatility"] = None

        logger.info("\n—— 模块4: 风险管理 ——")
        try:
            results["risk"] = self.run_risk_analysis(etf_code)
        except Exception as e:
            logger.warning(f"  跳过(数据获取异常): {e}")
            results["risk"] = None

        logger.info("\n—— 模块5: 综合择时 ——")
        try:
            results["timing"] = self.run_comprehensive_timing(index_code)
        except Exception as e:
            logger.warning(f"  跳过(数据获取异常): {e}")
            results["timing"] = None

        logger.info("\n" + "=" * 55)
        logger.info("  分析完成！")
        logger.info("=" * 55)
        return results


# -----------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------
def setup_parser() -> argparse.ArgumentParser:
    """配置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="红利ETF量化分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py all          # 运行全部分析
  python main.py spread       # 仅股债利差分析
  python main.py macro        # 仅宏观状态分析
  python main.py volatility   # 仅波动率分析
  python main.py risk         # 仅风险管理
  python main.py grid         # 仅网格优化
  python main.py timing       # 综合择时
        """,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "spread", "macro", "volatility", "risk", "grid", "timing", "factor"],
        help="分析模块",
    )
    parser.add_argument(
        "--etf", type=str, default=DIVIDEND_ETFS[0].code,
        help=f"ETF代码 (默认: {DIVIDEND_ETFS[0].code} {DIVIDEND_ETFS[0].name})",
    )
    parser.add_argument(
        "--index", type=str, default=DIVIDEND_ETFS[0].index_code,
        help=f"指数代码 (默认: {DIVIDEND_ETFS[0].index_code} {DIVIDEND_ETFS[0].index_name})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="输出图表到 output/ 目录",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="对HMM模型进行参数对比调优 (需搭配 macro 命令)",
    )
    return parser


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = setup_parser()
    args = parser.parse_args()
    system = DividendETFQuantSystem()

    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    cmd_map = {
        "all": lambda: system.run_all(args.index, args.etf, args.plot),
        "spread": lambda: system.run_spread_timing(args.index, plot=args.plot),
        "macro": lambda: system.run_macro_analysis(plot=args.plot, tune=args.tune),
        "volatility": lambda: system.run_volatility_analysis(args.etf, plot=args.plot),
        "risk": lambda: system.run_risk_analysis(args.etf),
        "grid": lambda: system.run_grid_optimization(args.etf),
        "timing": lambda: system.run_comprehensive_timing(args.index),
        "factor": lambda: system.run_factor_attribution(args.etf),
    }

    func = cmd_map.get(args.command)
    if func:
        func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

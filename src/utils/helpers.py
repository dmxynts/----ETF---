"""
工具函数: 绘图、统计、格式化
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # 非交互后端，兼容无GUI环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, List
from pathlib import Path

# 中文字体设置（显式注册字体文件路径）
import matplotlib.font_manager as fm
_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
_FONT_PROP = fm.FontProperties(fname=_FONT_PATH) if Path(_FONT_PATH).exists() else None

sns.set_style("whitegrid")

if _FONT_PROP:
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = _FONT_PROP.get_name()
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["figure.dpi"] = 120


def plot_spread_history(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制股债利差历史走势图
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    # 兼容列名差异：dividend_yield 或 div_yield
    dy_col = "dividend_yield" if "dividend_yield" in df.columns else "div_yield"
    by_col = "yield" if "yield" in df.columns else "bond_yield"
    # 统一转为百分比显示
    div_y = df[dy_col] * 100 if df[dy_col].max() < 1 else df[dy_col]
    bond_y = df[by_col] * 100 if df[by_col].max() < 1 else df[by_col]
    ax1.plot(df["date"], div_y, label="股息率", color="red", alpha=0.8)
    ax1.plot(df["date"], bond_y, label="十年期国债收益率", color="blue", alpha=0.8)
    ax1.set_ylabel("收益率 (%)")
    ax1.legend(loc="upper left")
    ax1.set_title("中证红利股息率 vs 十年期国债收益率")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(df["date"], df["spread"], 0,
                      where=(df["spread"] > 0), color="red", alpha=0.3, label="利差>0")
    ax2.fill_between(df["date"], df["spread"], 0,
                      where=(df["spread"] < 0), color="green", alpha=0.3, label="利差<0")
    ax2.plot(df["date"], df["spread"], color="black", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("利差 (%)")
    ax2.set_title("股债利差 (股息率 - 国债收益率)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df["date"], df["position"], label="策略仓位", color="green", linewidth=2)
    ax3.fill_between(df["date"], 0, df["position"],
                      where=(df["position"] > 0.7), color="red", alpha=0.2)
    ax3.fill_between(df["date"], 0, df["position"],
                      where=(df["position"] < 0.5), color="green", alpha=0.2)
    ax3.set_ylabel("仓位比例")
    ax3.set_xlabel("日期")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("策略仓位信号")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_backtest_result(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制回测结果
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(df["date"], df["strategy_nav"], label="策略净值", color="red", linewidth=2)
    ax1.plot(df["date"], df["index_nav"], label="基准净值", color="blue", linewidth=1.5, alpha=0.7)
    ax1.set_ylabel("净值")
    ax1.set_title("策略 vs 基准净值")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    # 资金曲线
    if "drawdown" in df.columns:
        ax2.fill_between(df["date"], 0, df["drawdown"],
                          color="red", alpha=0.3, label="回撤")
        ax2.plot(df["date"], df["drawdown"], color="red", linewidth=1)
        ax2.set_ylabel("回撤")
        ax2.set_title("策略回撤曲线")
    elif "excess_return" in df.columns:
        ax2.plot(df["date"], df["excess_return"], color="green", linewidth=1.5)
        ax2.fill_between(df["date"], 0, df["excess_return"],
                          where=(df["excess_return"] > 0), color="green", alpha=0.2)
        ax2.fill_between(df["date"], df["excess_return"], 0,
                          where=(df["excess_return"] < 0), color="red", alpha=0.2)
        ax2.set_ylabel("超额收益")
        ax2.set_title("策略累计超额收益")

    ax2.set_xlabel("日期")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_volatility_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """绘制波动率分析图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(df.index, df["returns"], alpha=0.5, label="日收益率", color="gray")
    ax1.plot(df.index, df["conditional_vol"], label="条件波动率(GARCH)", color="red")
    ax1.set_title("GARCH(1,1) 条件波动率")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if "vol_zscore" in df.columns:
        ax2 = axes[1]
        ax2.plot(df.index, df["vol_zscore"], color="blue", alpha=0.7)
        ax2.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="+2σ(加仓)")
        ax2.axhline(y=-2, color="green", linestyle="--", alpha=0.5, label="-2σ(减仓)")
        ax2.fill_between(df.index, 2, df["vol_zscore"],
                          where=(df["vol_zscore"] > 2), color="red", alpha=0.2)
        ax2.set_title("波动率 Z-Score（红色区域 = 恐慌加仓机会）")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_risk_heatmap(returns: pd.Series, save_path: Optional[str] = None):
    """绘制风险热力图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 滚动VaR
    window = 252
    rolling_var = returns.rolling(window).apply(
        lambda x: np.percentile(x, 5), raw=True
    )
    rolling_cvar = returns.rolling(window).apply(
        lambda x: x[x <= np.percentile(x, 5)].mean(), raw=True
    )

    ax.plot(returns.index, returns.cumsum(), label="累计收益", alpha=0.6)
    ax.plot(rolling_var.index, rolling_var, label="滚动VaR(95%)", color="red")
    ax.plot(rolling_cvar.index, rolling_cvar, label="滚动CVaR(95%)", color="darkred", linestyle="--")

    ax.set_title("滚动风险指标 (252日窗口)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def print_header(title: str, width: int = 60):
    """打印格式化标题"""
    logger.info("")
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


def print_metrics(metrics: dict):
    """格式化打印评价指标"""
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v}")


def save_dataframe_to_excel(df: pd.DataFrame, path: str, sheet_name: str = "Sheet1"):
    """保存 DataFrame 到 Excel"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info(f"数据已保存到: {path}")


def analyze_return_distribution(returns: pd.Series) -> dict:
    """收益率分布统计分析"""
    return {
        "均值": f"{returns.mean():.4%}",
        "标准差": f"{returns.std():.4%}",
        "偏度": f"{stats.skew(returns):.3f}",
        "峰度": f"{stats.kurtosis(returns):.3f}",
        "最小值": f"{returns.min():.4%}",
        "最大值": f"{returns.max():.4%}",
        "正收益占比": f"{(returns > 0).mean():.2%}",
        "JB正态检验p值": f"{stats.jarque_bera(returns)[1]:.4f}",
        "收益特征": "右偏(走楼梯式上涨)" if stats.skew(returns) > 0 else "左偏(注意尾部风险)",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("工具模块加载成功")

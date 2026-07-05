"""
红利ETF量化分析系统 - 全局配置
支持 config.yaml 覆盖默认参数
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 项目根目录
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# YAML 配置加载
# ============================================================
def _load_yaml_config() -> dict:
    """加载 config.yaml，不存在则返回空字典"""
    yaml_path = _PROJECT_ROOT / "config.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.warning("config.yaml 加载失败，使用默认配置: %s", e)
        return {}


# ============================================================
# ETF 标的配置
# ============================================================
@dataclass
class ETFConfig:
    """单只ETF配置"""
    code: str          # ETF代码
    name: str          # ETF名称
    index_code: str    # 跟踪指数代码
    index_name: str    # 跟踪指数名称


# 主流红利ETF列表（可通过 config.yaml 覆盖）
_DEFAULT_ETFS: List[ETFConfig] = [
    ETFConfig("510880", "华泰柏瑞红利ETF", "000922", "中证红利指数"),
    ETFConfig("159581", "万家中证红利ETF", "000922", "中证红利指数"),
    ETFConfig("515180", "易方达中证红利ETF", "000922", "中证红利指数"),
    ETFConfig("516828", "华泰柏瑞红利低波ETF", "930846", "中证红利低波动指数"),
    ETFConfig("563020", "易方达红利低波ETF", "930846", "中证红利低波动指数"),
    ETFConfig("517100", "富国中证红利ETF", "000922", "中证红利指数"),
]

# 运行时可被 YAML 覆盖
DIVIDEND_ETFS: List[ETFConfig] = list(_DEFAULT_ETFS)


# ============================================================
# 无风险利率配置
# ============================================================
@dataclass
class RiskFreeConfig:
    """无风险利率相关配置"""
    bond_code: str = "CN10Y"         # 十年期国债收益率代码
    bond_name: str = "中国10年期国债收益率"
    lookback_years: int = 10         # 回溯年限


# ============================================================
# 策略参数
# ============================================================
@dataclass
class SpreadTimingConfig:
    """股债利差择时策略参数"""
    high_percentile: float = 0.80    # 高估阈值分位（利差高于此 → 满仓）
    low_percentile: float = 0.20     # 低估阈值分位（利差低于此 → 轻仓）
    full_position: float = 1.0       # 满仓比例
    light_position: float = 0.3      # 轻仓比例
    window_years: int = 10           # 滚动窗口年数


@dataclass
class ProxyConfig:
    """代理配置（留空 = 不启用代理）"""
    http: str = ""                               # HTTP代理, e.g. "http://127.0.0.1:7890"
    https: str = ""                              # HTTPS代理, e.g. "http://127.0.0.1:7890"
    no_proxy: str = "localhost,127.0.0.1"        # 不走代理的地址, e.g. "localhost,127.0.0.1"


@dataclass
class HMMConfig:
    """HMM宏观状态模型参数"""
    n_states: int = 4                # 状态数（复苏/过热/滞胀/衰退）
    n_iter: int = 5000              # 迭代次数
    lookback_years: int = 10         # 回溯年限


@dataclass
class GARCHConfig:
    """GARCH波动率模型参数"""
    p: int = 1
    q: int = 1
    forecast_days: int = 5           # 预测未来天数
    vol_threshold: float = 2.0       # 加仓信号：波动率超过均值+threshold*标准差
    dist: str = "studentst"          # 分布假设: normal / studentst / skewedstudent
    model_type: str = "EGARCH"        # 模型: Garch / EGARCH / GJR-GARCH


@dataclass
class GridConfig:
    """网格交易参数"""
    num_grids: int = 10              # 网格数量
    max_sharpe: bool = True          # 是否最大化夏普比率


@dataclass
class RiskConfig:
    """风险管理参数"""
    confidence_level: float = 0.95   # VaR/ES置信度
    max_position_ratio: float = 0.4  # 单行业最大仓位
    etf_holding: float = 1_000_000  # 持仓金额（默认100万）


# ============================================================
# 全局配置单例
# ============================================================
class Config:
    """全局配置（支持 YAML 覆盖）"""
    def __init__(self):
        self.spread_timing = SpreadTimingConfig()
        self.hmm = HMMConfig()
        self.garch = GARCHConfig()
        self.grid = GridConfig()
        self.risk = RiskConfig()
        self.risk_free = RiskFreeConfig()
        self.proxy = ProxyConfig()

        # 路径配置（相对于项目根目录）
        self.cache_dir: Path = _PROJECT_ROOT / "cache"
        self.output_dir: Path = _PROJECT_ROOT / "output"

        # 从 YAML 覆盖默认值
        self._apply_yaml(_load_yaml_config())

    def _apply_yaml(self, yaml_config: dict):
        """将 YAML 配置合并到 dataclass 实例"""
        if not yaml_config:
            return

        # 映射：YAML 键 → dataclass 实例
        sections = {
            "spread_timing": self.spread_timing,
            "hmm": self.hmm,
            "garch": self.garch,
            "grid": self.grid,
            "risk": self.risk,
            "risk_free": self.risk_free,
            "proxy": self.proxy,
        }
        for key, obj in sections.items():
            section = yaml_config.get(key)
            if isinstance(section, dict):
                for field_name, value in section.items():
                    if hasattr(obj, field_name):
                        # 类型转换：YAML 可能把整数读成浮点数
                        expected_type = type(getattr(obj, field_name))
                        try:
                            if expected_type == int and not isinstance(value, int):
                                value = int(value)
                            elif expected_type == float and not isinstance(value, float):
                                value = float(value)
                            elif expected_type == bool and not isinstance(value, bool):
                                value = bool(value)
                            setattr(obj, field_name, value)
                        except (TypeError, ValueError):
                            logger.warning("配置项 %s.%s 类型错误，跳过", key, field_name)

        # ETF 列表覆盖
        etfs_yaml = yaml_config.get("etfs")
        if isinstance(etfs_yaml, list):
            global DIVIDEND_ETFS
            new_etfs = []
            for item in etfs_yaml:
                if isinstance(item, dict) and all(k in item for k in ("code", "name", "index_code", "index_name")):
                    new_etfs.append(ETFConfig(**item))
            if new_etfs:
                DIVIDEND_ETFS.clear()
                DIVIDEND_ETFS.extend(new_etfs)
                logger.info("已从 config.yaml 加载 %d 只 ETF 配置", len(new_etfs))

        # 路径配置
        paths = yaml_config.get("paths", {})
        if isinstance(paths, dict):
            cache_str = paths.get("cache_dir", "")
            if cache_str:
                self.cache_dir = _PROJECT_ROOT / cache_str
            output_str = paths.get("output_dir", "")
            if output_str:
                self.output_dir = _PROJECT_ROOT / output_str

    @property
    def etf_codes(self) -> List[str]:
        return [etf.code for etf in DIVIDEND_ETFS]

    @property
    def index_codes(self) -> List[str]:
        return list(set(etf.index_code for etf in DIVIDEND_ETFS))


# 全局单例
CFG = Config()

"""
数据获取模块
使用 AKShare 获取红利ETF及相关宏观数据
支持多数据源容错 + 文件缓存
"""
import os
import time
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import warnings

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _get_cache_dir() -> Path:
    """获取缓存目录（优先使用配置，否则用默认路径）"""
    try:
        from config import CFG
        return CFG.cache_dir.resolve()
    except (ImportError, Exception):
        return Path(__file__).resolve().parent.parent.parent / "cache"


class DataFetcher:
    """
    数据获取器：封装AKShare接口，提供统一的ETF/指数/宏观数据获取
    特性: 多数据源容错、文件缓存、网络中断时自动使用缓存
    """
    logger = logging.getLogger(__name__)

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0,
                 use_cache: bool = True, cache_expire_days: int = 1,
                 proxy: Optional[dict] = None,
                 etf_index_map: Optional[dict] = None):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        self.cache_expire_days = cache_expire_days
        self.proxy = proxy
        # ETF代码→指数代码映射（用于数据获取回退）
        self.etf_index_map = etf_index_map or {}
        # 反向映射：指数代码→第一个找到的ETF代码
        self.index_to_etf = {}
        for etf_code, idx_code in self.etf_index_map.items():
            if idx_code not in self.index_to_etf:
                self.index_to_etf[idx_code] = etf_code

        # 配置网络代理
        if proxy:
            if proxy.get("http"):
                os.environ["HTTP_PROXY"] = proxy["http"]
            if proxy.get("https"):
                os.environ["HTTPS_PROXY"] = proxy["https"]
            if proxy.get("no_proxy"):
                os.environ["NO_PROXY"] = proxy["no_proxy"]
            else:
                os.environ.pop("NO_PROXY", None)
                os.environ.pop("no_proxy", None)
        else:
            # 绕过系统代理（国内金融API走代理容易出问题）
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"

        if use_cache:
            _get_cache_dir().mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # 缓存管理
    # -----------------------------------------------------------
    def _cache_path(self, name: str) -> Path:
        return _get_cache_dir() / f"{name}.pkl"

    def _save_cache(self, name: str, data: pd.DataFrame):
        if not self.use_cache:
            return
        try:
            path = self._cache_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"data": data, "time": datetime.now()}, f)
        except Exception as e:
            self.logger.debug("缓存写入失败 (%s): %s", name, e)

    def _load_cache(self, name: str) -> Optional[pd.DataFrame]:
        if not self.use_cache:
            return None
        try:
            path = self._cache_path(name)
            if not path.exists():
                return None
            # 检查过期
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if (datetime.now() - mtime).days > self.cache_expire_days:
                return None
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj["data"]
        except Exception as e:
            self.logger.debug("缓存读取失败 (%s): %s", name, e)
            return None

    # -----------------------------------------------------------
    # 缓存交互管理
    # -----------------------------------------------------------
    CACHE_CATEGORIES = {
        "指数行情": "idx_",
        "ETF行情": "etf_",
        "股息率": "div_",
        "国债收益率": "bond_",
        "宏观数据": "macro_",
    }

    def get_cache_info(self) -> dict:
        """返回缓存文件信息: {分类: {文件名: 最后修改时间}}"""
        cache_dir = _get_cache_dir()
        if not cache_dir.exists():
            return {}
        info = {}
        for cat_name, prefix in self.CACHE_CATEGORIES.items():
            files = list(cache_dir.glob(f"{prefix}*.pkl"))
            if files:
                info[cat_name] = {}
                for f in sorted(files):
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    info[cat_name][f.name] = mtime.strftime("%Y-%m-%d %H:%M")
        # 不在分类中的也列出来
        other = [f for f in cache_dir.glob("*.pkl") if not any(f.name.startswith(p) for p in self.CACHE_CATEGORIES.values())]
        if other:
            info["其他"] = {f.name: datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M") for f in sorted(other)}
        return info

    def clear_cache(self, category: str = None):
        """
        清除缓存
        Parameters
        ----------
        category : str, optional
            None=全部清除, 或 CACHE_CATEGORIES 的 key, 或自定义文件名前缀
        """
        cache_dir = _get_cache_dir()
        if not cache_dir.exists():
            return

        if category is None:
            # 全部清除
            for f in cache_dir.glob("*.pkl"):
                f.unlink()
            return

        prefix = self.CACHE_CATEGORIES.get(category, category)
        for f in cache_dir.glob(f"{prefix}*.pkl"):
            f.unlink()

    @staticmethod
    def _ask_refresh() -> bool:
        """询问用户是否更新数据"""
        ans = input("  是否更新数据? [y/N]: ").strip().lower()
        return ans == "y"

    def _safe_request(self, func, *args, **kwargs):
        """带重试的安全请求"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise ConnectionError(f"数据获取失败({func.__name__}): {e}")
        return None

    # -----------------------------------------------------------
    # 1. ETF / 指数行情数据
    # -----------------------------------------------------------
    def _fetch_index_price(self, index_code: str, start_date: str,
                                          end_date: str) -> Optional[pd.DataFrame]:
        """
        获取指数日线行情（缓存优先 + 多数据源容错）
        数据源顺序: 缓存 → 东方财富 → 中证指数官网 → 腾讯
        """
        import akshare as ak

        cache_name = f"idx_{index_code}"
        # --------------------------------------------------
        # 1. 缓存命中 → 直接返回
        # --------------------------------------------------
        cached = self._load_cache(cache_name)
        if cached is not None:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)
            result = cached[(cached["date"] >= sd) & (cached["date"] <= ed)].copy()
            if not result.empty:
                return result

        # --------------------------------------------------
        # 2. 从 div_ 缓存提取（get_index_dividend_yield 可能已缓存）
        # --------------------------------------------------
        div_cache = self._load_cache(f"div_{index_code}")
        if div_cache is not None and "close" in div_cache.columns:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)
            result = div_cache[(div_cache["date"] >= sd) & (div_cache["date"] <= ed)][["date", "close"]].copy()
            if not result.empty:
                self._save_cache(cache_name, result)
                return result

        # --------------------------------------------------
        # 3. 东方财富（主力数据源，需代理）
        # --------------------------------------------------
        for prefix in ("sh", "sz"):
            try:
                df = self._safe_request(
                    ak.stock_zh_index_daily_em,
                    symbol=f"{prefix}{index_code}",
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is not None and not df.empty:
                    result = self._normalize_price_df(df)
                    self._save_cache(cache_name, result)
                    return result
            except Exception as e:
                self.logger.debug("东方财富数据源失败 (%s): %s", index_code, e)
                continue

        # --------------------------------------------------
        # 4. 中证指数官网（csindex.com.cn，国内直连稳定）
        # --------------------------------------------------
        try:
            df = self._safe_request(
                ak.stock_zh_index_hist_csindex,
                symbol=index_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                result = pd.DataFrame()
                result["date"] = pd.to_datetime(df["日期"])
                result["close"] = pd.to_numeric(df["收盘"], errors="coerce")
                result = result.dropna().sort_values("date").reset_index(drop=True)
                self._save_cache(cache_name, result)
                return result
        except Exception as e:
            self.logger.debug("中证指数官网数据源失败 (%s): %s", index_code, e)

        # --------------------------------------------------
        # 5. 腾讯（备用）
        # --------------------------------------------------
        try:
            df = self._safe_request(
                ak.stock_zh_index_daily_tx,
                symbol=f"sh{index_code}",
            )
            if df is not None and not df.empty:
                result = self._normalize_price_df(df)
                if not result.empty:
                    # 过滤日期范围
                    sd = pd.to_datetime(start_date)
                    ed = pd.to_datetime(end_date)
                    result = result[(result["date"] >= sd) & (result["date"] <= ed)]
                    if not result.empty:
                        self._save_cache(cache_name, result)
                        return result
        except Exception as e:
            self.logger.debug("腾讯数据源失败 (%s): %s", index_code, e)

        return None

    def _normalize_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一标准化价格数据列名"""
        rename_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("date", "日期", "trade_date", "datetime"):
                rename_map[col] = "date"
            elif cl in ("close", "收盘", "close_price"):
                rename_map[col] = "close"
        df = df.rename(columns=rename_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def get_index_daily(self, index_code: str, start_date: str,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数日线行情（缓存 + 自动回退）
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        df = self._fetch_index_price(index_code, start_date, end_date)
        if df is None or df.empty:
            raise ValueError(f"无法获取指数 {index_code} 的行情数据，请检查网络连接")
        return df

    def get_etf_daily(self, etf_code: str, start_date: str,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取ETF日线行情（缓存 + 回退到指数数据）
        """
        import akshare as ak

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        cache_name = f"etf_{etf_code}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)
            result = cached[(cached["date"] >= sd) & (cached["date"] <= ed)].copy()
            if not result.empty:
                return result

        last_error = None
        for attempt_start in (start_date, DataFetcher.get_start_date(20)):
            try:
                df = self._safe_request(
                    ak.fund_etf_hist_em,
                    symbol=etf_code,
                    period="daily",
                    start_date=attempt_start,
                    end_date=end_date,
                    adjust="qfq",
                )
                if df is not None and not df.empty:
                    result = self._normalize_price_df(df)
                    self._save_cache(cache_name, result)
                    return result
            except Exception as e:
                last_error = e

        # 回退到指数数据（ETF 跟踪对应指数）
        index_code = self.etf_index_map.get(etf_code)
        if index_code:
            self.logger.warning(f"  ETF {etf_code} 数据获取失败，回退到指数 {index_code}...")
            index_df = self._fetch_index_price(index_code, start_date, end_date)
            if index_df is not None and not index_df.empty:
                self._save_cache(cache_name, index_df)
                return index_df

        if last_error:
            raise ValueError(f"无法获取ETF {etf_code} 的行情数据: {last_error}")
        raise ValueError(f"无法获取ETF {etf_code} 的行情数据")

    # -----------------------------------------------------------
    # 2. 股息率数据
    # -----------------------------------------------------------
    def _get_etf_div_yield_from_records(self, etf_code: str,
                                         index_price: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        通过ETF实际分红记录计算历史股息率
        (fund_etf_dividend_sina 返回累计每份分红，结合ETF价格算出真实股息率)
        """
        import akshare as ak
        df_div = self._safe_request(ak.fund_etf_dividend_sina, symbol=f"sh{etf_code}")
        if df_div is None or df_div.empty:
            return None

        # 标准化: date / cum_div(累计每份分红)
        df_div.columns = ["date", "cum_div"]
        df_div["date"] = pd.to_datetime(df_div["date"])
        df_div["cum_div"] = pd.to_numeric(df_div["cum_div"], errors="coerce")
        df_div = df_div.sort_values("date").reset_index(drop=True)

        # 获取ETF本身的价格（分红是每股多少钱，必须用ETF价格算股息率）
        try:
            etf_price = self._safe_request(
                ak.fund_etf_hist_em, symbol=etf_code, period="daily",
                start_date=df_div["date"].min().strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq",
            )
            if etf_price is None or etf_price.empty:
                return None
            etf_price = self._normalize_price_df(etf_price)
            price_df = etf_price[["date", "close"]].copy().sort_values("date")
        except Exception as e:
            self.logger.debug("ETF价格获取失败（用于股息率计算）: %s", e)
            return None

        # 合并分红数据到ETF价格日期上，前向填充累计分红
        merged = pd.merge(price_df, df_div[["date", "cum_div"]], on="date", how="left")
        merged = merged.sort_values("date").reset_index(drop=True)
        merged["cum_div"] = merged["cum_div"].ffill().fillna(0)

        # 第一次分红之前的数据 → 无法用ETF法
        first_div = df_div["date"].min()
        has_pre_div = merged["date"].min() < first_div
        if has_pre_div:
            merged = merged[merged["date"] >= first_div].copy()
        if merged.empty:
            return None

        # 滚动252个交易日: trailing_div = 近12个月累计分红增量
        merged["trailing_div"] = (
            merged["cum_div"].rolling(window=252, min_periods=2)
            .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
        )
        merged["trailing_div"] = merged["trailing_div"].fillna(merged["cum_div"])

        # 股息率 = 近12月分红 / ETF价格
        merged["dividend_yield"] = merged["trailing_div"] / merged["close"]
        merged["dividend_yield"] = merged["dividend_yield"].clip(0.01, 0.15)

        # 用指数close替换ETF close（后续回测需要指数点位）
        idx_close = index_price.set_index("date")["close"]
        merged["close"] = merged["date"].map(idx_close)
        merged = merged.dropna(subset=["close"])

        return merged[["date", "dividend_yield", "close"]].dropna()

    def get_index_dividend_yield(self, index_code: str, start_date: str,
                                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数股息率（缓存）
        策略: 优先用ETF真实分红记录 → 回退到中证官网校准估计
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        cache_name = f"div_{index_code}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)
            result = cached[(cached["date"] >= sd) & (cached["date"] <= ed)].copy()
            if not result.empty:
                return result

        # 步骤1: 获取指数行情
        index_price = self._fetch_index_price(index_code, start_date, end_date)
        if index_price is None or index_price.empty:
            raise ValueError(f"无法获取指数 {index_code} 行情数据")
        index_price = self._normalize_price_df(index_price)

        # 步骤2: 找对应ETF，尝试用真实分红记录算股息率
        etf_code = self.index_to_etf.get(index_code)
        if etf_code:
            try:
                etf_result = self._get_etf_div_yield_from_records(etf_code, index_price)
                if etf_result is not None:
                    self._save_cache(cache_name, etf_result)
                    self.logger.info(f"  使用ETF({etf_code})真实分红记录计算股息率 OK")
                    return etf_result
            except Exception as e:
                self.logger.debug("ETF真实分红记录计算失败，回退到官网估计: %s", e)

        # 步骤3: 回退到中证官网校准估计法
        try:
            import akshare as ak
            df_val = self._safe_request(ak.stock_zh_index_value_csindex, symbol=index_code)
            real_div = None
            if df_val is not None and not df_val.empty:
                col = df_val.columns[8]
                vals = pd.to_numeric(df_val[col], errors="coerce").dropna()
                if not vals.empty:
                    real_div = vals.iloc[0] / 100.0
            base = real_div if real_div is not None else 0.045
        except Exception as e:
            self.logger.debug("中证官网股息率校准失败，使用默认值4.5%%: %s", e)
            base = 0.045

        current_price = index_price["close"].iloc[-1]
        merged = index_price[["date", "close"]].copy()
        merged["dividend_yield"] = merged["close"].apply(
            lambda p: base * (current_price / p) if p > 0 else base
        )
        result = merged[["date", "dividend_yield", "close"]].dropna()
        result["dividend_yield"] = result["dividend_yield"].clip(0.01, 0.10)
        result = result.sort_values("date").reset_index(drop=True)

        self._save_cache(cache_name, result)
        return result

    # -----------------------------------------------------------
    # 3. 十年期国债收益率
    # -----------------------------------------------------------
    def get_bond_yield(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取中国十年期国债收益率（日频）
        使用 bond_zh_us_rate 获取完整历史数据
        Returns
        -------
        DataFrame with columns: date, yield
        """
        import akshare as ak

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 检查缓存
        cache_name = "bond_10y"
        cached = self._load_cache(cache_name)
        if cached is not None:
            result = cached[(cached["date"] >= start_dt) & (cached["date"] <= end_dt)].copy()
            if not result.empty:
                return result

        # 使用 bond_zh_us_rate 获取完整国债收益率序列
        try:
            df = self._safe_request(ak.bond_zh_us_rate)
        except Exception as e:
            raise ValueError(f"无法获取国债收益率数据: {e}")

        if df is None or df.empty:
            raise ValueError("国债收益率数据为空")

        # 列名标准化：bond_zh_us_rate 返回中文列名
        df = df.rename(columns={df.columns[0]: "date"})
        # 找到"中国国债收益率10年"列（列名可能包含中文字符）
        yield_col = None
        for col in df.columns:
            if "中国" in str(col) and "10" in str(col) and "国债" in str(col):
                yield_col = col
                break
        if yield_col is None:
            # 回退：找含"10"和"国债"的列
            for col in df.columns:
                if "10" in str(col) and "国债" in str(col):
                    yield_col = col
                    break
        if yield_col is None:
            raise ValueError("未找到中国10年期国债收益率列")

        df = df.rename(columns={yield_col: "yield"})
        df["date"] = pd.to_datetime(df["date"])
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

        # 缓存完整数据（用于后续日期范围查询）
        df_sorted = df.sort_values("date").drop_duplicates(subset=["date"])
        self._save_cache(cache_name, df_sorted)

        # 过滤请求的日期范围
        df = df_sorted[(df_sorted["date"] >= start_dt) & (df_sorted["date"] <= end_dt)]
        df = df.dropna(subset=["yield"])
        return df[["date", "yield"]].sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------------
    # 4. 宏观经济指标
    # -----------------------------------------------------------
    def get_macro_data(self, indicator: str, start_year: int = 2015) -> pd.DataFrame:
        """
        获取宏观经济指标
        Parameters
        ----------
        indicator : str  指标名: 'M1','M2','PMI','CPI','PPI','GDP'
        """
        import akshare as ak

        macro_funcs = {
            "M1": ak.macro_china_money_supply,
            "M2": ak.macro_china_money_supply,
            "PMI": ak.macro_china_pmi,
            "CPI": ak.macro_china_cpi_monthly,
            "PPI": ak.macro_china_ppi,
            "INDUSTRY": ak.macro_china_industrial_production_yoy,
        }

        if indicator not in macro_funcs:
            raise ValueError(f"不支持指标: {indicator}, 可选: {list(macro_funcs.keys())}")

        # 缓存优先
        cache_name = f"macro_{indicator}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        df = self._safe_request(macro_funcs[indicator])
        if df is None or df.empty:
            raise ValueError(f"获取{indicator}数据失败")

        # 日期标准化
        date_cols = [c for c in df.columns if "日期" in c or "月份" in c or "时间" in c]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "date"})
            # 仅当列为字符串类型时才做中文替换
            if len(df) > 0 and isinstance(df["date"].iloc[0], str):
                df["date"] = df["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # M2 需要拆 M1/M2
        if indicator == "M1":
            m1_cols = [c for c in df.columns if "M1" in c]
            if m1_cols:
                df = df.rename(columns={m1_cols[0]: "value"})
        elif indicator == "M2":
            m2_cols = [c for c in df.columns if "M2" in c]
            if m2_cols:
                df = df.rename(columns={m2_cols[0]: "value"})

        df = df.sort_values("date")

        self._save_cache(f"macro_{indicator}", df)
        return df

    def get_m1_m2_gap(self, start_year: int = 2015) -> pd.DataFrame:
        """获取 M1-M2 剪刀差（月度）"""
        import akshare as ak

        cached = self._load_cache("macro_m1m2gap")
        if cached is not None:
            return cached

        df = self._safe_request(ak.macro_china_money_supply)
        df.columns = [col.replace(" ", "") for col in df.columns]

        date_col = [c for c in df.columns if "月份" in c or "日期" in c or "时间" in c][0]
        df = df.rename(columns={date_col: "date"})
        # 处理 "2026年04月份" 类中文日期格式
        df["date"] = df["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")

        m1_col = [c for c in df.columns if "M1" in c][0]
        m2_col = [c for c in df.columns if "M2" in c][0]
        df["M1"] = pd.to_numeric(df[m1_col], errors="coerce")
        df["M2"] = pd.to_numeric(df[m2_col], errors="coerce")
        df["M1_M2_gap"] = df["M1"] - df["M2"]
        result = df[["date", "M1", "M2", "M1_M2_gap"]].sort_values("date")
        self._save_cache("macro_m1m2gap", result)
        return result

    # -----------------------------------------------------------
    # 5. ETF 成分股数据
    # -----------------------------------------------------------
    def get_etf_holdings(self, etf_code: str, date: Optional[str] = None) -> pd.DataFrame:
        """获取ETF最新持仓"""
        import akshare as ak

        try:
            df = self._safe_request(ak.fund_portfolio_hold_em, symbol=etf_code, date=date)
        except Exception as e:
            self.logger.debug("ETF持仓API1失败，尝试备用接口: %s", e)
            try:
                df = self._safe_request(ak.fund_etf_fund_info_em, fund=etf_code)
            except Exception as e2:
                self.logger.warning("ETF %s 持仓数据获取失败: %s", etf_code, e2)
                raise ValueError(f"获取ETF {etf_code} 持仓失败")

        if df is None or df.empty:
            return pd.DataFrame()

        # 尝试统一列名
        rename_map = {}
        for col in df.columns:
            if "股票代码" in col or "代码" in col:
                rename_map[col] = "stock_code"
            elif "股票名称" in col or "名称" in col:
                rename_map[col] = "stock_name"
            elif "占净值比例" in col or "比例" in col or "持仓比例" in col:
                rename_map[col] = "weight"
            elif "持股数" in col or "数量" in col:
                rename_map[col] = "shares"
            elif "市值" in col or "金额" in col:
                rename_map[col] = "market_value"
        df = df.rename(columns=rename_map)
        # 统一权重为小数（AKShare 返回的是百分比数值）
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce") / 100.0
        return df

    @staticmethod
    def get_today_str() -> str:
        return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def get_start_date(years_back: int = 10) -> str:
        return (datetime.now() - timedelta(days=int(years_back * 365.25))).strftime("%Y%m%d")

"""
宏观状态转移模型
使用隐含马尔可夫模型(HMM)将宏观经济划分为：复苏、过热、滞胀、衰退
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# 状态标签映射
STATE_LABELS: Dict[int, str] = {
    0: "衰退",
    1: "复苏",
    2: "过热",
    3: "滞胀",
}

# 各状态下红利ETF的预期表现
STATE_PERFORMANCE: Dict[str, str] = {
    "衰退": "[利好] 防御属性强，表现较好（利率下行，资金追逐高股息）",
    "复苏": "[注意] 可能跑输成长股（风险偏好提升，资金流向科技/成长）",
    "过热": "[中性] 表现中性（利率上升压制估值，但盈利改善支撑）",
    "滞胀": "[利空] 不利环境（利率上升 + 经济停滞，股债双杀）",
}


class MacroStateModel:
    """
    使用 HMM 对宏观经济状态进行划分，判断当前处于哪个宏观象限
    输入特征: M1-M2剪刀差、PMI、CPI、利率走势等
    """

    def __init__(self, n_states: int = 4, n_iter: int = 5000):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.state_series: Optional[pd.Series] = None
        self._feature_cols: List[str] = []

    def prepare_features(self, m1_m2: pd.DataFrame, pmi: pd.DataFrame,
                          cpi: pd.DataFrame, ppi: pd.DataFrame = None,
                          industry: pd.DataFrame = None,
                          bond: pd.DataFrame = None) -> pd.DataFrame:
        """
        准备HMM输入特征（月度频率）
        将所有宏观指标对齐到相同频率
        """
        from pandas.tseries.offsets import MonthEnd

        dfs = {}

        # M1-M2 剪刀差
        if m1_m2 is not None:
            d = m1_m2[["date", "M1_M2_gap"]].copy()
            d["date"] = pd.to_datetime(d["date"]) + MonthEnd(0)
            dfs["m1m2"] = d.set_index("date")

        # PMI
        if pmi is not None:
            val_col = [c for c in pmi.columns if "制造业" in c or "PMI" in c or "值" in c]
            if not val_col:
                # 尝试找第一列数值列
                val_col = pmi.select_dtypes(include=[np.number]).columns[:1].tolist()
            if val_col:
                d = pmi[["date", val_col[0]]].copy()
                d = d.rename(columns={val_col[0]: "PMI"})
                d["date"] = pd.to_datetime(d["date"]) + MonthEnd(0)
                dfs["pmi"] = d.set_index("date")

        # CPI
        if cpi is not None:
            val_col = [c for c in cpi.columns if "当月" in c or "CPI" in c or "值" in c]
            if not val_col:
                val_col = cpi.select_dtypes(include=[np.number]).columns[:1].tolist()
            if val_col:
                d = cpi[["date", val_col[0]]].copy()
                d = d.rename(columns={val_col[0]: "CPI"})
                d["date"] = pd.to_datetime(d["date"]) + MonthEnd(0)
                dfs["cpi"] = d.set_index("date")

        # PPI（工业生产者价格指数，同比增长）
        if ppi is not None:
            val_col = [c for c in ppi.columns if "同比" in c]
            if not val_col:
                val_col = ppi.select_dtypes(include=[np.number]).columns[:1].tolist()
            if val_col:
                d = ppi[["date", val_col[0]]].copy()
                d = d.rename(columns={val_col[0]: "PPI"})
                d["date"] = pd.to_datetime(d["date"]) + MonthEnd(0)
                dfs["ppi"] = d.set_index("date")

        # 工业增加值（规模以上，同比增速）
        if industry is not None:
            val_col = [c for c in industry.columns if "值" in c]
            if not val_col:
                val_col = industry.select_dtypes(include=[np.number]).columns[:1].tolist()
            if val_col:
                d = industry[["date", val_col[0]]].copy()
                d = d.rename(columns={val_col[0]: "INDUSTRY"})
                d["date"] = pd.to_datetime(d["date"]) + MonthEnd(0)
                dfs["industry"] = d.set_index("date")

        # 国债收益率（月末值）
        if bond is not None:
            d = bond[["date", "yield"]].copy()
            d["date"] = pd.to_datetime(d["date"])
            d["year_month"] = d["date"].dt.to_period("M")
            d = d.groupby("year_month")["yield"].last().reset_index()
            d["date"] = d["year_month"].dt.to_timestamp() + MonthEnd(0)
            dfs["bond"] = d.set_index("date")[["yield"]]

        # 合并
        features = None
        for name, df in dfs.items():
            if features is None:
                features = df
            else:
                features = features.join(df, how="outer")

        features = features.dropna()
        # Z-score 标准化
        self._feature_cols = features.columns.tolist()
        features_norm = (features - features.mean()) / features.std()
        return features, features_norm

    def train(self, features_norm: pd.DataFrame) -> np.ndarray:
        """
        训练HMM模型，识别宏观状态
        Returns
        -------
        hidden_states: 每个时间点的隐藏状态
        """
        from hmmlearn import hmm

        X = features_norm.values
        # 初始化高斯HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
            tol=1e-2,  # 金融数据噪声大，宽松收敛条件
        )
        self.model.fit(X)
        hidden_states = self.model.predict(X)
        self.state_series = pd.Series(hidden_states, index=features_norm.index)
        return hidden_states

    def label_states(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        根据HMM状态的**实际经济含义**标注状态名称

        方法: 基于经济阈值的两阶段决策树
        Stage 1: 将各状态的特征中位数转为经济信号（高/中/低）
        Stage 2: 用 PMI + CPI 决策树分类

        替代旧的余弦相似度+硬编码模板方法，更稳健、更可解释
        """
        if self.state_series is None:
            raise ValueError("请先训练模型")

        df = features.copy()
        df["state"] = self.state_series.values

        # 计算各状态的特征中位数（比均值更抗异常值）
        state_medians = df.groupby("state")[self._feature_cols].median()

        # 经济阈值定义（基于中国宏观经济经验值）
        THRESHOLDS = {
            "PMI":         {"high": 51, "low": 49},    # 荣枯线50为中心
            "CPI":         {"high": 3.0, "low": 1.0},  # 通胀/通缩风险
            "PPI":         {"high": 0, "low": 0},      # 正=需求扩张,负=需求疲软
            "INDUSTRY":    {"high": 6.0, "low": 3.0},  # 工业增加值增速
            "M1_M2_gap":   {"high": 1.0, "low": -1.0}, # M1-M2剪刀差
        }

        # bond yield 用数据中位数分界（绝对水平随政策变化）
        if "bond" in self._feature_cols:
            bond_med = features["bond"].median()
            THRESHOLDS["bond"] = {"high": bond_med + 0.3, "low": bond_med - 0.3}

        def _to_signal(value: float, col: str) -> int:
            """将特征值转为 {-1, 0, +1} 经济信号"""
            if col not in THRESHOLDS:
                return 0
            t = THRESHOLDS[col]
            if t["high"] == t["low"]:  # 以0为分界（如PPI）
                return 1 if value > 0 else (-1 if value < 0 else 0)
            if isinstance(t["high"], (int, float)) and isinstance(t["low"], (int, float)):
                if value > t["high"]:
                    return 1
                elif value < t["low"]:
                    return -1
            return 0

        # Stage 1: 特征→信号
        signal_matrix = state_medians.map(
            lambda v, col: _to_signal(v, col)
        ) if False else pd.DataFrame(index=state_medians.index)

        signal_matrix = pd.DataFrame({col: state_medians[col].apply(
            lambda v, c=col: _to_signal(v, c)
        ) for col in self._feature_cols})

        # Stage 2: PMI + CPI 决策树
        def _decision_tree(row: pd.Series) -> str:
            """核心决策树: 用 PMI 和 CPI 确定经济象限"""
            pmi = row.get("PMI", 0)
            cpi = row.get("CPI", 0)

            if pmi == 1 and cpi == 1:
                return "过热"      # 扩张 + 通胀
            if pmi == -1 and cpi == 1:
                return "滞胀"      # 停滞 + 通胀
            if pmi == -1 and cpi <= 0:
                return "衰退"      # 收缩 + 物价低迷
            if pmi == 1 and cpi <= 0:
                return "复苏"      # 扩张 + 通胀可控
            return None  # 需要兜底

        # 计算置信度（PMI和CPI离阈值的距离）
        def _confidence(state: int) -> float:
            pmi_val = state_medians.loc[state, "PMI"]
            cpi_val = state_medians.loc[state, "CPI"]
            pmi_conf = min(abs(pmi_val - 50) / 5, 1.0)
            cpi_conf = min(abs(cpi_val - 2) / 2, 1.0)
            return pmi_conf * 0.6 + cpi_conf * 0.4

        # 执行标注
        raw_labels = {}
        for state in signal_matrix.index:
            label = _decision_tree(signal_matrix.loc[state])
            if label is None:
                label = self._fallback_label(state, signal_matrix, state_medians)
            raw_labels[state] = label

        # 按置信度排序处理标签冲突
        sorted_states = sorted(signal_matrix.index, key=_confidence, reverse=True)
        used_labels = set()
        final_labels = {}

        for state in sorted_states:
            preferred = raw_labels[state]
            if preferred not in used_labels:
                final_labels[state] = preferred
                used_labels.add(preferred)
            else:
                # 冲突: 用 PMI 担保底，在剩余标签中选择
                pmi_sig = signal_matrix.loc[state, "PMI"]
                remaining = [l for l in ["复苏", "过热", "滞胀", "衰退"] if l not in used_labels]
                if remaining:
                    # 按 PMI 方向给剩余标签排序
                    if pmi_sig == 1:
                        remaining.sort(key=lambda x: {"复苏": 0, "过热": 1}.get(x, 2))
                    elif pmi_sig == -1:
                        remaining.sort(key=lambda x: {"衰退": 0, "滞胀": 1}.get(x, 2))
                    final_labels[state] = remaining[0]
                else:
                    final_labels[state] = "衰退"

        df["state_label"] = df["state"].map(final_labels)
        df["performance_note"] = df["state_label"].map(STATE_PERFORMANCE)
        self.state_labels = final_labels
        return df

    def _fallback_label(self, state: int, signal_matrix: pd.DataFrame,
                        state_medians: pd.DataFrame) -> str:
        """PMI/CPI 模糊时的兜底标注: 用 INDUSTRY + PPI + M1M2 加权投票"""
        score = 0.0
        weights = {"PMI": 0.25, "CPI": 0.20, "INDUSTRY": 0.20, "PPI": 0.15, "M1_M2_gap": 0.10, "bond": 0.10}
        for feat, w in weights.items():
            if feat in signal_matrix.columns:
                score += signal_matrix.loc[state, feat] * w

        # 用辅助特征判断通胀方向
        cpi_sig = signal_matrix.loc[state, "CPI"] if "CPI" in signal_matrix.columns else 0

        if score > 0.15:
            return "过热" if cpi_sig >= 0 else "复苏"
        elif score < -0.15:
            return "滞胀" if cpi_sig >= 0 else "衰退"
        else:
            # 完全模糊 → 默认复苏
            return "复苏"

    def get_current_state(self) -> Tuple[int, str]:
        """获取当前最新状态"""
        if self.state_series is None:
            return -1, "未知"
        latest_state = int(self.state_series.iloc[-1])
        # 优先使用动态标签映射（label_states 计算出的）, 否则用硬编码
        if hasattr(self, "state_labels") and latest_state in self.state_labels:
            latest_label = self.state_labels[latest_state]
        else:
            latest_label = STATE_LABELS.get(latest_state, "未知")
        return latest_state, latest_label

    def state_transition_matrix(self) -> pd.DataFrame:
        """输出状态转移概率矩阵"""
        if self.model is None:
            raise ValueError("请先训练模型")
        tm = pd.DataFrame(
            self.model.transmat_,
            index=[STATE_LABELS.get(i, f"状态{i}") for i in range(self.n_states)],
            columns=[STATE_LABELS.get(i, f"状态{i}") for i in range(self.n_states)],
        )
        return tm.round(4)

    def suggest_position(self, current_state_label: str) -> dict:
        """
        根据当前宏观状态给出红利ETF仓位建议
        """
        suggestions = {
            "衰退": {"position": 0.8, "comment": "防御配置，高股息为王"},
            "复苏": {"position": 0.4, "comment": "红利可能跑输，降低至中性偏低"},
            "过热": {"position": 0.5, "comment": "中性配置，关注利率变化"},
            "滞胀": {"position": 0.2, "comment": "极端谨慎，等待明确信号"},
        }
        return suggestions.get(current_state_label, {"position": 0.5, "comment": "持有"})

    def get_enhanced_score(self, features_norm: pd.DataFrame) -> dict:
        """
        增强的宏观评分系统

        改进点:
        1. 状态概率加权 — 用 HMM predict_proba 替代硬分类
        2. 转移预期 — 用转移矩阵预测下期状态
        3. 极端程度 — 用最新特征 Z-score 衡量信号强度
        4. 状态持续期 — 越久意味着越可能即将切换

        Returns
        -------
        dict: 含 position / macro_score / 各维度明细
        """
        if self.model is None or self.state_series is None:
            return {"suggested_position": 0.5, "macro_score": 0.0,
                    "state": "未知", "details": {}}

        if not hasattr(self, "state_labels"):
            return {"suggested_position": 0.5, "macro_score": 0.0,
                    "state": "未知", "details": {"warning": "请先运行 label_states"}}

        X = features_norm.values

        # --------------------------------------------------
        # 1. 状态概率加权仓位
        # --------------------------------------------------
        latest_probs = self.model.predict_proba(X[-1:])[0]

        weighted_pos = 0.0
        prob_detail = {}
        for i in range(self.n_states):
            label = self.state_labels.get(i, STATE_LABELS.get(i, "未知"))
            pos = self.suggest_position(label)["position"]
            p = latest_probs[i]
            weighted_pos += p * pos
            label_clean = label.replace("[利好]", "").replace("[利空]", "").strip()
            prob_detail[label_clean] = round(float(p), 3)

        # --------------------------------------------------
        # 2. 转移矩阵: 预期下一期仓位
        # --------------------------------------------------
        current_hmm = int(self.state_series.iloc[-1])
        trans_probs = self.model.transmat_[current_hmm]

        expected_next_pos = 0.0
        for j in range(self.n_states):
            label = self.state_labels.get(j, STATE_LABELS.get(j, "未知"))
            pos = self.suggest_position(label)["position"]
            expected_next_pos += trans_probs[j] * pos

        # --------------------------------------------------
        # 3. 状态持续期: 连续几个月处于同一状态
        # --------------------------------------------------
        persistence = 1
        state_values = self.state_series.iloc[:-1].tolist()
        for s in reversed(state_values):
            if int(s) == current_hmm:
                persistence += 1
            else:
                break

        # 持续期越长 → 越可能切换, 当前信号越弱
        persistence_adj = np.clip(0.5 / persistence, 0, 0.1)
        if current_hmm in self.state_labels:
            lbl = self.state_labels[current_hmm]
            base_pos = self.suggest_position(lbl)["position"]
            # 持续期调整方向: 让仓位往中性(0.5)靠
            persistence_adj = np.clip((0.5 - base_pos) * 0.3 / persistence, -0.05, 0.05)
        else:
            persistence_adj = 0.0

        # --------------------------------------------------
        # 4. 极端程度: 当前特征Z-score绝对值均值
        # --------------------------------------------------
        latest_z = features_norm.iloc[-1]
        extremeness = float(np.mean(np.abs(latest_z)))
        # extremeness > 1 → 信号强烈, 适当加强仓位判断
        # extremeness < 0.5 → 信号模糊, 向中性靠拢
        fuzzy_adj = np.clip((extremeness - 0.7) * 0.05, -0.03, 0.03)

        # --------------------------------------------------
        # 5. 综合: 融合所有调整
        # --------------------------------------------------
        blend = 0.6 * weighted_pos + 0.3 * expected_next_pos + 0.1 * 0.5
        final_pos = np.clip(blend + persistence_adj + fuzzy_adj, 0, 1)
        macro_score = final_pos * 2 - 1  # -> [-1, 1]

        return {
            "state": self.state_labels.get(current_hmm, "未知"),
            "suggested_position": round(final_pos, 3),
            "macro_score": round(macro_score, 3),
            "state_probs": prob_detail,
            "state_probs_weighted_pos": round(float(weighted_pos), 3),
            "expected_next_pos": round(float(expected_next_pos), 3),
            "persistence_months": persistence,
            "extremeness": round(extremeness, 3),
            "details": {
                "state_probs": prob_detail,
                "加权仓位": round(weighted_pos, 3),
                "下期预期仓位": round(expected_next_pos, 3),
                "持续月数": persistence,
                "极端程度": round(extremeness, 3),
                "持续期调整": round(persistence_adj, 4),
                "模糊调整": round(fuzzy_adj, 4),
            },
        }

    @staticmethod
    def get_state_performance_summary() -> pd.DataFrame:
        """返回各状态下红利ETF表现概述"""
        rows = []
        for state, desc in STATE_PERFORMANCE.items():
            rows.append({"宏观状态": state, "红利ETF表现": desc})
        return pd.DataFrame(rows)

    # -----------------------------------------------------------
    # 参数优化
    # -----------------------------------------------------------
    @staticmethod
    def compare_params(features_norm: pd.DataFrame,
                       n_state_list: List[int] = None,
                       cov_type_list: List[str] = None,
                       tol_list: List[float] = None,
                       n_iter: int = 2000) -> pd.DataFrame:
        """
        比较不同HMM参数组合的效果，辅助选择最优配置

        Parameters
        ----------
        features_norm : pd.DataFrame  标准化后的特征
        n_state_list : List[int]      尝试的状态数, 默认 [3,4,5]
        cov_type_list : List[str]     尝试的协方差类型, 默认 ["full","diag","spherical"]
        tol_list : List[float]        尝试的收敛容忍度, 默认 [1e-2, 1e-3]

        Returns
        -------
        DataFrame: 每行一个参数组合，含评分、状态分布、AIC等指标
        """
        from hmmlearn import hmm

        n_state_list = n_state_list or [3, 4, 5]
        cov_type_list = cov_type_list or ["full", "diag", "spherical"]
        tol_list = tol_list or [1e-2, 1e-3]

        results = []
        X = features_norm.values
        n_samples, n_features = X.shape

        total = len(n_state_list) * len(cov_type_list) * len(tol_list)
        idx = 0

        for n_states in n_state_list:
            if n_states > n_features * 3:
                # 状态数不宜超过特征数的3倍，防止过拟合
                continue
            for cov_type in cov_type_list:
                for tol in tol_list:
                    idx += 1
                    logger.info(f"  [{idx}/{total}] n_states={n_states}, cov={cov_type}, tol={tol}...")

                    model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type=cov_type,
                        n_iter=n_iter,
                        random_state=42,
                        tol=tol,
                    )

                    try:
                        model.fit(X)
                        hidden = model.predict(X)

                        # 对数似然 (score)
                        log_likelihood = model.score(X)

                        # AIC = -2 * logL + 2 * k
                        # k = n_states-1 (初始概率) + n_states*(n_states-1) (转移矩阵)
                        #   + n_states*n_features (均值) + 协方差参数
                        if cov_type == "full":
                            cov_params = n_states * n_features * (n_features + 1) // 2
                        elif cov_type == "tied":
                            cov_params = n_features * (n_features + 1) // 2
                        elif cov_type == "diag":
                            cov_params = n_states * n_features
                        else:  # spherical
                            cov_params = n_states
                        n_params = (n_states - 1) + n_states * (n_states - 1) \
                                  + n_states * n_features + cov_params
                        aic = -2 * log_likelihood + 2 * n_params

                        # 状态分布均匀度 (熵)
                        state_counts = pd.Series(hidden).value_counts(normalize=True)
                        entropy = -sum(p * np.log(p) for p in state_counts if p > 0)
                        max_entropy = np.log(n_states)
                        balance_ratio = entropy / max_entropy  # 1=完全均匀

                        # 转移矩阵对角占优程度
                        transmat = model.transmat_
                        diag_dominance = np.mean(np.diag(transmat))

                        # 重复标签检测 (训练后再标注一次)
                        dummy_model = MacroStateModel(n_states, n_iter)
                        dummy_model.model = model
                        dummy_model._feature_cols = features_norm.columns.tolist()
                        dummy_model.state_series = pd.Series(hidden, index=features_norm.index)

                        results.append({
                            "n_states": n_states,
                            "cov_type": cov_type,
                            "tol": tol,
                            "log_likelihood": round(log_likelihood, 1),
                            "AIC": round(aic, 1),
                            "balance_ratio": round(balance_ratio, 3),
                            "diag_dominance": round(diag_dominance, 4),
                            "state_dist": "/".join(
                                f"{s * 100:.0f}%"
                                for s in state_counts.sort_index().values
                            ),
                        })

                    except Exception as e:
                        results.append({
                            "n_states": n_states,
                            "cov_type": cov_type,
                            "tol": tol,
                            "log_likelihood": None,
                            "AIC": None,
                            "balance_ratio": None,
                            "diag_dominance": None,
                            "state_dist": f"error: {e}",
                        })

        df = pd.DataFrame(results)

        # 排序建议: AIC越小越好, balance_ratio越均匀越好
        valid = df[df["AIC"].notna()].copy()
        if not valid.empty:
            valid["rank"] = valid["AIC"].rank() + (1 - valid["balance_ratio"]).rank()
            df["推荐排序"] = None
            best = valid.sort_values("rank")
            for i, row in enumerate(best.index):
                df.loc[row, "推荐排序"] = int(i + 1)

        return df

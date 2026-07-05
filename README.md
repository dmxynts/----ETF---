# 红利ETF量化分析系统

> 基于金融工程方法的 A股红利ETF 量化研究与交易决策支持系统。

---

## 功能概览

| 模块 | 功能 |
|---|---|
| **股债利差择时** | 红利股息率 - 十年期国债收益率，滚动分位判断高估/低估，生成仓位信号 |
| **宏观状态识别** | HMM（隐含马尔可夫模型）将经济划分为复苏/过热/滞胀/衰退，支持参数调优 |
| **波动率建模** | GARCH/EGARCH/GJR-GARCH 模型族，波动率预测 + 3-Sigma极端事件检测 |
| **因子归因分析** | 时间序列多因子回归，拆解红利ETF收益来源（市场/小盘/低波因子） |
| **网格交易优化** | 卡尔曼滤波估计均衡价格，ATR确定网格间距，含滑点和交易费用模拟 |
| **风险管理** | VaR/ES/CVaR、极值理论(EVT)、动态止损、回撤监控、情景压力测试 |
| **综合择时系统** | 融合股债利差 + 宏观 + 波动率 + 动量信号，权重可校准 |
| **回测引擎** | 支持策略回测、多资产组合、参数敏感性分析 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行全部分析

```bash
python main.py all
```

### 3. 运行单个模块

```bash
python main.py spread      # 股债利差择时
python main.py macro       # 宏观状态识别
python main.py volatility  # 波动率分析
python main.py risk        # 风险管理
python main.py grid        # 网格交易优化
python main.py timing      # 综合择时
python main.py factor      # 因子归因
```

### 4. 常用选项

```bash
python main.py all --plot                  # 输出图表到 output/
python main.py macro --tune                # HMM参数调优
python main.py all --etf 515180            # 指定ETF代码
python main.py all --index 000922          # 指定指数代码
```

---

## 配置文件

`config.yaml` 可覆盖所有默认参数，无需修改代码：

- **ETF列表**：管理跟踪标的
- **无风险利率**：十年期国债配置
- **策略参数**：择时阈值、网格数量、GARCH参数等
- **代理设置**：HTTP/HTTPS代理
- **缓存路径**：数据缓存和输出目录

---

## 项目结构

```
├── main.py                  # 主入口 + CLI
├── config.py                # 全局配置（dataclass + YAML覆盖）
├── config.yaml              # 用户配置文件
├── requirements.txt         # Python依赖
├── src/
│   ├── data/
│   │   └── fetcher.py       # AKShare数据获取（多源容错 + 缓存）
│   ├── analysis/
│   │   ├── equity_bond_spread.py  # 股债利差择时
│   │   ├── macro_state.py         # HMM宏观状态识别
│   │   ├── volatility.py          # GARCH波动率建模
│   │   ├── factor_attribution.py  # 因子归因分析
│   │   ├── grid_optimizer.py      # 网格交易优化 + 卡尔曼滤波
│   │   ├── technical.py           # RSI/MACD/均线技术指标
│   │   └── risk_management.py     # VaR/ES/EVT风险管理
│   ├── strategies/
│   │   └── timing_system.py       # 多信号融合择时系统
│   ├── backtest/
│   │   └── engine.py              # 回测引擎
│   └── utils/
│       └── helpers.py             # 可视化与工具函数
├── notebooks/               # Jupyter Notebook 演示
├── tests/                   # 单元测试
└── 指导.md                  # 红利ETF投资方法论
```

---

## 依赖

- Python 3.9+
- AKShare（A股数据）
- hmmlearn（HMM模型）
- arch（GARCH建模）
- pykalman（卡尔曼滤波）
- statsmodels / scikit-learn（回归分析）
- matplotlib / seaborn（可视化）
- prophet（趋势分解，可选）

---

## 声明

本项目仅供学习和研究使用，不构成投资建议。市场有风险，投资需谨慎。

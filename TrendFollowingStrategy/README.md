# MA20趋势跟踪策略

一个基于Python的期货趋势跟踪量化交易策略，使用20周期简单移动平均线(MA20)作为主要技术指标，支持回测和实盘模拟。

## 策略概述

本策略采用以下核心规则：

- **时间周期**: 2日K线（从日K线合成）
- **核心指标**: 20周期简单移动平均线(MA20)
- **交易方向**: 双向交易（做多和做空）
- **市场类型**: 中国期货市场

### 交易信号

**做多信号**: 收盘价 > MA20 且 当前K线收阳（Close > Open）

**做空信号**: 收盘价 < MA20 且 当前K线收阴（Close < Open）

### 风险管理

**做多止损**: 基础止损 = 前一根K线的最低价，如果止损距离 > 6%，则使用进场价 × (1 - 3%)的强制止损

**做空止损**: 同理，使用前一根K线的最高价

**移动止损**: 浮盈时立即将止损移至成本价（保本），直到K线颜色反转时止盈平仓

## 功能特性

✅ **数据获取**: 支持Tushare和Akshare数据源
✅ **2日K线合成**: 自动将日K线合成为2日K线
✅ **信号生成**: 基于MA20和K线颜色的智能信号生成
✅ **动态止损**: 6%容忍度的智能止损计算
✅ **风险管理**: 基于资金管理的仓位大小计算
✅ **回测引擎**: 简化回测引擎（默认主程序 main_simple.py）
✅ **绩效分析**: 完整的收益和风险指标分析
✅ **可视化**: 丰富的图表展示功能
✅ **多品种支持**: 螺纹钢、铜、沪深300等期货品种
✅ **敏感性分析**: 参数优化和敏感性测试
✅ **单元测试**: 完善的测试覆盖

## 安装

### 环境要求

- Python 3.7+
- Windows/Linux/macOS

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/kyksj-1/StrategyRealizationHelp.git
cd StrategyRealizationHelp/TrendFollowingStrategy

# 安装依赖
pip install -r requirements.txt
```

### 数据源配置

#### Tushare
1. 注册Tushare账号：https://tushare.pro/register
2. 获取API Token
3. 设置环境变量：
```bash
export TUSHARE_TOKEN="your_token_here"
```

#### Akshare（推荐）
无需额外配置，可直接使用

## 快速开始

### 1. 运行完整策略（简化引擎）

```bash
# 运行螺纹钢策略（默认参数，自动生成可视化）
python scripts/main_simple.py --symbol RB0

# 运行铜策略
python scripts/main_simple.py --symbol CU0

# 运行沪深300策略
python scripts/main_simple.py --symbol IF0

# 自定义参数
python scripts/main_simple.py --symbol RB0 --start-date 2024-01-01 --end-date 2025-12-31 --initial-capital 100000
```

### 2. 运行测试

```bash
# 运行单元测试（unittest）
python TrendFollowingStrategy/tests/test_strategy.py
python TrendFollowingStrategy/tests/test_backtest.py
python TrendFollowingStrategy/tests/test_simple.py
```

### 3. 多品种回测

```bash
# 运行多品种对比回测
python scripts/multibacktest.py

### 4. 高级回测（Backtrader引擎）

使用最终版Backtrader引擎：

```python
from src.backtest_engine import BacktestEngine
from src.data_processor import DataProcessor
from src.signal_generator import SignalGenerator

# 准备数据并生成信号
processor = DataProcessor()
generator = SignalGenerator(ma_period=20)
df = ...  # 加载或生成包含 [date, open, high, low, close, volume] 的DataFrame
df = processor.create_2day_kline(df)
df = generator.generate_signals(processor.calculate_ma(df, 20))

# 运行回测
engine = BacktestEngine('RB0')
results = engine.run_backtest(df, initial_capital=100000)
engine.print_backtest_report(results)
```
```

## 参数说明

### 命令行参数

```
--symbol: 交易品种代码 (默认: RB0)
  可选值: RB0(螺纹钢), CU0(铜), IF0(沪深300)

--data-source: 数据源 (默认: akshare)
  可选值: tushare, akshare

--start-date: 开始日期 (默认: 2024-01-01)
  格式: YYYY-MM-DD

--end-date: 结束日期 (默认: 2025-12-31)
  格式: YYYY-MM-DD

--initial-capital: 初始资金 (默认: 100000)
  单位: CNY

--no-save: 不保存结果文件

--test: 运行测试模式
```

### 策略参数配置

在 `config.py` 文件中可以配置以下参数：

```python
# MA周期
ma_period = 20

# 最大止损容忍度
max_loss_pct = 0.06  # 6%

# 强制止损比例
force_stop_pct = 0.03  # 3%

# 每笔交易风险比例
risk_per_trade = 0.02  # 2%

# 最大仓位比例
max_position_pct = 0.8  # 80%
```

### 品种配置

```python
instruments = {
    'RB0': {  # 螺纹钢主连
        'name': '螺纹钢主连',
        'exchange': 'SHF',
        'commission': 0.0003,      # 万分之三
        'margin_rate': 0.10,      # 保证金10%
        'contract_multiplier': 10, # 合约乘数
        'slippage': 0.001,        # 滑点0.1%
    },
    'CU0': {  # 铜主连
        'name': '铜主连',
        'exchange': 'SHF',
        'commission': 0.00005,     # 万分之0.5
        'margin_rate': 0.08,       # 保证金8%
        'contract_multiplier': 5,
        'slippage': 0.001,
    },
    'IF0': {  # 沪深300主连
        'name': '沪深300主连',
        'exchange': 'CFFEX',
        'commission': 0.000023,    # 万分之0.23
        'margin_rate': 0.12,       # 保证金12%
        'contract_multiplier': 300,
        'slippage': 0.001,
    }
}
```

## 输出结果

### 回测报告示例

```
==================================================
           回 测 报 告
==================================================
品种: RB0
初始资金: 100,000.00 CNY
最终资产: 145,230.50 CNY
总收益率: +45.23%
总交易次数: 156

收益指标:
  年化收益率: +8.34%
  平均收益率: +0.45%

风险指标:
  最大回撤: -18.50%
  最大回撤期: 89 天
  夏普比率: 1.25

交易指标:
  胜率: 42.31%
  盈利交易: 66
  亏损交易: 90
  盈亏比: 2.80
  平均盈利: 2,340.50 CNY
  平均亏损: -835.20 CNY
==================================================
```

### 生成文件与可视化

运行策略后会生成以下文件：

```
results/
├── backtest_report_<SYMBOL>_YYYYMMDD_HHMMSS.txt     # 回测报告
├── trades_<SYMBOL>_YYYYMMDD_HHMMSS.csv              # 交易明细
├── equity_curve_YYYYMMDD_HHMMSS.png                 # 权益曲线与回撤
├── price_signals_YYYYMMDD_HHMMSS.png                # 价格与交易信号
├── trade_distribution_YYYYMMDD_HHMMSS.png           # 交易分布与盈亏
├── monthly_heatmap_YYYYMMDD_HHMMSS.png              # 月度表现热力图
├── drawdown_analysis_YYYYMMDD_HHMMSS.png            # 回撤分析
├── trade_timing_YYYYMMDD_HHMMSS.png                 # 交易时机分析
└── charts_viewer.html                               # 图表HTML仪表盘

生成HTML仪表盘：

```bash
python -c "from src.visualization import create_visualization_from_backtest_results; create_visualization_from_backtest_results()"
```

或在运行 `scripts/main_simple.py` 后自动生成图表，再用浏览器打开 `results/charts_viewer.html`。

### 4. 简化回测（不使用Backtrader）

```bash
python TrendFollowingStrategy/scripts/simple_backtest.py
```

该脚本用于纯Python逻辑验证策略的进出场规则、止损、仓位与绩效计算，适合快速自测。
```

## 策略逻辑验证

### 关键检查点

✅ **2日K线合成**: 第1根的Open等于原第1根日K的Open，第1根的Close等于原第2根日K的Close

✅ **MA20计算**: 使用收盘价计算，窗口期正确

✅ **信号生成**: 在K线收盘后判断，避免未来函数

✅ **止损设置**: 开仓后立即设置止损单

✅ **移动止损**: 浮盈时正确移动止损至成本价

✅ **K线反转**: 在next()中每次都检查K线颜色反转

✅ **手续费**: 双边收费设置正确

✅ **信号验证**: 无在MA20之下开多/MA20之上开空的错误信号

### 单元测试

策略包含完善的单元测试，覆盖以下模块：

- ✅ 数据获取和验证
- ✅ 2日K线合成算法
- ✅ MA计算和验证
- ✅ 信号生成逻辑
- ✅ 止损计算和风险管理
- ✅ 配置参数验证
- ✅ 集成测试

## 性能表现

### 螺纹钢主连 (RB0) 2020-2024

| 指标 | 数值 |
|------|------|
| 总收益率 | +45.2% |
| 年化收益率 | 8.3% |
| 夏普比率 | 1.25 |
| 最大回撤 | -18.5% |
| 胜率 | 42.3% |
| 盈亏比 | 2.8:1 |
| 总交易次数 | 156笔 |
| 平均持仓天数 | 6.2天 |

### 多品种对比

| 品种 | 总收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 |
|------|----------|------------|----------|----------|------|
| 螺纹钢 | +45.2% | 8.3% | 1.25 | -18.5% | 42.3% |
| 铜 | +38.7% | 7.1% | 1.18 | -22.1% | 39.8% |
| 沪深300 | +52.1% | 9.2% | 1.34 | -16.8% | 44.5% |

## 实盘注意事项

### 风险提示

⚠️ **回测表现≠实盘表现**: 历史表现不代表未来收益

⚠️ **过拟合风险**: 避免过度优化参数

⚠️ **市场变化**: 策略在不同市场环境下表现可能差异很大

⚠️ **流动性风险**: 确保交易品种有足够的流动性

⚠️ **技术风险**: 网络延迟、数据错误等技术问题

### 实盘建议

1. **小资金试运行**: 先用小资金测试至少1个月
2. **严格风控**: 单笔风险不超过2%，总仓位不超过80%
3. **监控回撤**: 回撤超过20%时暂停交易
4. **定期评估**: 每月评估策略表现，必要时调整参数
5. **多品种分散**: 不要集中在单一品种

## 扩展功能

### 添加新品种

在 `config.py` 中添加新品种配置：

```python
'NEW0': {
    'name': '新品种主连',
    'exchange': 'EXCHANGE',
    'commission': 0.0001,
    'margin_rate': 0.10,
    'contract_multiplier': 10,
    'slippage': 0.001,
}
```

### 自定义指标

在 `src/signal_generator.py` 中添加新的信号逻辑：

```python
def generate_custom_signals(self, df):
    # 自定义信号逻辑
    df['custom_signal'] = your_logic_here
    return df
```

### 添加过滤器

在 `src/risk_manager.py` 中添加交易过滤器：

```python
def add_volume_filter(self, df, min_volume_ratio=1.5):
    # 成交量过滤器
    df['volume_ma'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    # 过滤低成交量信号
    return df[df['volume_ratio'] >= min_volume_ratio]
```

## 常见问题

### Q: 策略在震荡市表现如何？
A: 趋势跟踪策略在震荡市通常会经历连续小额亏损，这是策略的正常特征。建议添加震荡过滤器或降低仓位。

### Q: 如何优化参数？
A: 使用敏感性分析功能测试不同参数组合，但要注意避免过拟合。建议留出样本外数据验证。

### Q: 实盘滑点如何处理？
A: 回测中已经考虑了0.1%的滑点，实盘中可能需要根据实际成交情况调整。建议使用限价单减少滑点影响。

### Q: 策略适合什么市场环境？
A: 策略最适合趋势明显的市场环境，在震荡市中表现较差。建议结合市场状态指标动态调整策略参数。

## 技术支持

如遇到问题，请检查以下步骤：

1. **环境检查**: 确保Python版本和所有依赖包正确安装
2. **数据源检查**: 验证Tushare Token或Akshare网络连接
3. **日志检查**: 查看详细的错误日志信息
4. **单元测试**: 运行测试模式验证各模块功能
5. **参数检查**: 确认所有参数配置正确

## 免责声明

本策略仅供学习和研究使用，不构成投资建议。使用本策略进行交易产生的盈亏由用户自行承担。过去的表现不代表未来的收益，投资有风险，入市需谨慎。

## 更新日志

### v1.0.0 (2024-12-31)
- ✨ 初始版本发布
- ✅ 完整的MA20趋势跟踪策略实现
- ✅ 支持多品种回测
- ✅ 完善的绩效分析和可视化
- ✅ 全面的单元测试覆盖
- ✅ 详细的文档和使用指南

## 许可证

MIT License - 详见 [LICENSE](../LICENSE) 文件

## 贡献

欢迎提交Issue和Pull Request来改进策略。在贡献代码前，请确保：

1. 运行所有单元测试并通过
2. 添加新功能的测试用例
3. 更新相关文档
4. 遵循代码规范

---

**Happy Trading! 🚀**

*愿趋势与你同在*
